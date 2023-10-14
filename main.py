"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import apex
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
from losses.l1l2loss import L1L2Loss

from datasets.hammer import HammerDataset
from datasets.hammer_old import HammerDatasetOld

from utils.mics import save_output
from utils.metrics import PDNEMetric
from torch.utils.tensorboard import SummaryWriter
from model.PDNE import PDNE
from utils import train_utils
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import json
from config import args as args_config
import time
import random
import os

# -- models --
from model.completionformer_original.completionformer import CompletionFormer
from model.completionformer_vpt_v1.completionformer_vpt_v1 import CompletionFormerVPTV1
from model.completionformer_vpt_v2.completionformer_vpt_v2 import CompletionFormerVPTV2
from model.completionformer_vpt_v2.completionformer_vpt_v2_1 import CompletionFormerVPTV2_1
from model.completionformer_prompt_finetune.completionformer_prompt_finetune import CompletionFormerPromptFinetune
from model.completionformer_rgb_finetune.completionformer_rgb_finetune import CompletionFormerRgbFinetune
from model.completionformer_polar_cat.completionformer import CompletionFormerPolarCat

from summary.cfsummary import CompletionFormerSummary
from metric.cfmetric import CompletionFormerMetric
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

torch.autograd.set_detect_anomaly(True)

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_rmse = 100
best_mae = 100

# Minimize randomness

def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain, map_location='cpu')

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.layer0=False
            new_args.save_freq = 2

    return new_args


def train(gpu, args):
    global best_rmse
    global best_mae

    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                            world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)


    # Prepare dataset
    dataset = HammerDataset(args, "train")

    sampler_train = DistributedSampler(
        dataset, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size

    loader_train = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)

    # Network
    if args.model == 'CompletionFormer':
        net = CompletionFormer(args)
    elif args.model == 'CompletionFormerFreezed':
        net = CompletionFormer(args)
        if args.pretrained_completionformer is not None:
            net.load_state_dict(torch.load(args.pretrained_completionformer)['net'])
            for p in net.parameters():
                p.requires_grad = False
    elif args.model == 'PDNE':
        net = PDNE(args)
    elif args.model == 'VPT-V1':
        net = CompletionFormerVPTV1(args)
    elif args.model == 'VPT-V2':
        net = CompletionFormerVPTV2_1(args)
    elif args.model == 'PromptFinetune':
        net = CompletionFormerPromptFinetune(args)
    elif args.model == 'RgbFinetune':
        net = CompletionFormerRgbFinetune(args)
    else:
        raise TypeError(args.model, ['CompletionFormer', 'PDNE', 'VPT-V1', 'PromptFintune', 'VPT-V2', 'RgbFinetune'])

    net.cuda(gpu)

    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))

    # Loss
    loss = L1L2Loss(args)
    loss.cuda(gpu)

    # Optimizer
    optimizer, scheduler = train_utils.make_optimizer_scheduler(args, net, len(loader_train))
    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)
    
    init_epoch = 1

    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                print('Resume:', args.resume)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])
                    init_epoch = checkpoint['epoch']

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    net = DDP(net)

    metric = PDNEMetric(args)

    if gpu == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/train', exist_ok=True)
        writer_train = SummaryWriter(log_dir=args.save_dir + '/' + 'train')
        writer_val = SummaryWriter(log_dir=args.save_dir + '/' + 'val')
        total_losses = np.zeros(np.array(loss.loss_name).shape)
        total_metrics = np.zeros(np.array(metric.metric_name).shape)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    for epoch in range(init_epoch, args.epochs+1):
        # Train
        print("--> Epoch {}/{}".format(epoch, args.epochs))
        net.train()

        sampler_train.set_epoch(epoch)
        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

        num_sample = len(loader_train) * \
            loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        init_seed(seed=int(time.time()))

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if (val is not None) and key != 'base_name'}

            sample["input"] = sample["rgb"]

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                        * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)
            
            output['pred'] = output['pred']  * sample['net_mask']
            sample['gt'] = sample['gt'] 

            loss_sum, loss_val = loss(sample, output)
                
            loss_sum_norm=0

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            if gpu == 0:
                for i in range(len(loss.loss_name)):
                    total_losses[i] += loss_val[0][i]

                log_cnt += 1
                log_loss += loss_sum.item()

                e_string = f"{(log_loss/log_cnt):.2f}"
                if batch % args.print_freq == 0:
                    pbar.set_description(e_string)
                    pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            if epoch % 2 == 0:
                # -- save visualization --
                folder_name = os.path.join(args.save_dir, "epoch-{}".format(str(epoch)))
                os.makedirs(folder_name, exist_ok=True)
                rand_idx = np.random.randint(0, args.batch_size)

                def depth_to_colormap(depth, max_depth):
                    npy_depth = depth.detach().cpu().numpy()[0]
                    vis = ((npy_depth / max_depth) * 255).astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    return vis

                out = depth_to_colormap(output["pred"][rand_idx], 2.6)
                gt = depth_to_colormap(sample["gt"][rand_idx], 2.6)
                sparse = depth_to_colormap(sample["dep"][rand_idx], 2.6)

                cv2.imwrite(os.path.join(folder_name, "out.png"), out)
                cv2.imwrite(os.path.join(folder_name, "sparse.png"), sparse)
                cv2.imwrite(os.path.join(folder_name, "gt.png"), gt)

            for i in range(len(loss.loss_name)):
                writer_train.add_scalar(
                    loss.loss_name[i], total_losses[i] / len(loader_train), epoch)

            writer_train.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            if ((epoch) % args.save_freq == 0) or epoch==5 or epoch==args.epochs:
                if args.save_full or epoch == args.epochs:
                    state = {
                        'net': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }
                else:
                    state = {
                        'net': net.module.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }

                torch.save(
                    state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))
                
        scheduler.step()

        if gpu == 0:
            total_losses = np.zeros(np.array(loss.loss_name).shape)
            total_metrics = np.zeros(np.array(metric.metric_name).shape)

    if gpu == 0:
        writer_train.close()
        writer_val.close()

def load_pretrain(args, net, ckpt):
    assert os.path.exists(ckpt), \
            "file not found: {}".format(ckpt)

    checkpoint = torch.load(ckpt, map_location='cpu')
    key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

    if key_u:
        print('Unexpected keys :')
        print(key_u)

    if key_m:
        print('Missing keys :')
        print(key_m)
        raise KeyError

    print('Checkpoint loaded from {}!'.format(ckpt))

    return net

def test_one_model(args, net, loader_test, save_samples, epoch_idx=0, summary_writer=None, is_old=False):
    net = nn.DataParallel(net)

    metric = CompletionFormerMetric(args)

    vis_dir = os.path.join(args.save_dir, 'test', 'visualization')
    try:
        os.makedirs(vis_dir, exist_ok=True)
        result_file = open(os.path.join(args.save_dir, 'test', 'results.txt'), 'w')
    except OSError:
        pass

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    total_metrics = None
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        if is_old:
            sample['gt'] = sample['gt'] / 1000.0
            sample['dep'] = sample['dep'] / 1000.0
            output['pred'] = output['pred'] / 1000.0

        metric_val = metric.evaluate(sample, output, 'test')

        if total_metrics is None:
            total_metrics = metric_val[0]
        else:
            total_metrics += metric_val[0]

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)

        if batch in save_samples:
            dep = sample['dep'] # in m
            gt = sample['gt'] # in m
            pred = output['pred'] # in m

            def depth2vis(depth, MAX_DEPTH=2.15):
                depth = depth.detach().cpu().numpy()[0].transpose(1,2,0)
                vis = ((depth / MAX_DEPTH) * 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                return vis

            gt_vis = depth2vis(gt, 2.15)
            dep_vis = depth2vis(dep, 2.15)
            pred_vis = depth2vis(pred, 2.15)
            # -- error map --
            gt_mask = gt.detach().cpu().numpy()[0].transpose(1,2,0)
            gt_mask[gt_mask <= 0.001] = 0
            err = torch.abs(pred-gt)
            print("--> Min err {} max err {}".format(torch.min(err), torch.max(err)))
            error_map_vis = depth2vis(err, 0.55)
            error_map_vis[np.tile(gt_mask, (1,1,3))==0] = 0

            os.makedirs(os.path.join(vis_dir, 'e{}'.format(epoch_idx)), exist_ok=True)
            cv2.imwrite(os.path.join(vis_dir, 'e{}'.format(epoch_idx), 's{}_gt.png'.format(batch)), gt_vis)
            cv2.imwrite(os.path.join(vis_dir, 'e{}'.format(epoch_idx), 's{}_err.png'.format(batch)), error_map_vis)
            cv2.imwrite(os.path.join(vis_dir, 'e{}'.format(epoch_idx), 's{}_pred.png'.format(batch)), pred_vis)
            cv2.imwrite(os.path.join(vis_dir, 'e{}'.format(epoch_idx), 's{}_dep.png'.format(batch)), dep_vis)

    pbar.close()

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    metric_avg = total_metrics / num_sample
    if summary_writer is not None:
        for i, metric_name in enumerate(metric.metric_name):
            summary_writer.add_scalar('test/{}'.format(metric_name), metric_avg[i], epoch_idx)

    return metric_avg

def test(args):

    # -- prepare network --
    is_old = False
    if args.model == 'VPT-V1':
        pass
    elif args.model == 'VPT-V2':
        net = CompletionFormerVPTV2(args)
    elif args.model == 'CompletionFormer':
        pass
    elif args.model == 'PDNE':
        pass
    elif args.model == 'CompletionFormerFreezed':
        pass
    elif args.model == 'PromptFinetune':
        net = CompletionFormerPromptFinetune(args)
    elif args.model == 'POLAR-CAT':
        is_old = True
        net = CompletionFormerPolarCat(args)
    else:
        raise TypeError(args.model, ['CompletionFormer', 'PDNE', 'VPT-V1', 'CompletionFormerFreezed', 'VPT-V2', 'POLAR-CAT', 'PromptFinetune'])

    # -- prepare dataset --
    data_test = HammerDataset(args, 'test') if not is_old else HammerDatasetOld(args, 'test')
    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    net.to(0)

    if args.pretrain is not None:
        net = load_pretrain(args, net, args.pretrain)
        save_samples = np.random.randint(0, len(loader_test), 10)
        test_one_model(args, net, loader_test, save_samples, is_old=is_old)

    elif args.pretrain_list_file is not None:
        summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'test', 'logs'))

        pretrain_list = open(args.pretrain_list_file, 'r').read().split("\n")
        num_samples_to_save = 3 if len(pretrain_list) >= 5 else 50
        if len(pretrain_list) == 1:
            num_samples_to_save = int(len(loader_test) / 6.0)
        save_samples = np.random.randint(0, len(loader_test), num_samples_to_save)

        for line in pretrain_list:
            epoch_idx = line.split(" - ")[0]
            ckpt = line.split(" - ")[1]
            net = load_pretrain(args, net, ckpt)
            test_one_model(args, net, loader_test, save_samples, epoch_idx, summary_writer, is_old=is_old)

def main(args):
    init_seed()
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        args.pretrain = '{}/model_best.pt'.format(args.save_dir)

    test(args)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
    os.environ["MASTER_ADDR"] = args_config.address
    os.environ["MASTER_PORT"] = args_config.port
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
    time.sleep(5)

    main(args_main)
