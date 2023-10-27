#! /bin/bash
export CUDA_VISIBLE_DEVICES=8
# simplest completionformer with RGB input

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 8 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-10-1-1 \
                --model EarlyFusion \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --test_only \
                --data_percentage 0.15 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-10-1.txt \
                