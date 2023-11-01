#! /bin/bash

# iun + rho + phi + vd

export CUDA_VISIBLE_DEVICES="6"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 6 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 100 \
                --log_dir ./experiments/ \
                --save model-15-narrow \
                --model PromptFinetuneNorm \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.00105 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --use_norm \
                --pol_rep leichenyang-7 \
                --test_only \
                --data_percentage 1.0 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-15/model-15-narrow.txt