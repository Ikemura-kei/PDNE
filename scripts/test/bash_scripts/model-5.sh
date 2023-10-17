#! /bin/bash

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 3,0 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-5 \
                --model PromptFinetune \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --use_pol \
                --pol_rep grayscale-4 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --test_only \
                --data_percentage 1.0 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-5.txt