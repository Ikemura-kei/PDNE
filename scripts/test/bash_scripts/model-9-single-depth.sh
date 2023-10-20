#! /bin/bash

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 7,8 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-9-single-depth-tof \
                --model CompletionFormerFreezed \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --test_only \
                --data_percentage 1.0 \
                --use_single \
                --depth_type 2 \
                --pretrain /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt 