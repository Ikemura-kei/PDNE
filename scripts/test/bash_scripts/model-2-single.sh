#! /bin/bash

model_id=2
model_name=CompletionFormerFreezed

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 8,9 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-${model_id}-single \
                --model ${model_name} \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --test_only \
                --data_percentage 1.0 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-${model_id}-single.txt \