#! /bin/bash

# simplest completionformer with RGB input

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 3,1 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-3 \
                --model POLAR-CAT \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.001005 \
                --use_pol \
                --pol_rep leichenyang-7 \
                --data_percentage 1.0 \
                # --resume \
                # --pretrain /root/autodl-tmp/yiming/ikemura_ws/PDNE_CONCAT/PDNE/experiments/230919_061840_model-3/model_00198.pt \