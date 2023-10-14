#! /bin/bash

# simplest completionformer with RGB input

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 5,7 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-6 \
                --model RgbFinetune \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.00105 \
                --num_threads 16 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --resume --pretrain /root/autodl-tmp/yiming/ikemura_ws/PDNE_RGB_FINETUNE/PDNE/experiments/231014_184054_model-6/model_00026.pt               