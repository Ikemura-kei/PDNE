#! /bin/bash

# iun + rho + phi + vd

export CUDA_VISIBLE_DEVICES="0,1"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 0,1 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-26 \
                --model CompletionFormerFinetuneNormDirect \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.00095 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --use_norm \
                --normal_loss_weight 0.000075 \
                --pol_rep leichenyang-7 \
                --camera_matrix_file /root/autodl-tmp/yiming/datasets/polar_hammer/scene3_traj1_1/intrinsics.txt \
                --resume --pretrain /root/autodl-tmp/yiming/ikemura_ws/MODEL-26/PDNE/experiments/231101_082211_model-26/model_00008.pt \