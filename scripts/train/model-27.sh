#! /bin/bash

# iun + rho + phi + vd

export CUDA_VISIBLE_DEVICES="3,6"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 3,6 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 7 \
                --epochs 85 \
                --log_dir ./experiments/ \
                --save model-27 \
                --model NormalDepthBranching \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.000805 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --use_norm \
                --normal_loss_weight 0.00175 \
                --pol_rep leichenyang-7 \
                --adaptive_norm_loss \
                --data_percentage 1.0 \
                --camera_matrix_file /root/autodl-tmp/yiming/datasets/polar_hammer/scene3_traj1_1/intrinsics.txt \
                # --resume --pretrain /root/autodl-tmp/yiming/ikemura_ws/MODEL-27/PDNE/experiments/231103_010617_model-27/model_00008.pt \