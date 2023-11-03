#! /bin/bash

# iun + rho + phi + vd

export CUDA_VISIBLE_DEVICES="1"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 1 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 11 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-30-narrow \
                --model ParidaEtAl \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.00105 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --data_percentage 1.0 \
                --camera_matrix_file /root/autodl-tmp/yiming/datasets/polar_hammer/scene3_traj1_1/intrinsics.txt \
                --test_only \
                --pretrain_list_file ./scripts/test/ckpt_list/model-30/model-30-narrow.txt \