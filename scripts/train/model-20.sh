#! /bin/bash

# iun + rho + phi + vd
export CUDA_VISIBLE_DEVICES="2,3"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 2,3 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 250 \
                --log_dir ./experiments/ \
                --save model-20 \
                --model PromptFinetuneV2 \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --save_full \
                --warm_up \
                --lr 0.00105 \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --use_pol \
                --pol_rep leichenyang-7 \
                --resume --pretrain /root/autodl-tmp/yiming/ikemura_ws/PDNE_PROMPT_FINETUNE_V2/PDNE/experiments/231026_010750_model-20/model_00056.pt