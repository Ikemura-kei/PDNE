export CUDA_VISIBLE_DEVICES="4,5"

python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 4,5 \
                --loss 1.0*L1+1.0*L2 \
                --batch_size 14 \
                --epochs 100 \
                --log_dir ./experiments/ \
                --save model-22 \
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
                --use_cosine_loss \
                --data_percentage 1 \
                --test_only \
                --pol_rep leichenyang-7 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-22-2.txt