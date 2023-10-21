#! /bin/bash

model_id=8
model_name=RgbScratch

for depth_type in "d435" "l515" "tof"
do

depth_type_id=3

if [ "$depth_type" = "tof" ]; then
    depth_type_id=2
elif [ "$depth_type" = "d435" ]; then
    depth_type_id=0
elif [ "$depth_type" = "l515" ]; then
    depth_type_id=1
else
    echo INVALID DEPTH TYPE ${depth_type}
    exit 1
fi


python main.py --dir_data /root/autodl-tmp/yiming/datasets/polar_hammer \
                --data_name HAMMER \
                --data_txt /root/autodl-tmp/yiming/ikemura_ws/PDNE/data_paths/hammer_MODE.txt \
                --gpus 7,6 \
                --loss 1.0*L1+1.0*L2 \
                --log_dir ./experiments/ \
                --save model-${model_id}-${depth_type} \
                --model ${model_name} \
                --completionformer_mode rgbd \
                --pre_pvt \
                --pre_res \
                --pretrained_completionformer /root/autodl-tmp/yiming/PDNE/pretrained/comp/NYUv2.pt \
                --test_only \
                --data_percentage 1.0 \
                --pretrain_list_file ./scripts/test/ckpt_list/model-${model_id}/model-${model_id}-single.txt \
                --use_single \
                --depth_type ${depth_type_id}

done