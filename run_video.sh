#!/bin/bash

################################################################################
### Machine parameters

threads=4
gpu='cuda:0'

################################################################################
### Training parameters

db='./data/BCC/fold_te.csv'

epochs=3000
inf=18

################################################################################
### Model details

batch_size=8
hlnum=3
hlweights=(256 128)

################################################################################
################################################################################

model='./model_weights/BCC/model_video_tmp.pth'

hfeat=64
mode='lstm'

save_file='./out/video//combo_est_bcc.csv'

python3 ./source/infer_video.py \
--model $model \
--infeat $inf \
--hiddenfeat $hfeat \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db \
--save_file $save_file \
--mode $mode \
--device $gpu
