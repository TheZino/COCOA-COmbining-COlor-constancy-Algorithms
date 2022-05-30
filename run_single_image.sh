#!/bin/bash

################################################################################
### Machine parameters

threads=4
gpu='cuda:0'

################################################################################

inf=18
mode='onlyest'

### Model parameters
batch_size=8
hlnum=4
hlweights=(256 128 64)

################################################################################
################################################################################

db='./data/cube+/fast'

model_weights_dir='./model_weights/COCOA-HI-fast/SG568/'

save_dir='./out/single_image/test_cross_SGCube_fast/'

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f0.pth \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te0.csv \
--save_file $save_dir/fold0.csv \
--device $gpu

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f1.pth \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te1.csv \
--save_file $save_dir/fold1.csv \
--device $gpu

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f2.pth \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te2.csv \
--save_file $save_dir/fold2.csv \
--device $gpu


