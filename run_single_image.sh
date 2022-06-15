#!/bin/bash

################################################################################
### Machine parameters

threads=4
gpu='cuda:0'

################################################################################

inf=3
mode='onlyest'

### Model parameters
batch_size=8
hlnum=4
hlweights=(256 128 64)

################################################################################
################################################################################


db_name='SG568'
version='default'
model='SG568_red5'
red=5

db='./data/'$db_name'/'$version

echo $db_name

if [ "$version" = "fast" ]; then
    model_weights_dir='./model_weights/single_image/COCOA-HI-fast/'$model'/'
else
    model_weights_dir='./model_weights/single_image/COCOA-HI/'$model'/'
fi

echo $model_weights_dir


if [ "$db_name" = "$model" ]; then
    save_dir='./out/single_image/'$db_name'/'$version'/'
else
    save_dir='./out/single_image/'$db_name'/'$version'_'$model'/'
fi

echo $save_dir

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f0.pth \
--reduced $red \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te0.csv \
--save_file $save_dir/fold0.csv \
--device $gpu

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f1.pth \
--reduced $red \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te1.csv \
--save_file $save_dir/fold1.csv \
--device $gpu

python3 ./source/infer_single.py \
--model $model_weights_dir/model_singleimage_f2.pth \
--reduced $red \
--infeat $inf \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--in_dir $db/fold_te2.csv \
--save_file $save_dir/fold2.csv \
--device $gpu


