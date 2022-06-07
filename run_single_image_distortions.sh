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

distortions=("awgn_s0" "awgn_s20" "awgn_s40" "blur_s0" \
             "blur_s1" "blur_s4" "jpeg_q10" "jpeg_q15" "jpeg_q60" \
             "awgn_s10" "awgn_s2.5" "awgn_s5" "blur_s0.5" \
             "blur_s2" "blur_s8" "jpeg_q100" "jpeg_q30" "jpeg_q90")

for dist_dir in ${distortions[*]}; do
    
    model_weights_dir='./model_weights/single_image/COCOA-HI/SG568/'

    db='./data/estimations_shigehler_distortions/'$dist_dir'/fast'
    save_dir='./out/single_image/SG568_distortions/fast_'$dist_dir'/'

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


done