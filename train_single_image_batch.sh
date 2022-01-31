#!/bin/bash

clear

################################################################################
### Machine parameters

threads=4
gpu='cuda:0'

################################################################################
### Training parameters

for fract in 4 8 16 32
do

    db='./data/SG568_1-'$fract'/'

    epochs=3000
    input_ests=18

    ################################################################################
    ### Model details

    batch_size=8
    hlnum=4
    hlweights=(256 128 64)
    lr=0.003

    ################################################################################


    for seed in 123123 4321 5678
    do

        iddate=`date '+%Y%m%d_%H%M'`

        savedir='./experiments/single_image/ensamble/1-'$fract'_db/exp_HC_RGB_nG_'$iddate'_'$hlnum'hl_'$seed'/'

        time python3 ./source/train_single.py  \
        --epochs $epochs \
        --batch_size $batch_size \
        --threads $threads \
        --seed $seed \
        --lr $lr \
        --inest $input_ests \
        --hlnum $hlnum \
        --hlweights ${hlweights[@]} \
        --save_dir $savedir'/fold0/' \
        --trainset_dir $db/fold_tr0.csv \
        --validation_dir $db/fold_val0.csv \
        --device $gpu

        time python3 ./source/train_single.py  \
        --epochs $epochs \
        --batch_size $batch_size \
        --threads $threads \
        --seed $seed \
        --lr $lr \
        --inest $input_ests \
        --hlnum $hlnum \
        --hlweights ${hlweights[@]} \
        --save_dir $savedir'/fold1/' \
        --trainset_dir $db/fold_tr1.csv \
        --validation_dir $db/fold_val1.csv \
        --device $gpu

        time python3 ./source/train_single.py  \
        --epochs $epochs \
        --batch_size $batch_size \
        --threads $threads \
        --seed $seed \
        --lr $lr \
        --inest $input_ests \
        --hlnum $hlnum \
        --hlweights ${hlweights[@]} \
        --save_dir $savedir'/fold2/' \
        --trainset_dir $db/fold_tr2.csv \
        --validation_dir $db/fold_val2.csv \
        --device $gpu
    done
done