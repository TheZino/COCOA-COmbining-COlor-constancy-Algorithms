#!/bin/bash

clear

################################################################################
### Machine parameters

threads=0
gpu='cuda:0'

################################################################################
### Training parameters

db='./data/SG568/default'

epochs=1500
input_ests=18

################################################################################
### Model details

batch_size=8
hlnum=4
hlweights=(256 128 64)
lr=0.003

################################################################################


iddate=`date '+%Y%m%d_%H%M'`

savedir='./experiments/single_image/SG568/exp_default_'$iddate'_'$hlnum'hl/'


time python3 ./source/train_single.py  \
--epochs $epochs \
--batch_size $batch_size \
--threads $threads \
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
--lr $lr \
--inest $input_ests \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--save_dir $savedir'/fold2/' \
--trainset_dir $db/fold_tr2.csv \
--validation_dir $db/fold_val2.csv \
--device $gpu
