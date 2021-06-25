#!/bin/bash

# --model_chkp ./experiments/11_lstm/exp_HC_RGB_nG_20210610_1417_3hl_lstm_64_3_baseline/checkpoints/netComboNN_epochbest.pth \
clear

################################################################################
### Machine parameters

threads=4
gpu='cuda:0'

################################################################################
### Training parameters

db='./data/BCC/'

epochs=3000
inest=18

################################################################################
### Model details

batch_size=8
hlnum=3
hlweights=(256 128)

lr=0.0001

################################################################################
################################################################################

iddate=`date '+%Y%m%d_%H%M'`

hfeat=64

savedir='./experiments/video/exp_HC_RGB_nG_'$iddate'_'$hlnum'hl_'$hfeat


time python3 ./source/train_video.py  \
--epochs $epochs \
--batch_size $batch_size \
--threads $threads \
--lr $lr \
--wd \
--infeat $inest \
--hfeat $hfeat \
--hlnum $hlnum \
--hlweights ${hlweights[@]} \
--save_dir $savedir \
--trainset_dir $db/fold_tr.csv \
--validation_dir $db/fold_val.csv \
--device $gpu
