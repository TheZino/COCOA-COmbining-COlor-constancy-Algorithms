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

db='./data/SG568'

for frac in 4 8 16 32
do
	for seed in 123123 4321 5678
	do
		model_weights_dir='./model_weights/ensamble/SG568_1_'$frac'/'$seed'/'

		save_dir='./out/single_image/ensamble/1-'$frac'_db/'$seed'/'

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
done