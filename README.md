# COCOA - COmbining COlor constancy Algorithms

Official implementation of the paper "COCOA: Combining Color Constancy Algorithms for Images and Videos".

Written in Pytorch v1.7.0

## Getting Started

### Required

 - Python3.6.7+
 - This project works with PyTorch and Torchvision, then please install it following the instructions at the page [PyTorch](http://pytorch.org/)

## Training

### Running the training

To change the training parameters check --h/--help

For training the single image model run
```
> ./train_single_image.sh

```

For training the video model run
```
> ./train_video.sh

```


### Train dataset configuration

The trainset must be a single csv file containing the name of the file,
the RGB values corresponding to the Ground Truth ant the RGB values corresponding to the input estimations.

The parameter input_ests refers to the number of inputs in the first layer, e.g.
if 6 estimations are collected, input_ests = 6\*3. Please change the parameter in
relation to your dataset.

Dataset csv example:
```
8D5U5525.png,0.7187906,1,0.6323975,0.7767372,1,0.6859942,0.7723222,1,0.5876065,0.7480156,1,0.6133124
8D5U5527.png,0.8144092,1,0.4818445,0.827463,1,0.4701669,0.8357095,1,0.4771664,0.8131861,1,0.4757451
8D5U5529.png,0.6546573,1,0.7029878,0.6457862,1,0.7189271,0.6468081,1,0.7245924,0.6751357,1,0.6947643
8D5U5531.png,0.726101,1,0.5617616,0.8017547,1,0.6110068,0.8091225,1,0.535962,0.7881969,1,0.5344098
8D5U5532.png,0.7457341,1,0.5887373,0.7643351,1,0.9477963,0.7220832,1,0.8587372,0.7570649,1,0.9033304
8D5U5533.png,0.623542,1,0.7158006,0.6388794,1,0.6928443,0.6504907,1,0.6988658,0.6871067,1,0.6601636
8D5U5536.png,0.5662869,1,0.8176366,0.6684254,1,0.7331354,0.6802689,1,0.717933,0.6432516,1,0.7385637
8D5U5537.png,0.5898304,1,0.7576272,0.7669067,1,0.5782619,0.7701685,1,0.4708538,0.7467304,1,0.5203703

```

A custom dataset can be passed to the training as an option.

e.g.

```
> python3 train_single_image.py --trainset_folder './datasets/data/train_set'
```


## Inference

### Pretrained model weights

[Google Drive](https://drive.google.com/drive/folders/1e5wqGLlHSPri72SpBVsMWwIrz4nDYn-C?usp=sharing)

### Testset configuration

The testset must be formatted in the same way as the training and validation set.

### Running the code

Two sh files are provided with the calls to test the model in single image and video cases.
The single image runs the model trained on the Shi-Gehler reprocessed and the video one 
tests the video model trained on BCC.

To change the testset and the model, please change the corresponding parameters in the sh call.

## Cite

If you use the code provided in this repository please cite our original work:
```
@article{zini2022cocoa,
  title={COCOA: Combining Color Constancy Algorithms for Images and Videos},
  author={Zini, Simone and Buzzelli, Marco and Bianco, Simone and Schettini, Raimondo},
  journal={IEEE Transactions on Computational Imaging},
  year={2022},
  publisher={IEEE}
}
```
