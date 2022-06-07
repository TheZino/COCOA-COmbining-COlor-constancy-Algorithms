# Data generation

## File hierarchy


```

AWB_estimations
|
|- dataset_name
  |
  |- estimations.mat
  |- folds.mat
  |- folds_fairt.mat
  
```

### Estimations file

Must contains the six algorithms estimations named:
* est_GE1
* est_GE2
* est_GGW
* est_GW
* est_SoG
* est_WP

Each variables is a table of dimension n_imgs x 4 containing file_name and 
the three channels estimation

```
File                    GE1_1_1_6_R         GE1_1_1_6_G         GE1_1_1_6_B
'CubePlus_0001.tiff'	0.290113130051540	0.738692994890282	0.608413536233149
'CubePlus_0002.tiff'	0.393934798392941	0.776344504785880	0.492041242685894
'CubePlus_0003.tiff'	0.360493212193084	0.773847304263971	0.520773457125167
'CubePlus_0004.tiff'	0.344348971824716	0.779095564463262	0.523864378476845
...
```

### Fold file

fold.mat and fold_fair.mat files contains the fold divsion indexes to generate the dataset for training the model.
The files contains tr_split and te_split in unfair mode, while contains also val_split in fair mode.

```
te_split =

  1×3 cell array

    {1×n_imgs double}    {1×n_imgs double}    {1×n_imgs double}
```




## Files


* makegt.m : genertes gt file from original SG568 file list.
* make_folds : creates fold division (unfair) from zero for a given estimations file.
* make_fair_fold : generates a fair fold division for a fold division file.
* make_new_dataset.m : file for the creation of a dataset, given a estimation file and a fold division file
* data_preprocessing.m : used by make make_new_dataset.m
* fold_division.m : used by make make_new_dataset.m
* make_adv_dataset.m : for generating the advanced methods dataset.

