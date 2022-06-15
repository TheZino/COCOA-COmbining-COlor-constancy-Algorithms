clear
clc

db_name = 'NUS'

load(['./AWB_estimations/' db_name '/folds.mat'])

tesz = size(te_split{:,1});
trsz = size(tr_split{:,1});

db_dim = trsz(2) + trsz(2);
val_percentage = 0.2;

%% Generating new splits for fair training

% Fold 0
msize = numel(tr_split{1});

te_size = numel(te_split{1});
val_size = (db_dim-te_size) * val_percentage;

tmp = tr_split{1};
val_0 = tmp(randperm(msize, floor(val_size)));

tr_0 = setdiff(tr_split{1}, val_0);

% Fold 1

tmp = tr_split{2};
msize = numel(tmp);

te_size = numel(te_split{2});
val_size = (db_dim-te_size) * val_percentage;

val_1 = tmp(randperm(msize, floor(val_size)));

tr_1 = setdiff(tr_split{2}, val_1);

% Fold 2

tmp = tr_split{3};
msize = numel(tmp);
te_size = numel(te_split{3});
val_size = (db_dim-te_size) * val_percentage;

val_2 = tmp(randperm(msize, floor(val_size)));

tr_2 = setdiff(tr_split{3}, val_2);

clear tr_split

tr_split = {};

tr_split{1,1} = tr_0;
tr_split{1,2} = tr_1;
tr_split{1,3} = tr_2;


val_split = {};

val_split{1,1} = val_0;
val_split{1,2} = val_1;
val_split{1,3} = val_2;

save(['./AWB_estimations/' db_name '/folds_fair.mat'], 'te_split', 'tr_split', 'val_split');
