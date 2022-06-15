clear
clc

db_name = 'NUS'

%% Generating the fold division from zero

db = readtable(['./AWB_estimations/'  db_name '/estimations/est_illuminants_GE1.txt']);

fold_length = floor(height(db) / 3);

indexes = 1:height(db);

indexes = indexes(randperm(length(indexes)));

f1 = indexes(1:fold_length);
f2 = indexes(fold_length+1:fold_length*2);
f3 = indexes(fold_length*2+1:length(indexes));

f1 = sort(f1);
f2 = sort(f2);
f3 = sort(f3);


tr_split{1} = [f1, f2];
te_split{1} = f3;

tr_split{2} = [f2, f3];
te_split{2} = f1;

tr_split{3} = [f1, f3];
te_split{3} = f2;


save(['./AWB_estimations/' db_name '/folds.mat'], 'te_split', 'tr_split');

%% tcc dataset
% clear
% clc

% load(['./AWB_estimations/' db_name '/est_illuminants_train_base.mat'])

% db_len = 400;

% val_len = 80;

% indexes = randperm(400);

% indexes = string(sort(indexes(1:80)));

% nms = split(est_GE1{:,1}, '/');
% nms = nms(:,1);

% mask = zeros(length(nms),1);
% for ii =1:length(indexes)
%     mask = mask + strcmp(nms, indexes{ii});
% end

% val_split = find(mask==1);

% tr_idxs = 1:height(est_GE1);

% tr_split = setdiff(tr_idxs, val_split);
% val_split = val_split';

% save('./AWB_estimations/tcc/folds.mat', 'val_split', 'tr_split');
