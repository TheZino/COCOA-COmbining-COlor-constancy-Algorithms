clear
clc

%% options

norm_type = 'nG'; %'nG', 'nmax'

fast = false;

dataset = 'NUS';


%% save name preparation

db_name = 'default';
if fast
    db_name = 'fast';
end
save_file = ['./dts/' dataset '/' db_name '.csv'];
if ~exist(['./dts/' dataset '/'], 'dir')
       mkdir(['./dts/' dataset '/'])
end

%% data loading

load(['./AWB_estimations/' dataset '/folds_fair.mat']);
GT = readtable(['./AWB_estimations/' dataset '/GT_' dataset '.csv']);
load(['./AWB_estimations/' dataset '/est_baseawbbest.mat']);
if fast
    load(['./AWB_estimations/' dataset '/est_baseawb_fast.mat']);
end
methods = {est_SoG, est_GE1, est_GE2, est_GGW, est_GW, est_WP};

%% Make dataset

% Data setup

names = GT{:,1};
mask = 1:length(names);
names = names(mask);
GT = GT{mask,2:4};

for ii = 1:length(methods)
    methods{ii} = methods{ii}{mask,2:4};    
end

switch norm_type
    case 'nG'
        GT= GT ./ GT(:,2);
        for ii = 1:length(methods)
            methods{ii} = methods{ii} ./ methods{ii}(:,2);
        end
    case 'nmax'
        GT = GT ./ max(GT,[], 2); 
        for ii = 1:length(methods)
            methods{ii} = methods{ii} ./ max(methods{ii}, [], 2);
        end
    otherwise
        error('normalization type not valid!')
end

% Data creation
fid = fopen( save_file, 'w' );
for jj = 1 : length(GT)
    fprintf( fid, '%s,%d,%d,%d', names{jj}, ...
                      GT(jj, 1), GT(jj,2), GT(jj,3));
    for ii = 1:length(methods)
        fprintf( fid, ',%d,%d,%d', methods{ii}(jj,1), methods{ii}(jj,2), methods{ii}(jj,3));
    end
    fprintf( fid, '\n');
end
fclose( fid );



%% Division by fold

data = readtable(save_file);

out_dir = ['./fold_division/' dataset '/' db_name];
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end


for i=0:2
    tr_split{i+1} = sort(tr_split{i+1});
    idx = tr_split{i+1};

    % train
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_tr', num2str(i) '.csv'], 'WriteVariableNames',0)

    % validation
    val_split{i+1} = sort(val_split{i+1});
    idx = val_split{i+1};
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_val', num2str(i) '.csv'], 'WriteVariableNames',0)

    % test
    te_split{i+1} = sort(te_split{i+1});
    idx = te_split{i+1};
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_te', num2str(i) '.csv'], 'WriteVariableNames',0)
end

