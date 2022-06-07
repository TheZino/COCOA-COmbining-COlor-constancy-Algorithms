clear
clc

%% options

norm_type = 'nG'; % Estimation normalization: 'nG', 'nmax'

fast_db = true;

db_dir = './AWB_estimations/estimations_shigehler_distortions/';

%% fold division data

load('./AWB_estimations/sg568/folds_fair.mat');

        
%% data to use

versions = { 'awgn_s0', 'awgn_s20', 'awgn_s40', 'blur_s0', ... 
             'blur_s1', 'blur_s4', 'jpeg_q10', 'jpeg_q15', 'jpeg_q60', ...
             'awgn_s10', 'awgn_s2.5', 'awgn_s5', 'blur_s0.5', ...
             'blur_s2', 'blur_s8', 'jpeg_q100', 'jpeg_q30', 'jpeg_q90'};

for gg = 1:length(versions)

    GT = readtable('./AWB_estimations/sg568/GT_Shi_reprocessed.csv');

    db_dir = [db_dir '/'];
    
    db_name = 'default';
    
    if fast_db
        db_name = 'fast';
    end

    disp(db_name)
    
    save_file = ['./dts/' db_dir '/' versions{gg} '/' db_name '.csv'];
    if ~exist(['./dts/' db_dir '/' versions{gg} '/'], 'dir')
           mkdir(['./dts/' db_dir '/' versions{gg} '/'])
    end

    load([db_dir, '/', versions{gg}, '/est_baseawbbest_shi_256.mat'])
    
    if fast_db
        load([db_dir, '/', versions{gg}, '/est_baseawb_1_1_shi_256.mat'])
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

    out_dir = ['./fold_division/' db_dir '/' versions{gg} '/' db_name];
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


end