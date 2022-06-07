
data = readtable(save_file);

if plausibility
    db_name = [db_name '_pla'];
end

out_dir = ['./fold_division/' db_dir '/' db_name];
if ~exist(out_dir, 'dir')
    mkdir(out_dir)
end

    
idx = tr_split;    

%% train

out = data(idx,:);

writetable(out, [out_dir, '/fold_tr.csv'], 'WriteVariableNames',0)

%% validation

idx = val_split;

out = data(idx,:);
writetable(out, [out_dir, '/fold_val.csv'], 'WriteVariableNames',0)
