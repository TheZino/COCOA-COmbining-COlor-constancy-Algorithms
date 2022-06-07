

for i=0:2
    tr_split{i+1} = sort(tr_split{i+1});
    idx = tr_split{i+1};

    %% train
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_tr', num2str(i) '.csv'], 'WriteVariableNames',0)

    %% validation
    val_split{i+1} = sort(val_split{i+1});
    idx = val_split{i+1};
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_val', num2str(i) '.csv'], 'WriteVariableNames',0)

    %% test
    te_split{i+1} = sort(te_split{i+1});
    idx = te_split{i+1};
    out = data(idx,:);
    writetable(out, [out_dir, '/fold_te', num2str(i) '.csv'], 'WriteVariableNames',0)
end
