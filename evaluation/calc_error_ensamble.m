
for i=0:2
    res1 = readtable(['./results/',model1,'/fold',num2str(i),'.csv']);
    res2 = readtable(['./results/',model2,'/fold',num2str(i),'.csv']);
    res3 = readtable(['./results/',model3,'/fold',num2str(i),'.csv']);
    
    
    names = res1.Var1;
    if arc
        res_rgb = arc2rgb(res{:, 4:5}, 'InputFormat', mode);
        gt_rgb = arc2rgb(res{:, 2:3}, 'InputFormat', mode);
    else
        res_rgb = (res1{:,5:7} + res2{:,5:7} + res3{:,5:7})./3;% res{:, 5:7};
        gt_rgb = res1{:, 2:4};
    end
    
    fid = fopen( ['./results/',model,'/angular_error_fold',num2str(i),'.csv'], 'w' );

    for jj = 1 : length(gt_rgb)
        
        if strcmp(evaluation_type, '5D')
            if contains(names{jj}, 'IMG_')
            
                rec_err = recovery_error(res_rgb(jj,:), gt_rgb(jj,:));
                fprintf( fid, '%s,%d\n', names{jj}, rec_err);

            end
        elseif strcmp(evaluation_type, '1D')
            if ~contains(names{jj}, 'IMG_')
            
                rec_err = recovery_error(res_rgb(jj,:), gt_rgb(jj,:));
                fprintf( fid, '%s,%d\n', names{jj}, rec_err);

            end
        else
            
            rec_err = recovery_error(res_rgb(jj,:), gt_rgb(jj,:));
            fprintf( fid, '%s,%d\n', names{jj}, rec_err);
        end

    end
    fclose( fid );
    
end