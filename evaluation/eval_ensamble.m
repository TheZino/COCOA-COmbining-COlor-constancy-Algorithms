clear
clc


model = './out/single_image/SG568/reduced_inputs/';
model1  = './out/single_image/SG568/reduced_inputs/default_SG568_red1_123123';
model2  = './out/single_image/SG568/reduced_inputs/default_SG568_red1_4321';
model3  = './out/single_image/SG568/reduced_inputs/default_SG568_red1_5678';


for i=0:2
    res1 = readtable([model1,'/fold',num2str(i),'.csv']);
    res2 = readtable([model2,'/fold',num2str(i),'.csv']);
    res3 = readtable([model3,'/fold',num2str(i),'.csv']);
    
    
    names = res1.Var1;
    res_rgb = (res1{:,5:7} + res2{:,5:7} + res3{:,5:7})./3;
    gt_rgb = res1{:, 2:4};
    
    fid = fopen( [model,'/angular_error_fold',num2str(i),'.csv'], 'w' );

    for jj = 1 : length(gt_rgb)
        
        rec_err = recovery_error(res_rgb(jj,:), gt_rgb(jj,:));
        fprintf( fid, '%s,%d\n', names{jj}, rec_err);

    end
    fclose( fid );
    
end

f0 = readtable([model '/angular_error_fold0.csv']);
f1 = readtable([model '/angular_error_fold1.csv']);
f2 = readtable([model '/angular_error_fold2.csv']);


tot = [f0; f1; f2];

min = min(tot{:,2});
avg = mean(tot{:,2});
med = median(tot{:,2});

p90 = prctile(tot{:,2}, 90);
p95 = prctile(tot{:,2}, 95);
mx = max(tot{:,2});


fprintf('%f %f %f %f %f %f\n', min, avg, med, p90, p95, mx);