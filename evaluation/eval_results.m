clear
clc

% Baseline for HC fair with hand tuned model
model = '../out/single_image/SG568/default';

%% Error calculation

for i=0:2
    res = readtable([model,'/fold',num2str(i),'.csv']);
    names = res.Var1;

    res_rgb = res{:, 5:7};
    gt_rgb = res{:, 2:4};

    
    fid = fopen( [model,'/angular_error_fold',num2str(i),'.csv'], 'w' );

    for jj = 1 : length(gt_rgb)

        rec_err = recovery_error(res_rgb(jj,:), gt_rgb(jj,:));
        fprintf( fid, '%s,%d\n', names{jj}, rec_err);

    end
    fclose( fid );
    
end

%% Print statistics

f0 = readtable([model, '/angular_error_fold0.csv']);
f1 = readtable([model, '/angular_error_fold1.csv']);
f2 = readtable([model, '/angular_error_fold2.csv']);

tot = [f0; f1; f2];

min = min(tot{:,2});
avg = mean(tot{:,2});
med = median(tot{:,2});

p90 = prctile(tot{:,2}, 90);
p95 = prctile(tot{:,2}, 95);
mx = max(tot{:,2});


% fprintf('%f %f %f %f %f %f\n', min, avg, med, p90, p95, mx);
fprintf('min: %f   avg: %f   median: %f   p90: %f   p95: %f   max: %f\n', min, avg, med, p90, p95, mx);

%%
clear
