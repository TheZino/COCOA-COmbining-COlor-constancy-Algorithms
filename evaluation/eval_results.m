clear
clc


distortions = {'awgn_s0', 'awgn_s20', 'awgn_s40', 'blur_s0', ... 
            'blur_s1', 'blur_s4', 'jpeg_q10', 'jpeg_q15', 'jpeg_q60', ...
            'awgn_s10', 'awgn_s2.5', 'awgn_s5', 'blur_s0.5', ...
            'blur_s2', 'blur_s8', 'jpeg_q100', 'jpeg_q30', 'jpeg_q90'};

results = cell(length(distortions), 7);

for gg = 1:length(distortions)

    results{gg, 1} = distortions{gg};

    % Baseline for HC fair with hand tuned model
    model = ['../out/single_image/SG568_distortions/fast_' distortions{gg}];

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

    minim = min(tot{:,2});
    avg = mean(tot{:,2});
    med = median(tot{:,2});

    p90 = prctile(tot{:,2}, 90);
    p95 = prctile(tot{:,2}, 95);
    mx = max(tot{:,2});

    results(gg, 2:7) = {minim, avg, med, p90, p95, mx};
    % fprintf('%f %f %f %f %f %f\n', min, avg, med, p90, p95, mx);
    % fprintf('min: %f   avg: %f   median: %f   p90: %f   p95: %f   max: %f\n', min, avg, med, p90, p95, mx);

end

res_tab = cell2table(results, 'VariableNames', {'distortion', 'min', 'mean', 'median', '90 perc', '95 perc', 'max'});

writetable(res_tab, './distortions.csv')
