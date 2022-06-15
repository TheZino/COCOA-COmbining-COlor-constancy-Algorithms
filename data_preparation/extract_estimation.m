% Extract data from .mat files

db_dir = './AWB_estimations/NUS/';
data_name = 'est_baseawbbest.mat';

out_dir = [db_dir, '/estimations/'];

if ~exist(out_dir, 'dir')
   mkdir(out_dir)
end


%%
load([db_dir, data_name]);

writetable(est_GE1, [db_dir, '/estimations/est_illuminants_GE1']);
writetable(est_GE2, [db_dir, '/estimations/est_illuminants_GE2']);
writetable(est_GGW, [db_dir, '/estimations/est_illuminants_GGW']);
writetable(est_GW, [db_dir, '/estimations/est_illuminants_GW']);
writetable(est_SoG, [db_dir, '/estimations/est_illuminants_SoG']);
writetable(est_WP, [db_dir, '/estimations/est_illuminants_WP']);


%% GT data

% gt_table = readtable([db_dir, '/image_list.txt']);
% writetable(gt_table, [db_dir, '/GT_' db_name '.csv']);
