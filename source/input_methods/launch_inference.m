% Add to path the code for edge-based color constancy (fast)
addpath('edgebased_fast');

% Color Checker dataset
files_path = 'filelist_CC.txt'; % List of input files
traw_path = '/home/marco/Datasets/ColorConstancy/ColorChecker/Shi/Masked/'; % Images in 16-bit TIFF-RAW format, with masked color target

verbose = true;

% Edge-based color constancy algorithms
[est_GE1, est_GE2, est_GGW, est_GW, est_SoG, est_WP] = run_eb(files_path, traw_path, verbose);
save('est_baseawbbest.mat', 'est_GE1', 'est_GE2', 'est_GGW', 'est_GW', 'est_SoG', 'est_WP');
% Backup
est_GE1_slow = est_GE1;
est_GGW_slow = est_GGW;
est_SoG_slow = est_SoG;

% Edge-based color constancy algorithms (fast version)
[est_GE1, est_GGW, est_SoG] = run_eb_fast(files_path, traw_path, verbose);
save('est_baseawb_fast.mat', 'est_GE1', 'est_GGW');
