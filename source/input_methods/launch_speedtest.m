% Add to path the code for edge-based color constancy (fast)
addpath('edgebased_fast');

% Color Checker dataset
files_path = 'filelist_CC.txt'; % List of input files
traw_path = '/home/marco/Datasets/ColorConstancy/ColorChecker/Shi/Masked/'; % Images in 16-bit TIFF-RAW format, with masked color target

filelist = readtable(files_path, 'Delimiter', ',');

%% Original implementation by van de Weijer et al.

time = 0;
for f = 1:size(filelist,1)
    img_filename = fullfile(traw_path, filelist.File{f});
    im = double(imread(img_filename));

    tic;
    im = imresize(im, 256/max(size(im)));
    
    mass = max(im,[],3);
    mask = zeros(size(mass));
    mask(mass==0) = 1;

    % Grey Edge 1 (GE1)
    [R, G, B, ~] =  general_cc(im,1,1,6,mask);
    % Grey Edge 2 (GE2)
    [R, G, B, ~] =  general_cc(im,2,1,1,mask);
    % General GreyWorld (GGW)
    [R, G, B, ~] =  general_cc(im,0,9,9,mask);
    % Shades of Gray (SoG)
    [R, G, B, ~] = general_cc(im,0,4,0,mask);
    % GreyWorld (GW)
    [R, G, B, ~] =  general_cc(im,0,1,0,mask);
    % WhitePoint (WP)
    [R, G, B, ~] = general_cc(im,0,-1,0,mask);
    t = toc;
    time = time+t;
end
time

%% Efficient adaptation as found in:
% "COCOA: Combining Color Constancy Algorithms for Images and Videos"

time = 0;
for f = 1:size(filelist,1)
    img_filename = fullfile(traw_path, filelist.File{f});
    im = double(imread(img_filename));

    tic;
    im = imresize(im, 256/max(size(im)));
    
    mass = max(im,[],3);
    mask = zeros(size(mass));
    mask(mass==0) = 1;

    whites_fast = general_cc_all(im, [1 1 9 4 1], [6 1 9 0 0], mask);

    t = toc;
    time = time+t;
end
time


%% Efficient adaptation (fast version with all parameters set to 1) as found in:
% "COCOA: Combining Color Constancy Algorithms for Images and Videos"

time = 0;
for f = 1:size(filelist,1)
    img_filename = fullfile(traw_path, filelist.File{f});
    im = double(imread(img_filename));

    tic;
    im = imresize(im, 256/max(size(im)));
    
    mass = max(im,[],3);
    mask = zeros(size(mass));
    mask(mass==0) = 1;

    whites_faster = general_cc_1_1(im, mask);

    t = toc;
    time = time+t;
end
time

