function [est_GE1, est_GGW, est_GW] = run_eb_fast(files_path, traw_path, verbose, is8bit)
% Run base AWB with predefined parameters

    if nargin < 3
        verbose = false;
    end
    if nargin < 4
        is8bit = false;
    end

    % Load file list
    filelist = readtable(files_path, 'Delimiter', ',');

    est_template = table(filelist.File, NaN(size(filelist,1),1), NaN(size(filelist,1),1), NaN(size(filelist,1),1), 'VariableNames', {'File', 'R', 'G', 'B'});
    est_GE1 = est_template;
    est_GGW = est_template;
    est_GW = est_template;
    for f = 1:size(filelist,1)
        if verbose
            fprintf('%d/%d (%.2f%%)\n', f, size(filelist,1), 100*f/size(filelist,1));
        end
        im = imread(fullfile(traw_path, filelist.File{f}));

        % Preprocess image
        if is8bit
            im = im2double(im);
        else
            % ColorChecker
            if contains(filelist.File{f}, '8D5U') % Canon EOS-1DS
                im = 255*double(abs(im-0))/3500;
    %             im = imresize(im, 256/2041);
            elseif contains(filelist.File{f}, 'IMG_') % Canon EOS 5D
                im = 255*double(abs(im-129))/3692;
    %             im = imresize(im, 256/2193);
            % NUS
            elseif contains(filelist.File{f}, 'Canon1DsMkIII')
                im = 255*double(abs(im-0))/15279;
            elseif contains(filelist.File{f}, 'Canon600D')
                im = 255*double(abs(im-0))/15303;
            elseif contains(filelist.File{f}, 'FujifilmXM1')
                im = 255*double(abs(im-0))/4079;
            elseif contains(filelist.File{f}, 'NikonD40')
                im = 255*double(abs(im-0))/3981;
            elseif contains(filelist.File{f}, 'NikonD5200')
                im = 255*double(abs(im-0))/15892;
            elseif contains(filelist.File{f}, 'OlympusEPL6')
                im = 255*double(abs(im-0))/4043;
            elseif contains(filelist.File{f}, 'PanasonicGX1')
                im = 255*double(abs(im-0))/4095;
            elseif contains(filelist.File{f}, 'SamsungNX2000')
                im = 255*double(abs(im-0))/4095;
            elseif contains(filelist.File{f}, 'SonyA57')
                im = 255*double(abs(im-0))/4093;
            % INTEL-TAU
            elseif contains(filelist.File{f}, 'Canon_5DSR')
                im = 255*double(abs(im-0))/(2^16);
            elseif contains(filelist.File{f}, 'Nikon_D810')
                im = 255*double(abs(im-0))/(2^16);
            elseif contains(filelist.File{f}, 'Sony_IMX135')
                im = 255*double(abs(im-0))/(2^16);
            else
                im = double(im);
                error('Unknown camera. Unknown preprocessing.');
            end
        end
        im = imresize(im,256/max(size(im)));
%         im = imresize(im,[32 32]);
        
        mass = max(im,[],3);
        mask = zeros(size(mass));
        mask(mass==0) = 1;

        % Grey Edge 1 (GE1)
        [R, G, B, ~] =  general_cc(im,1,1,1,mask);
        est_GE1.R(f) = R;
        est_GE1.G(f) = G;
        est_GE1.B(f) = B;

        % General GreyWorld (GGW)
        [R, G, B, ~] =  general_cc(im,0,1,1,mask);
        est_GGW.R(f) = R;
        est_GGW.G(f) = G;
        est_GGW.B(f) = B;

        % GreyWorld (GW)
        [R, G, B, ~] =  general_cc(im,0,1,0,mask);
        est_GW.R(f) = R;
        est_GW.G(f) = G;
        est_GW.B(f) = B;

    end

    est_GE1.Properties.VariableNames{2} = 'GE1_1_1_1_R';
    est_GE1.Properties.VariableNames{3} = 'GE1_1_1_1_G';
    est_GE1.Properties.VariableNames{4} = 'GE1_1_1_1_B';

    est_GGW.Properties.VariableNames{2} = 'GGW_0_1_1_R';
    est_GGW.Properties.VariableNames{3} = 'GGW_0_1_1_G';
    est_GGW.Properties.VariableNames{4} = 'GGW_0_1_1_B';

    est_GW.Properties.VariableNames{2} = 'GW_0_1_0_R';
    est_GW.Properties.VariableNames{3} = 'GW_0_1_0_G';
    est_GW.Properties.VariableNames{4} = 'GW_0_1_0_B';

end
