% general_cc_all: estimates the light source of an input_data, using
% algorithms from "Edge-Based Color Constancy" by van de Weijer et al.
%
% S. Zini, M. Buzzelli, S. Bianco, R. Schettini
% "COCOA: Combining Color Constancy Algorithms for Images and Videos"
% IEEE Transactions on Computational Imaging, 2022
% vol.8 pp. 795--807
%
% Adapted from original code referenced in:
% J. van de Weijer, Th. Gevers, A. Gijsenij
% "Edge-Based Color Constancy"
% IEEE Trans. Image Processing, 2007.

function whites = general_cc_all(input_data, mink_norms, sigmas, mask_im)

    if(nargin<2), mink_norms=[1 1 9 4 1]; end
    if(nargin<3), sigmas=[6 1 9 0 0]; end
    if(nargin<4), mask_im=zeros(size(input_data,1),size(input_data,2)); end

    mink_norm_ge1 = mink_norms(1);
    mink_norm_ge2 = mink_norms(2);
    mink_norm_ggw = mink_norms(3);
    mink_norm_sog = mink_norms(4);
%     mink_norm_gw = 1; % fixed
    
    sigma_ge1 = sigmas(1);
    sigma_ge2 = sigmas(2);
    sigma_ggw = sigmas(3);
%     sigma_sog = 0; % fixed
%     sigma_gw = 0; % fixed

    %% Mask
    
    % remove all saturated points
    saturation_threshold = 255;
    mask_im2 = mask_im + (dilation33(double(max(input_data,[],3)>=saturation_threshold)));   
    mask_im2 = double(mask_im2==0);
    
    % Border
    temp = ones(size(mask_im2));
    [y, x] = ndgrid(1:size(mask_im2,1), 1:size(mask_im2,2));
    
    % ge1
    ttemp = temp.*( (x<size(temp,2)-sigma_ge1 ) & (x>sigma_ge1+1) );
    ttemp = ttemp.*( (y<size(temp,1)-sigma_ge1 ) & (y>sigma_ge1+1) );
    mask_im2_ge1 = ttemp.*mask_im2;
    
    % ge2
    ttemp = temp.*( (x<size(temp,2)-sigma_ge2 ) & (x>sigma_ge2+1) );
    ttemp = ttemp.*( (y<size(temp,1)-sigma_ge2 ) & (y>sigma_ge2+1) );
    mask_im2_ge2 = ttemp.*mask_im2;
    
    % ggw
    ttemp = temp.*( (x<size(temp,2)-sigma_ggw ) & (x>sigma_ggw+1) );
    ttemp = ttemp.*( (y<size(temp,1)-sigma_ggw ) & (y>sigma_ggw+1) );
    mask_im2_ggw = ttemp.*mask_im2;
    
    % rest
    ttemp = temp.*( (x<size(temp,2) ) & (x>1) );
    ttemp = ttemp.*( (y<size(temp,1) ) & (y>1) );
    mask_im2_rest = ttemp.*mask_im2;

    % the mask_im2 contains pixels higher saturation_threshold and which are
    % not included in mask_im.
    
    %% Derivatives
    gd00 = input_data;
    gd01 = input_data;
    gd10 = input_data;
    gd11 = input_data;
    gd02 = input_data;
    gd20 = input_data;
    
    % Initialize the filters
    break_off_sigma = 3.;
    fs_ggw = floor(break_off_sigma*sigma_ggw+0.5);
    fs_ge1 = floor(break_off_sigma*sigma_ge1+0.5);
    fs_ge2 = floor(break_off_sigma*sigma_ge2+0.5);
    f_ggw = fill_border(input_data, fs_ggw);
    f_ge1 = fill_border(input_data, fs_ge1);
    f_ge2 = fill_border(input_data, fs_ge2);
    x_ggw = -fs_ggw:1:fs_ggw;
    x_ge1 = -fs_ge1:1:fs_ge1;
    x_ge2 = -fs_ge2:1:fs_ge2;
    
    % gd00 (ggw)
    Gauss = 1/(sqrt(2 * pi) * sigma_ggw)* exp((x_ggw.^2)/(-2 * sigma_ggw * sigma_ggw) );
    G = Gauss/sum(Gauss);
    for ii=1:3
        H = filter2(G', filter2(G, f_ggw(:,:,ii)));
        gd00(:,:,ii) = H(fs_ggw+1:size(H,1)-fs_ggw, fs_ggw+1:size(H,2)-fs_ggw);
    end
    
    % gd01, gd10 (ge1)
    Gauss = 1/(sqrt(2 * pi) * sigma_ge1)* exp((x_ge1.^2)/(-2 * sigma_ge1 * sigma_ge1) );
    G0 = Gauss/sum(Gauss);
    G1 =  -(x_ge1/sigma_ge1^2).*Gauss;
    G1  =  G1./(sum(sum(x_ge1.*G1)));
    for ii=1:3
        H = filter2(G1', filter2(G0, f_ge1(:,:,ii)));
        gd01(:,:,ii) = H(fs_ge1+1:size(H,1)-fs_ge1, fs_ge1+1:size(H,2)-fs_ge1);
        
        H = filter2(G0', filter2(G1, f_ge1(:,:,ii)));
        gd10(:,:,ii) = H(fs_ge1+1:size(H,1)-fs_ge1, fs_ge1+1:size(H,2)-fs_ge1);
    end
    
    % gd11, gd02, gd20 (ge2)
    Gauss = 1/(sqrt(2 * pi) * sigma_ge2)* exp((x_ge2.^2)/(-2 * sigma_ge2 * sigma_ge2) );
    G0 = Gauss/sum(Gauss);
    G1 =  -(x_ge2/sigma_ge2^2).*Gauss;
    G1  =  G1./(sum(sum(x_ge2.*G1)));
    G2 = (x_ge2.^2/sigma_ge2^4-1/sigma_ge2^2).*Gauss;
    G2 = G2-sum(G2)/size(x_ge2,2);
    G2 = G2/sum(0.5*x_ge2.*x_ge2.*G2);
    for ii=1:3
        H = filter2(G1', filter2(G1, f_ge2(:,:,ii)));
        gd11(:,:,ii) = H(fs_ge2+1:size(H,1)-fs_ge2, fs_ge2+1:size(H,2)-fs_ge2);
        
        H = filter2(G2', filter2(G0, f_ge2(:,:,ii)));
        gd02(:,:,ii) = H(fs_ge2+1:size(H,1)-fs_ge2, fs_ge2+1:size(H,2)-fs_ge2);
        
        H = filter2(G0', filter2(G2, f_ge2(:,:,ii)));
        gd20(:,:,ii) = H(fs_ge2+1:size(H,1)-fs_ge2, fs_ge2+1:size(H,2)-fs_ge2);
    end
    
    %% Norm derivatives + abs
    data_ge1 = sqrt(gd10.^2 + gd01.^2);
    data_ge2 = sqrt(gd20.^2 + 4*gd11.^2 + gd02.^2);
    data_ggw = abs(gd00);
    % gw, sog, wp
    data_rest = abs(input_data);

    %% Kleurs + mask
    kleur_ge1 = power(data_ge1, mink_norm_ge1).*mask_im2_ge1;
    kleur_ge2 = power(data_ge2, mink_norm_ge2).*mask_im2_ge2;
    kleur_ggw = power(data_ggw, mink_norm_ggw).*mask_im2_ggw;
    kleur_sog = power(data_rest, mink_norm_sog).*mask_im2_rest;
    % gw, wp
    kleur_rest = data_rest.*mask_im2_rest;
    
    %% Whites
    white_ge1 = power(sum(kleur_ge1, [1,2]), 1/mink_norm_ge1);
    white_ge2 = power(sum(kleur_ge2, [1,2]), 1/mink_norm_ge2);
    white_ggw = power(sum(kleur_ggw, [1,2]), 1/mink_norm_ggw);
    white_sog = power(sum(kleur_sog, [1,2]), 1/mink_norm_sog);
    white_gw = sum(kleur_rest, [1,2]);
    white_wp = max(kleur_rest, [], [1,2]);

    whites = squeeze(cat(2, white_ge1, white_ge2, white_ggw, white_sog, white_gw, white_wp));
    som = repmat(sqrt(sum(whites.^2,2)), 1,3);
    whites = whites ./ som;
    
end
