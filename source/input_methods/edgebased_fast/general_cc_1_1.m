% general_cc_1_1: estimates the light source of an input_data, using
% algorithms from "Edge-Based Color Constancy" by van de Weijer et al.
% with all parameters set to 1 for faster computation
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

function whites = general_cc_1_1(input_data, mask_im)

    if(nargin<2), mask_im=zeros(size(input_data,1),size(input_data,2)); end

    sigma = 1;
%     sigma_sog = 0; % fixed
%     sigma_gw = 0; % fixed

%     mink_norm_gw = 1;
    
    %% Mask
    
    % remove all saturated points
    saturation_threshold = 255;
    mask_im2 = mask_im + (dilation33(double(max(input_data,[],3)>=saturation_threshold)));   
    mask_im2 = double(mask_im2==0);
    
    % Border
    temp = ones(size(mask_im2));
    [y, x] = ndgrid(1:size(mask_im2,1), 1:size(mask_im2,2));
    
    % sigma1 masks: ge1 ge2 ggw
    ttemp = temp.*( (x<size(temp,2)-sigma ) & (x>sigma+1) );
    ttemp = ttemp.*( (y<size(temp,1)-sigma ) & (y>sigma+1) );
    mask_im2_sigma1 = ttemp.*mask_im2;
    
    % sigma0 masks: gw/sog wp
    ttemp = temp.*( (x<size(temp,2) ) & (x>1) );
    ttemp = ttemp.*( (y<size(temp,1) ) & (y>1) );
    mask_im2_sigma0 = ttemp.*mask_im2;

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
    fs = floor(break_off_sigma*sigma+0.5);
    f = fill_border(input_data, fs);
    x = -fs:1:fs;
    
    Gauss = 1/(sqrt(2 * pi) * sigma)* exp((x.^2)/(-2 * sigma * sigma) );
    G0 = Gauss/sum(Gauss);
    G1 = -(x/sigma^2).*Gauss;
    G1 = G1./(sum(sum(x.*G1)));
    G2 = (x.^2/sigma^4-1/sigma^2).*Gauss;
    G2 = G2-sum(G2)/size(x,2);
    G2 = G2/sum(0.5*x.*x.*G2);
    
    for ii=1:3
        filter2G0 = filter2(G0, f(:,:,ii));
        filter2G1 = filter2(G1, f(:,:,ii));
        filter2G2 = filter2(G2, f(:,:,ii));
        
        % gd00 (ggw)
        H = filter2(G0', filter2G0);
        gd00(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
        
        % gd01, gd10 (ge1)
        H = filter2(G1', filter2G0);
        gd01(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
        H = filter2(G0', filter2G1);
        gd10(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
        
        % gd11, gd02, gd20 (ge2)
        H = filter2(G1', filter2G1);
        gd11(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
        H = filter2(G2', filter2G0);
        gd02(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
        H = filter2(G0', filter2G2);
        gd20(:,:,ii) = H(fs+1:size(H,1)-fs, fs+1:size(H,2)-fs);
    end

    
    %% Norm derivatives + abs
    data_ge1 = sqrt(gd10.^2 + gd01.^2);
    data_ge2 = sqrt(gd20.^2 + 4*gd11.^2 + gd02.^2);
    data_ggw = abs(gd00);
    % gw/sog, wp
    data_rest = abs(input_data);

    %% Kleurs + mask

    kleur_ge1 = data_ge1.*mask_im2_sigma1;
    kleur_ge2 = data_ge2.*mask_im2_sigma1;
    kleur_ggw = data_ggw.*mask_im2_sigma1;
    % sog/gw, wp
    kleur_rest = data_rest.*mask_im2_sigma0;
    
    %% Whites

    white_ge1 = sum(kleur_ge1, [1,2]);
    white_ge2 = sum(kleur_ge2, [1,2]);
    white_ggw = sum(kleur_ggw, [1,2]);
    white_gw = sum(kleur_rest, [1,2]);
    white_wp = max(kleur_rest, [], [1,2]);

    whites = squeeze(cat(2, white_ge1, white_ge2, white_ggw, white_gw, white_wp));
    som = repmat(sqrt(sum(whites.^2,2)), 1,3);
    whites = whites ./ som;
    
end
