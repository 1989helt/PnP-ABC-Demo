%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Demo_PnP_ABC2D: Simultaneous Deblurring and Inpainting with Arbitrary Boundary Conditions 
%Paper information: Denoiser-guided Image Deconvolution with Unknown Boundaries and Incomplete Observations
%Written by Liangtian He; Email:  helt@ahu.edu.cn;
%Last modified by Liangtian He, March.16, 2023
%The updated version of our source code will be released later
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear,clc,clf,close all
randn('seed',0); %  Fix random seed
%%%***************************************************************
x0 = double(imread('cameraman.tif')); % Load the test image 
%%%***************************************************************
[xN,yN]=size(x0);  %% size of full image
x0 = x0-min(x0(:));
x0 = x0/max(x0(:));    
x0 = x0*255;   % Pixel value in the range [0,255]
%%%***************************************************************
BlurDim = 13 ;  % filter size (squared): 13x13 pixels
filter_type =      1      ; % filter type
%%%***************************************************************
switch filter_type
    case 1 
        ht=ones(BlurDim); ht=ht./sum(ht(:)); %% Uniform/average 13x13 pixels  
    case 2 
        ht = fspecial('disk',BlurDim/2);  ht=ht./sum(ht(:));  % out_of_focus 13x13 pixels 
    case 3    
        ht=fspecial('motion', 17, 135);   ht=ht./sum(ht(:));  %% Motion 13x13 pixels         
    case 4  
        ht=fspecial('gaussian', 13, sqrt(13));   ht=ht./sum(ht(:));  %% Gaussian 13x13 pixels        
end
%%%***************************************************************
fsize = round( (BlurDim-1)/2 );  % fsize - size of each side of the filter (6)

blur=@(f,k)imfilter(f,k,'circular');
y_blur = convn(x0, ht, 'valid');  %% blurry image only size 244*244 ;

[sz1_x0, sz2_x0] =  size(y_blur);

Ratio_Set = [0.2, 0.3, 0.5, 0.8]; %% (r = 0.8; 0.7; 0.5; 0.2 in the paper)

%%%***************************************************************
ratio = Ratio_Set(4); % ratio of available data, Options: [0.2, 0.3, 0.5, 0.8]
%%%***************************************************************

P = double(rand(size(y_blur)) > (1-ratio)); %% random pixel missing

%%%***************************************************************
sigma =   2   ; %% Noise level
%%%***************************************************************

y_observed = P.*y_blur + sigma*randn(size(y_blur)) ;

x0_crop = x0(1+fsize:end-fsize, 1+fsize:end-fsize);

%%% generating mask
Mask = zeros(xN,yN);
Mask(1+fsize:end-fsize, 1+fsize:end-fsize) = double(P) ;

y_input = zeros(xN,yN);
y_input(1+fsize:end-fsize, 1+fsize:end-fsize) =  y_observed  ;

figure(1) ; imshow(P,[]) ;
figure(2) ; imshow(y_observed,[]) ;
figure(3) ; imshow(Mask,[]) ;
figure(4) ; imshow(y_input,[]) ;

%%%***************************************************************
opts.lambda1 = 0.5 ; %% Nonlocal regularization parameter % manually tuned
opts.lambda2 = 2.5 ; %% Local regularization parameter % manually tuned
opts.mu1 = 1e-3  ; %% Nonlocal Lagrange penalty parameter
opts.mu2 = 1e-3 ; %% Local Lagrange penalty parameter
opts.mu3 = 1e-1 ; %% Lagrange penalty parameter
opts.tol = 1e-5;  %% tolerance condition
opts.maxit = 200 ; %% maximum iteration number
%%%***************************************************************

fprintf('***************************************************************\n')
fprintf('***************************************************************\n')
fprintf('Running Please waitting ...\n')
x_out = PnPABC2D_Deblur_Inpaint(opts, x0, y_input, ht, blur, Mask); %% Double denoisers
fprintf('Running end ...\n')
fprintf('***************************************************************\n')
fprintf('***************************************************************\n')
[mssim_out] = ssim_index(x_out,x0); psnr_out = psnr(x_out/255,x0/255) ;
fprintf('Final estimate PSNR: %4.2f SSIM: %4.4f  \n',  psnr_out, mssim_out);  

%  Show the recovery image
figure(5); imshow(x0,[]); 
title('Original full image','fontsize',13) ;
figure(6); imshow(y_observed,[]); 
title('Observed image','fontsize',13) ;
figure(7); imshow(x_out,[]); 
title(sprintf('Recovered full image, PSNR: %4.2fdB,SSIM: %4.4f',psnr_out,mssim_out),'fontsize',10);
