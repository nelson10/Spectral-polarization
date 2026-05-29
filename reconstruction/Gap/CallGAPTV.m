function Xrec = CallGAPTV(X,Y,T)

mask = T;
meas = Y;
% [1] load dataset

para.nframe =   1; % number of coded frames in this test
para.MAXB   = 255;
orig = double(X);
% [2] apply GAP-Denoise for reconstruction
para.Mfunc  = @(z) A_xy(z,mask);
para.Mtfunc = @(z) At_xy_nonorm(z,mask);

para.Phisum = sum(mask.^2,3);
para.Phisum(para.Phisum==0) = 1;
% common parameters
para.lambda   =     1; % correction coefficiency
para.acc      =     1; % enable GAP-acceleration
para.flag_iqa = false; % disable image quality assessments in iterations

%% [2.1] GAP-TV, ICIP'16
para.denoiser = 'tv'; % TV denoising
para.maxiter  = 300; % maximum iteration
para.tvweight =  15; % weight for TV denoising
para.tviter   =  15; % number of iteration for TV denoising

[vgaptv,psnr_gaptv,ssim_gaptv,tgaptv] = ...
    gapdenoise_cacti(mask,meas,orig,[],para);
Xrec = vgaptv;
end