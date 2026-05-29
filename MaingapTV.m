%%
clear;
clc;
close all;
%% Addpaths
addpath(genpath('./src'));
addpath(genpath('./reconstruction'));
addpath(genpath('./RGB'));
addpath(genpath('./Metrics'));
addpath(genpath('./codes'));
addpath(genpath('./polarization=4-bands=12/Simeng_lock'));
addpath(genpath('./results'));
%addpath(genpath('./polarization=4-bands=12/Simeng_ball'))
%addpath(genpath('./polarization=4-bands=12/Simeng_fruit'))

%% Initialization
NS = 12; % Number of bands
NF = 4; % Number of polarization angles
realData = 1;
[M,N,~]= sizeSpectralCube();
J = zeros(M,N,NS);
C5 = zeros(M,N,NS,NF); % coded aperture

RGB_frame = zeros(M,N,3,NF); % RGB reconstruction
Xrec = zeros(M,N,NS,NF); % Video-spectral recuperado
idx = 0;
idy = 0;
p = zeros(NF,1); % PSNR of each frame
ss = zeros(NF,1); % SSIM of each frame
sam = zeros(NF,1); % SAM of each metric
angles = ["0","45","90","135"];
method = 5;
% Compute 3D sphere packing
N1 = max([M,N]);

id = 1:NS;

%load("mask_mosaico.mat")
%load("mask_spherepacking.mat")
%load("mask_random.mat")

%T = C;

[a,b,ma,G]=DDDRSNNPLattice(N,NF*NS);
c = 0;
for k=1:NF
    for l=1:NS
        c = c + 1;
        T(:,:,l,k) = G==c;
    end
end
T2 = [];
G3 = zeros(M,N,NF);

for t=1:NF
    tempora = zeros(M,N);
    for s = 1:NS
        tempora = tempora + T(:,:,s,t)*s;
    end
    G3(:,:,t) = tempora;
end
G4 = sum(G3,3);

%% Sampling Spectral-video
[Y] = sampling(T,realData);
%imagesc(Y),colormap('gray'),pbaspect([1 1 1]);
imwrite(Y,".\results\measurement.png");

for t=1:NF
    disp("Frame "+num2str(t))
    
    NameDataset = "frame_"+num2str(angles(t))+".mat";
    load(NameDataset);

    X = imresize(cube,[M,N]);
    D = size(X,3);
    idx  = round(linspace(1,D,NS));
    X = X(:,:,idx);

    Xrec(:,:,:,t) = CallGAPTV(X,Y,T(:,:,:,t));
    %% Recovery spectral for each frame
    %[Xrec(:,:,:,t)] = reconstruction(J,G4,method); % Xrec: spectral-video reconstruction
    Xrec(:,:,:,t) = mat2gray(abs(Xrec(:,:,:,t)))*255;
    RGB_frame(:,:,:,t)= RGB_test(Xrec(:,:,:,t));
    [RGB_X(:,:,:,t)] = RGB_test((X));

    kdataset = 1;
    X = mat2gray(X);
    Xrec1 = mat2gray(Xrec(:,:,:,t));
    [p(t),ss(t),r,sam(t)] = metrics(X(:,:,:),Xrec1(:,:,:),kdataset);
    Gt = RGB_X(:,:,:,t);
    Recon = RGB_frame(:,:,:,t);
    imwrite(Gt,".\results\Gt_RGB_frame"+num2str(t)+".png");
    imwrite(Recon,".\results\Recon_RGB_frame"+num2str(t)+".png");

end
subplot(1,3,1),imagesc(Y),colormap('gray'),title("Measurement "),pbaspect([1 1 1]),axis off;
subplot(1,3,2),imagesc(RGB_frame(:,:,:,t)),title("Reconstruction "+"frame= "+num2str(t)),pbaspect([1 1 1]),axis off;
subplot(1,3,3),imagesc(RGB_X(:,:,:,t)),title("Groundtruth "+"frame= "+num2str(t)),pbaspect([1 1 1]),axis off;
%pause(0.1)


disp("Totals")
disp("PSNR "+ num2str(mean(p))+ " SSIM "+num2str(mean(ss))+" SAM "+num2str(mean(sam)));
disp("---------------------------------------------------------------------------------------------------------")
implay(RGB_frame(:,:,:,:)./max(RGB_frame(:)))
