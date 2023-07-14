%% INITIALISATION 
close all; 
clear all; 
img = imread('mnist_5_orig.png'); 
img = imresize(img, [32, 32]); 
img = 255 - rgb2gray(img);
img = double(img); 
img = img./255;
img = img(:);
% sparsity

N = size(img, 1); 
K = floor(0.05*N); % dispersion if we want to choose the sparsity

nnz(img)/(32*32); 


%% NOrmal try 
S = nnz(img); % non zero values
C = 1; % constant
%M = floor(C*S*log(N)); % mesurements 
M = 512;
phi = randn(M, N); % sampling matrix

% Normalizar las columnas para que sean unitarias
columnNorms = sqrt(sum(phi.^2, 1)); % Norma de cada columna
phi = phi ./ columnNorms;

y = phi*img ; % observations 
%% RUN HIT
maxiter = 2000; 
mu = 1e-1; % regularization 

x_est = HIT(img, y, phi, K, mu, maxiter); 

figure; 
subplot(1, 2, 1);
imshow(reshape(img, 32, 32)); 
title('Original X');
subplot(1, 2, 2);
imshow(reshape(x_est, 32, 32)); 
title('X estimated');

l2_norm = norm(img-x_est) % MSE 
similarity = ssim(x_est, img) % SSIM 
%% Noisy observation
SNRdB = 30;
w =  sqrt(var(img, 1)*exp(-0.1*SNRdB * log(10))).*randn(M,1);

y_noised = y + w; 

%% RUN with noise

maxiter = 2000; 
mu = 1e-1; % regularization 

tic

x_est = HIT(img, y_noised, phi, S, mu, maxiter); 

elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);

figure; 
subplot(1, 2, 1);
imshow(reshape(img, 32, 32)); 
title('Original X');
subplot(1, 2, 2);
imshow(reshape(x_est, 32, 32)); 
title('X estimated');

l2_norm = norm(img-x_est) % MSE 
similarity = ssim(x_est, img) % SSIM 

%% Binomial distribution  
phi = randi([0, 1], M, N) * 2 - 1;
columnNorms = sqrt(sum(phi.^2, 1)); % Norma de cada columna
phi = phi ./ columnNorms;

y_bin = phi*img + w; % observations 
%% RUN with Binomial

maxiter = 2000; 
mu = 1e-1; % regularization 
tic
x_est = HIT(img, y_bin, phi, S, mu, maxiter); 
elapsedTime = toc; 
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);

figure; 
subplot(1, 2, 1);
imshow(reshape(img, 32, 32)); 
title('Original X');
subplot(1, 2, 2);
imshow(reshape(x_est, 32, 32)); 
title('X estimated');

l2_norm = norm(img-x_est) % MSE 
similarity = ssim(x_est, img) % SSIM 
%% THETA PROBLEM 
img = imread('cameraman.tif'); 
img = double(img);
img = img./max(img(:));
img = imresize(img,[32,32]);
img = imadjust(img); 
theta = dct2(img); % normalize
% figure,imshow(idct2(theta))

theta_norm = theta(:)  % flatten
figure,imshow(idct2(reshape(theta_norm, [32, 32]))); 
%% Get a sparse matrix 
K = 256; 
[x_sorted,x_position] = sort(abs(theta_norm), 'descend'); 
theta_norm(x_position(K+1:end))=0; 
nnz(theta_norm)
figure,imshow(idct2(reshape(theta_norm, [32, 32]))); 

N = size(theta_norm, 1);
S = 52; % non zero values
C = 1; % constant
%M = floor(C*S*log(N)); % mesurements 
M = 512;
phi = randn(M, N); % sampling matrix

% Normalizar las columnas para que sean unitarias
columnNorms = sqrt(sum(phi.^2, 1)); % Norma de cada columna
phi = phi ./ columnNorms;

maxiter = 1000; 
mu = 1e-1; % regularization 

%% NEW IHT 
x_est = zeros(N, 1); 
psi = dctmtx(1024); 
y = phi*psi*theta_norm ; % observations 

%% RUN
for i=0:1000
    x_old = x_est;
    x_est = x_old + mu*psi'*phi'*(y - phi*psi*x_old); 
    %x_est = x_old + mu*dct2(phi')*(y - phi*x_old); 
    [x_sorted,x_position] = sort(abs(x_est), 'descend'); 
    x_est(x_position(K:end))=0; % thresholding operator

end 

figure; 
subplot(1, 2, 1);
imshow(idct2(reshape(theta_norm, 32, 32))); 
title('Original X');
subplot(1, 2, 2);
imshow(idct2(reshape(x_est, 32, 32))); 
title('X estimated');

l2_norm = norm(theta_norm-x_est) % MSE 