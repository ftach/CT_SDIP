%% READ IMAGE
%img = imread('cameraman.tif'); 
close all; 
clear all; 

% img = double(img);
% img = img./max(img(:));
% img = imresize(img,[32, 32]);
% img = imadjust(img);  % normalize
% theta = dct2(img);
% theta_norm = theta(:);  % flatten

img = imread('sinogram27.png'); 
figure;
imshow(img); 
xlabel('Angles', 'FontSize',24,'FontWeight','bold')
ylabel('Sensors', 'FontSize',24,'FontWeight','bold')

%img = imread('mnist_5_orig.png'); 
img = 255 - rgb2gray(img);

%% 
img = imresize(img, [32, 32]); 
img = double(img); 
%figure,imshow(reshape(psi*theta_norm, [32, 32])); 

img = img./max(img(:));
N = size(img(:), 1); 
%theta = dct2(img); 
%theta_norm = theta(:);

psi = dctmtx(N); % DCT BASIS 
theta_norm = psi'*img(:); % new
N = size(theta_norm, 1); 
K = floor(0.05*N); 
[x_sorted,x_position] = sort(abs(theta_norm), 'descend'); 
theta_norm(x_position(K+1:end))=0; 
%figure,imshow(reshape(psi*theta_norm, [32, 32])); 
nnz(theta_norm)

%% Get a sparse matrix 
N = size(theta_norm, 1);
%M = floor(2*K*log(N)); % mesurements 
M = 0.5*N; 

%% Binomial distribution  
H = randi([0, 1], M, N) * 2 - 1;
columnNorms = sqrt(sum(H.^2, 1)); % Norma de cada columna
H = H ./ columnNorms;

%% Gaussian sampling matrix
H = randn(M, N); 
columnNorms = sqrt(sum(H.^2, 1)); % Norma de cada columna
H = H ./ columnNorms;


%% PARAMETERS
maxiter = 30; 
mu = 1e-1; % regularization 
lambda = mu; 
rho = 1; 

coherence = sqrt(N)*max(H*psi, [], 'all')
%floor(coherence*coherence*K*log(N))

% Difference matrix 
D = eye(N);
IX = sub2ind([N N],2:N,1:N-1);
D(IX) = -1; 
D(1, N) = -1 ; 
% 
SNRdB = 30;
w =  sqrt(var(psi*theta_norm, 1)*exp(-0.1*SNRdB * log(10))).*randn(M,1);

%y = H*psi*theta_norm ; % observations 
y = H*psi*theta_norm + w; % observations 
%figure,imshow(idct2(reshape(y, [16, 16]))); 

%% RUN ADMM 
tic
theta_est = admm(theta_norm, y, H, psi, D, lambda, mu, rho, maxiter); 
elapsedTime = toc; 
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);
%% PLOT RESULTS
figure; 
subplot(1, 2, 1);
%imshow(idct2(reshape(theta_norm, 64, 64))); 
imshow(reshape(psi*theta_norm, 32, 32)); % new
title('Original X');
subplot(1, 2, 2);
imshow(reshape(psi*theta_est, 32, 32)); % new 

title('X estimated');

l2_norm = norm(psi*theta_norm - psi*theta_est) % MSE 
similarity = ssim(psi*theta_norm, psi*theta_est)
%% PLOT Y 
y_est = H*psi*theta_est ; % observations 
%figure,imshow(idct2(reshape(y, [16, 16]))); 
