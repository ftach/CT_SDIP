close all; 
%% Read image
img = imread('cameraman.tif'); 
f = double(img);

%% 2-D Wavelet 
[C1, S1] = wavedec2(f, 2, 'haar'); 

%% DCT-2D 
DC = dct2(f); 

%% Represent Wavelet and DC transform of x 

figure; 
subplot(1, 2, 1); 
plot(C1); 
title('Wavelet');

subplot(1, 2, 2); 
plot(DC); 
title('DC');

%% Eliminate 1-P coefficients

wavelet_x_filtered = eliminate2(C1, 1); 
dc_x_filtered = eliminate2(DC, 1); 

figure; 
subplot(1, 2, 1);
plot(wavelet_x_filtered); 
title('wavelet elimination');
subplot(1, 2, 2);
plot(dc_x_filtered); 
title('DC elimination');

%% WAVELET RECONSTRUCTION
wavelet_reconstructed = waverec2(wavelet_x_filtered, S1, 'haar'); 
PNSR = fun_PSNR(f, reshape(wavelet_reconstructed, 256, 256))


%% DISCRETE COSINE RECONSTRUCTION
dc_reconstructed = reshape(idct2(dc_x_filtered),size(f)); 
PNSR = fun_PSNR(f, dc_reconstructed)

%% Display both images
close all; 
figure; 
subplot(1, 3, 1); 
imshow(img); 
title('Original');

subplot(1, 3, 2); 
imshow(uint8(reshape(wavelet_reconstructed, 256, 256))); 
title('Reconstructed wavelet, p = 0.2');

subplot(1, 3, 3); 
imshow(uint8(dc_reconstructed)); 
title('Reconstructed DC, p = 0.2');


