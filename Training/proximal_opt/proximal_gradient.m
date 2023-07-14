% Here we choose x as a sparse matrix and A a random matrix
%% READ IMAGE
x = imread('mnist_5_orig.png');
x = imresize(x, [32, 32]);
x = 255 - rgb2gray(x);
x = double(x);
x = x./max(x(:));
x = reshape(x, [], 1); 
N = size(x, 1); 
K = floor(0.1*N); 
[x_sorted,x_position] = sort(abs(x), 'descend'); 
x(x_position(K+1:end))=0; 
imshow(reshape(x, [32, 32])); 
nnz(x)


%% PARAMETERS 
%r = 4;
%M = N / r ; 
M = floor(nnz(x)*log(N)); 
A = randn(M, N); 
y = A*x ; 

k = 0; 
x_new = zeros(size(x));
tau = 0.1;
epsilon = 1e-6;
maxiter = 100; 
%% RUN 
%while norm(x_new - x_old)> epsilon
for k=1:maxiter
    x_old = x_new; 
%     x_new = x_old - tau*(-2*A'*y + 2*A'*A*x_old);
    x_new = wthresh(x_old - tau*A'*(A*x_old - y), 's', tau); 
%     [x_sorted,x_position] = sort(abs(x_new), 'descend'); 
%     x_new(x_position(K+1:end))=0; 
    %k = k + 1;
end 

close all; 
figure; 
imshow(reshape(x, 32, 32))
figure;
imshow(reshape(x_new, 32, 32))