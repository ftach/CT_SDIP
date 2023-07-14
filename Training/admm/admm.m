function [theta_est] = admm(theta, y, H, psi, D, lambda, mu, rho, maxiter)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%   ===== Required inputs =====
%
%   y: observation vector, size N 
%   H: sampling matrix, size MxN
%   psi: DCT basis, size NxN 
%   D: Difference matrix 
%   lambda: regularization parameter 1, real > 0
%   mu: regularization parameter 2, real > 0
%   rho: regularization parameter 3, real > 0
%   maxiter: maximum number of iterations 
%
% 	===== Output =====
%   theta_est:    estimated data 
%
% ========================================================

% INIT 
N = size(psi, 1); 
theta_est = zeros(N, 1); 
v = zeros(N, 1); 
w  = zeros(N, 1); 
f = zeros(N, 1); 
g = zeros(N, 1); 
mse = zeros(maxiter, 1); 
similarity = zeros(maxiter, 1);  

%% RUN 
for j=1:maxiter
    theta_est = inv(psi'*H'*H*psi + rho*(eye(N) + psi'*D'*D*psi))*(psi'*H'*y + rho*(v-f + psi'*D'*(w-g))); % update theta 
    v = wthresh(theta_est + f,'s',lambda/rho);
    w = wthresh(D*psi*theta_est + g,'s',mu/rho);
    f = f + theta_est - v ; % update f
    g = g + D*psi*theta_est - w; % update g 
    mse(j, 1) = norm(psi*theta - psi*theta_est); 
    similarity(j, 1) = ssim(psi*theta_est, psi*theta); 
end 
figure;
x = 1:maxiter; 
plot(x, mse); 
hold on; 
plot(x, similarity); 
xlabel('Iterations')
legend('MSE', 'SSIM')
title('Error over iterations')
hold off; 
end

