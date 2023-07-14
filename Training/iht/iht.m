function [x_est] = HIT(x, y, phi, K, mu, maxiter)
% Iterative Hard Thresholding algorithm 
%   ===== Required inputs =====
%
%   y:    observation vector, size N 
%   phi: sampling matrix, size MxN
%   K:      dispersion
%   mu:     regularization parameter
%   maxiter:     maximum number of iterations 
%
% 	===== Output =====
%   x_est:    estimated data 
%
% ========================================================

N = size(phi, 2);
x_est = zeros(N, 1); 
error = zeros(maxiter, 1); 
similarity = zeros(maxiter, 1);  

for i=0:maxiter
    x_old = x_est; 
    x_est = x_old + mu*phi'*(y - phi*x_old); 
    [x_sorted,x_position] = sort(abs(x_est), 'ascend'); 
    x_est(x_position(1:N-K))=0; % thresholding operator
    %error(i+1, 1) = norm(x - x_est); 
    %similarity(i+1, 1) = ssim(x_est, x); 

end 
% figure;
% x = 1:maxiter+1; 
% plot(x, error); 
% hold on; 
% plot(x, similarity); 
% xlabel('Iterations')
% legend('MSE', 'SSIM')
% title('Error over iterations')
% hold off; 
end

