function [ yn, sigma] = snr_f( y, snr, shots )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

% y       = reshape(y, [], shots);
yn      = zeros(size(y));

for i = 1:size(y,2)
C       = y(:,i);

snraux  = 10^(snr/10); % Noise addition
sigma   = mean(C.^2)/snraux;

noise   = randn(size(C))*sqrt(sigma);
C       = y(:,i) + noise;
% C       = (C).*(C>0);

yn(:,i) = C;

end

yn      = yn(:);

end

