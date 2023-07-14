function [ Y,sigma ] = SNR( Y,snr )

          snraux=10^(snr/10);                     % Noise addition

          sigma=mean(Y(:))/snraux;

          noise=randn(size(Y))*sigma;

          Y=Y+noise;

          Y=(Y).*(Y>0);

    

end

