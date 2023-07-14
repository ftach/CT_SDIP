function res = myBasis1(x,H,M,N,L,t)

G=MakeONFilter('Symmlet',8);

    if t==0 %A
        
        f = fspecial('gaussian',5,0.75);    
        for i=1:L
            x(:,:,i) = imfilter(x(:,:,i),f);
        end        
        aux = H*x(:);
        aux = reshape(aux,[M,N+L-1,1]);
        res = KronerDCTdirect(aux,G,M,N+L-1,1);
        res = res(:);

    elseif t==1 %AT
        
        aux = reshape(x,[M,N+L-1]);
        aux = KronerDCTinverse(aux,G,M,N+L-1,1); 
        aux = H'*aux(:);
        res = reshape(aux,[M,N,L]);
        f = fspecial('gaussian',5,0.75);    
        for i=1:L
            res(:,:,i) = imfilter(res(:,:,i),f);
        end
    end
end