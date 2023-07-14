function res = myBasis(x,H,M,N,L,K,t,My,Ny)

    if t==0 %A
         
        aux = H*x(:);
        aux = reshape(aux,[My*L,Ny,K]);
        res = FourierKronerDCTdirect(aux,My*L,Ny,K);  
        res = res(:);

    elseif t==1 %AT
        
        aux = reshape(x,[My*L,Ny,K]);
        aux = FourierKronerDCTinverse(aux,My*L,Ny,K);
        aux = H'*aux(:);
        res = reshape(aux,[M,N,L]);    
 
        
    end
end