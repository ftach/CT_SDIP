function [ W ] = Filter_Gauss( f, M, N, L )

[m,n]=size(f); 

T1 = [];
T2 = [];
T3 = [];
T_1 = [];
T_2 = [];
T_3 = [];
cont=1;

for j=1:N
    j
    for i=1:M
       
        if i<=floor(m/2)
            P1=(1:i+floor(m/2))';
            i1=m-(floor(m/2)-1)-i;
            i2=m;
        elseif M-i<=floor(m/2)-1
            P1=(i-floor(n/2):M)';
            i1=1;
            i2=floor(m/2)+M-i+1;
        else
            P1=(i-floor(m/2):i+floor(m/2))';
            i1=1;
            i2=m;
        end
        
        if j<=floor(n/2)
            P2=(1:j+floor(n/2))';
            j1=n-(floor(n/2)-1)-j;
            j2=n;
        elseif N-j<=floor(n/2)-1
            P2=(j-floor(n/2):N)';
            j1=1;
            j2=floor(n/2)+N-j+1;
        else
            P2=(j-floor(n/2):j+floor(n/2))';
            j1=1;
            j2=n;
        end
        
        F=f(i1:i2,j1:j2);
        F=F(:);
        P=[];
        
        for k=1:size(P2,1)

            Pl=[P1,P2(k,1)*ones(size(P1,1),1)];
            P=cat(1,P,Pl);
            
        end
        
        aux=P(:,1)+M*(P(:,2)-1);
           
        T_3 = [T_3;F];
        T_2 = [T_2;aux];
        T_1 = [T_1;cont*ones(size(F))];
%         W(cont,aux)=F;
        
        cont=cont+1;
    end  

T1 = [T1;T_1];
T2 = [T2;T_2];
T3 = [T3;T_3];

T_1 = [];
T_2 = [];
T_3 = [];
end

W1 = sparse(T1,T2,T3);

W = [];

for i=1:L
   W = blkdiag(W,W1);
end

end

