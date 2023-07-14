function [R3,Ind,R1At,R2At,H2,Val]=AtransposeT(N,shift,H,dmd,Sh)
    
img = zeros(N,N+shift*(H-1));
if size(dmd,3)==1
    for k = 1:H
        img(1:N,shift*(k-1)+1:N+shift*(k-1),k) = dmd;

    end
else
    
    for k = 1:H
        img(1:N,shift*(k-1)+1:N+shift*(k-1),k) = dmd(:,:,k);
    end

end


at=[];
bt=[];
for r=1:H
    [a,b]=find(img(:,:,r));
    ax=a(:)+(b(:)-1)*N;
    bx=(r-1)*N^2+(b(:)-1-shift*(r-1))*N+a(:);
    at=[at;ax(:)];
    bt=[bt;bx(:)];
end
A=sparse(at,bt,1,Sh,N^2*H);
% A=sparse(Sh,N^2*H);
% for k=1:N
%     for j=1:N2
%         for r=1:H
%             if img(k,j,r)==1
%                 A((j-1)*N+k,(r-1)*N^2+(j-1-shift*(r-1))*N+k)=1;
%             end                   
%         end
%     end
% end

At=A';

[R1At,R2At]=find(At);
H2=max(sum(At));
%Img=compactT(R1,R2,N^2*H,Sh,H2,N);
[R1,R2,R3]=find(A);

[Ind,Val]=compactT(R1,R2,R3,Sh,N^2*H,H,N);
