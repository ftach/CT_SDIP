function [ tau1 ] = tauSeciter( tau, psnr )
A=[6 6 5 5 4 4 3 3];
C=ones(1,size(A,2)-1);

[~,j]=find(psnr(1,:)==max(psnr(1,:)));

j=j(1,end);

if mod(j,2)==0
   B=[2 3 4 6 7 8 9];
   C=C*A(1,j);
else
   B=[6 7 8 9 2 3 4];
   if j==1
       C(1,1:4)=C(1,1:4)*7;
   else
   C(1,1:4)=C(1,1:4)*A(1,j-1);
   end
   C(1,5:end)=C(1,5:end)*A(1,j+1);
end

%%

if j==1
   C=ones(1,size(A,2)-1);
   C(1,1:4)=C(1,1:4)*(A(1,j)+1);
   C(1,5:end)=C(1,5:end)*(A(1,j));
  for i=1:7
      tau1(1,i)=B(1,i)*10^(-(C(1,i)));
  end
elseif j==size(tau,2)
   C=ones(1,size(A,2)-1);
   C(1,1:4)=C(1,1:4)*A(1,j);
   C(1,5:end)=C(1,5:end)*(A(1,j)-1);
  for i=1:7
      tau1(1,i)=B(1,i)*10^(-(C(1,i)+1));
  end
else
    for i=1:7
      tau1(1,i)=B(1,i)*10^(-(C(1,i)));
  end  
end


end

