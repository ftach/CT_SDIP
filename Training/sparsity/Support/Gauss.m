function [ x ] = Gauss( x, mu1, tau )


LL = (1/mu1)*( x - (1/(1+tau))*AT(A(x)));

[~,~,L] = size(x);

f = fspecial('gaussian',5,0.75);

for i=1:L
    LL1(:,:,i) = imfilter(LL(:,:,i),f);
end

x = LL - 0.05*LL1;

end

