function H = Atranspose(M,N,L,dmd,Sh)

%  
% Atranspose.m
%
% This function design the forward CASSI sensing function (H matrix)
%
% -----------------------------------------------------------------------
% Copyright (2012): Gonzalo R. Arce
% 
% CASSI_UD is distributed under the terms
% of the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------
% 
%
%   ===== Required inputs =====
%
%   M:      Rows of Hyperspectral datacube
%   N:      Columns of Hyperspectral datacube 
%   L:      Bands of Hyperspectral datacube  
%   dmd:    Coded aperture    
%   Sh:     CASSI measurement length
%
% 	===== Output =====
%   H:    Forward CASSI function (H matrix)
%   Img:  Contain the rows, columns and values of the H matrix
%   Ind:  Indices of H matrix to do the inverse transform (H^-1)
%   Val:  Values of H matrix to do the inverse transform (H^-1)
%
% ========================================================

R1=[]; R2=[]; 
for r=1:L
    img = zeros(M,(N+L-1));
    img(:,(r-1)+1:N+(r-1)) = dmd;
    [a,b] = find(img);
    ax = a(:) + (b(:)-1)*M;
    bx = (r-1)*(M*N) + (b(:)-1-(r-1))*M+a(:);
    R1 = [R1; ax(:)];
    R2 = [R2; bx(:)];

end

H = sparse(R1,R2,1,M*(N+L-1),M*N*L);

