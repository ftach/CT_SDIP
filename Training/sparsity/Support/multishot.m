function y = multishot(M,N,L,dmd,hyperimg)

%  
% multishot.m
%
% This function simulates the CASSI sensing process
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
%   M:          Rows of Hyperspectral datacube
%   N:          Columns of Hyperspectral datacube 
%   L:          Bands of Hyperspectral datacube  
%   dmd:        Coded aperture    
%   hyperimg:   Hyperspectral datacube
%
% 	===== Output =====
%   y:  CASSI measurement
%
% ========================================================

y = zeros(M,N+L-1);
for k = 1:L
    tmp = zeros(M,N+L-1);
    tmp(:,k:N+k-1) = hyperimg(:,:,k).*dmd;  % Band coding and dispersion
    y = y+tmp;                              % FPA integration
end



