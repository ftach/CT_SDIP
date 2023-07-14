function [iter,x,x_debias,objective,times,debias_start,mses,taus]= ...
    GPSR_BB(y,A,tau,varargin)
%
% GPSR_BB version 5.0, December 4, 2007
%
% This function solves the convex problem 
% arg min_x = 0.5*|| y - A x ||_2^2 + tau || x ||_1
% using the algorithm GPSR-BB, described in the following paper
%
% "Gradient Projection for Sparse Reconstruction: Application
% to Compressed Sensing and Other Inverse Problems"
% by Mario A. T. Figueiredo, Robert D. Nowak, Stephen J. Wright,
% Journal of Selected Topics on Signal Processing, December 2007
% (to appear).
%
% 
% -----------------------------------------------------------------------
% Copyright (2007): Mario Figueiredo, Robert Nowak, Stephen Wright
% 
% GPSR is distributed under the terms
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
% Please check for the latest version of the code and paper at
% www.lx.it.pt/~mtf/GPSR
%
%  ===== Required inputs =============
%
%  y: 1D vector or 2D array (image) of observations
%     
%  A: if y and x are both 1D vectors, A can be a 
%     k*n (where k is the size of y and n the size of x)
%     matrix or a handle to a function that computes
%     products of the form A*v, for some vector v.
%     In any other case (if y and/or x are 2D arrays), 
%     A has to be passed as a handle to a function which computes 
%     products of the form A*x; another handle to a function 
%     AT which computes products of the form A'*x is also required 
%     in this case. The size of x is determined as the size
%     of the result of applying AT.
%
%  tau: usually, a non-negative real parameter of the objective 
%       function (see above). It can also be an array, the same 
%       size as x, with non-negative entries; in this  case,
%       the objective function weights differently each element 
%       of x, that is, it becomes
%       0.5*|| y - A x ||_2^2 + tau^T * abs(x)
%
%  ===== Optional inputs =============
%
%  
%  'AT'    = function handle for the function that implements
%            the multiplication by the conjugate of A, when A
%            is a function handle. If A is an array, AT is ignored.
%
%  'StopCriterion' = type of stopping criterion to use
%                    0 = algorithm stops when the relative 
%                        change in the number of non-zero 
%                        components of the estimate falls 
%                        below 'ToleranceA'
%                    1 = stop when the relative 
%                        change in the objective function 
%                        falls below 'ToleranceA'
%                    2 = stop when the norm of the difference between 
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below toleranceA
%                    3 = stop when LCP estimate of relative
%                        distance to solution
%                        falls below 'ToleranceA'
%                    4 = stop when the objective function 
%                        becomes equal or less than toleranceA.
%                    5 = stop when the norm of the difference between 
%                        two consecutive estimates, divided by the norm
%                        of one of them falls below toleranceA
%                    Default = 3.
%
%  'ToleranceA' = stopping threshold; Default = 0.01
% 
%  'Debias'     = debiasing option: 1 = yes, 0 = no.
%                 Default = 0.
%
%  'ToleranceD' = stopping threshold for the debiasing phase:
%                 Default = 0.0001.
%                 If no debiasing takes place, this parameter,
%                 if present, is ignored.
%
%  'MaxiterA' = maximum number of iterations allowed in the
%               main phase of the algorithm.
%               Default = 10000
%
%  'MiniterA' = minimum number of iterations performed in the
%               main phase of the algorithm.
%               Default = 5
%
%  'MaxiterD' = maximum number of iterations allowed in the
%               debising phase of the algorithm.
%               Default = 200
%
%  'MiniterD' = minimum number of iterations to perform in the
%               debiasing phase of the algorithm.
%               Default = 5
%
%  'Initialization' must be one of {0,1,2,array}
%               0 -> Initialization at zero. 
%               1 -> Random initialization.
%               2 -> initialization with A'*y.
%           array -> initialization provided by the user.
%               Default = 0;
%
%  'Monotone' =  enforce monotonic decrease in f, or not? 
%               any nonzero -> enforce monotonicity
%               0 -> don't enforce monotonicity.
%               Default = 1;
%
%  'Continuation' = Continuation or not (1 or 0) 
%                   Specifies the choice for a continuation scheme,
%                   in which we start with a large value of tau, and
%                   then decrease tau until the desired value is 
%                   reached. At each value, the solution obtained
%                   with the previous values is used as initialization.
%                   Default = 0
%
% 'ContinuationSteps' = Number of steps in the continuation procedure;
%                       ignored if 'Continuation' equals zero.
%                       If -1, an adaptive continuation procedure is used.
%                       Default = -1.
% 
% 'FirstTauFactor'  = Initial tau value, if using continuation, is
%                     obtained by multiplying the given tau by 
%                     this factor. This parameter is ignored if 
%                     'Continuation' equals zero.
%                     Default = such that the first tau is equal to
%                               0.5*max(abs(AT(y))).
% 
%  'True_x' = if the true underlying x is passed in 
%                this argument, MSE evolution is computed
%
%  'AlphaMin' = the alphamin parameter of the BB method.
%               (See the paper for details)
%               Default = 1e-30;
%
%  'AlphaMax' = the alphamax parameter of the BB method.
%               (See the paper for details)
%               Default = 1e30;
%
%  'Verbose'  = work silently (0) or verbosely (1)
%
% ===================================================  
% ============ Outputs ==============================
%   x = solution of the main algorithm
%
%   x_debias = solution after the debiasing phase;
%                  if no debiasing phase took place, this
%                  variable is empty, x_debias = [].
%
%   objective = sequence of values of the objective function
%
%   times = CPU time after each iteration
%
%   debias_start = iteration number at which the debiasing 
%                  phase started. If no debiasing took place,
%                  this variable is returned as zero.
%
%   mses = sequence of MSE values, with respect to True_x,
%          if it was given; if it was not given, mses is empty,
%          mses = [].
% ========================================================

% test for number of required parametres
if (nargin-length(varargin)) ~= 3
  error('Wrong number of required parameters');
end

%Inicializacion variables
[stopCriterion,tolA,tolD,debias,maxiter,...
    maxiter_debias,miniter,miniter_debias,enforceMonotone,alphamin,...
    alphamax,compute_mse,AT,verbose,continuation,cont_steps...
    ,debias_start,x_debias,mses,true,x] = VariablesIniciales(varargin,A,y);

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

G = MakeONFilter('Symmlet',8);  % Wavelet transformation basis

if ~isa(A, 'function_handle')
  AT = @(x) A'*x;
  A = @(x) A*x;
end

% initialize u and v
u =  x.*(x >= 0);
v = -x.*(x <  0);

% define the indicator vector or matrix of nonzeros in x
nz_x = (x ~= 0.0);
num_nz_x = sum(nz_x(:));

% start the clock
t0 = cputime;

% store given tau, because we're going to change it in the
% continuation procedure
final_tau = tau;

% store given stopping criterion and threshold, because we're going 
% to change them in the continuation procedure
final_stopCriterion = stopCriterion;
final_tolA = tolA;

if ~continuation
  cont_factors = 1;
  cont_steps = 1;
end

keep_continuation = 1;
cont_loop = 1;
iter = 1;
taus = [];
    
% loop for continuation
while keep_continuation
    

    % Compute and store initial value of the objective function

    resid =  (y - A(x));
    
 
        tau = final_tau * cont_factors(cont_loop);
        if cont_loop == cont_steps
            stopCriterion = final_stopCriterion;
            tolA = final_tolA;
            keep_continuation = 0;
        else
            stopCriterion = 1;
            tolA = 1e-5;
        end
    
    taus = [taus tau];
    
    if verbose
        fprintf(1,'\nSetting tau = %0.5g\n',tau)
    end
    
    % if in first continuation iteration, compute and store
    % initial value of the objective function
    if cont_loop == 1
        alpha = 1.0;
        
        f = 0.5*(resid(:)'*resid(:)) + ...
             sum(tau(:).*u(:)) + sum(tau(:).*v(:));
        objective(1) = f;
        if verbose
            fprintf(1,'Initial obj=%10.6e, alpha=%6.2e, nonzeros=%7d\n',...
                f,alpha,num_nz_x);
        end
    end

    % Compute the initial gradient and the useful 
    % quantity resid_base
    resid_base = y - resid;


    % control variable for the outer loop and iteration counter
    keep_going = 1;

    if verbose
      fprintf(1,'\nInitial obj=%10.6e, nonzeros=%7d\n',f,num_nz_x);
    end

    while keep_going

      % compute gradient de F=[gradu;gradv]
      temp = AT(resid_base);

      term  =  temp - Aty;
      gradu =  term + tau;
      gradv = -term + tau;

      % projection and computation of search direction vector
      %Ecuacion 14. Step 1:
      du = max(u - alpha*gradu, 0.0) - u;
      dv = max(v - alpha*gradv, 0.0) - v;
      dx = du-dv;
      old_u = u; 
      old_v = v;

      % calculate useful matrix-vector product involving dx
      % Ecuación 15
      auv = A(dx);
      dGd = auv(:)'*auv(:);

      if (enforceMonotone==1)
        % monotone variant: calculate minimizer along the direction (du,dv)
        lambda0 = - (gradu(:)'*du(:) + gradv(:)'*dv(:))/(realmin+dGd);
        if lambda0 < 0
          fprintf(' ERROR: lambda0 = %10.3e negative. Quit\n', lambda0);
          return;
        end
        lambda = min(lambda0,1);
      else
        %nonmonotone variant: choose lambda=1
        lambda = 1;
      end

      u = old_u + lambda * du;
      v = old_v + lambda * dv;
      uvmin = min(u,v);
%       uvmin=0;
      u = u - uvmin; 
      v = v - uvmin; 
      x = u - v;
      
      % calculate nonzero pattern and number of nonzeros (do this *always*)
      nz_x_prev = nz_x;
      nz_x = (x~=0.0);
      num_nz_x = sum(nz_x(:));
      
      % update residual and function
      resid = y - resid_base - lambda*auv;
      
      prev_f = f;
      
      f = 0.5*(resid(:)'*resid(:)) +  sum(tau(:).*u(:)) + ...
          sum(tau(:).*v(:));

      % compute new alpha
      dd  = du(:)'*du(:) + dv(:)'*dv(:);  
      if dGd <= 0
        % something wrong if we get to here
        fprintf(1,' dGd=%12.4e, nonpositive curvature detected\n', dGd);
        alpha = alphamax;
      else
        alpha = min(alphamax,max(alphamin,dd/dGd));
      end
      resid_base = resid_base + lambda*auv;    
     
      % print out stuff
      if verbose
         fprintf(1,'It=%4d, obj=%9.5e, alpha=%6.2e, nz=%8d  ',...
             iter, f, alpha, num_nz_x);
      end

      % update iteration counts, store results and times
      iter = iter + 1;
      objective(iter) = f;
      times(iter) = cputime-t0;

      if compute_mse
        err = true - x;
        mses(iter) = (err(:)'*err(:));
      end

      switch stopCriterion
          case 0,
              % compute the stopping criterion based on the change
              % of the number of non-zero components of the estimate
              num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
              if num_nz_x >= 1
                  criterionActiveSet = num_changes_active;
              else
                  criterionActiveSet = tolA / 2;
              end
              keep_going = (criterionActiveSet > tolA);
              if verbose
                  fprintf(1,'Delta n-zeros = %d (target = %e)\n',...
                      criterionActiveSet , tolA)
              end
          case 1,
              % compute the stopping criterion based on the relative
              % variation of the objective function.
              criterionObjective = abs(f-prev_f)/(prev_f);
              keep_going =  (criterionObjective > tolA);
              if verbose
                  fprintf(1,'Delta obj. = %e (target = %e)\n',...
                      criterionObjective , tolA)
              end
          case 2,
              % stopping criterion based on relative norm of step taken
              delta_x_criterion = norm(dx(:))/norm(x(:));
              keep_going = (delta_x_criterion > tolA);
              if verbose
                  fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
                      delta_x_criterion,tolA)
              end
          case 3,
              % compute the "LCP" stopping criterion - again based on the previous
              % iterate. Make it "relative" to the norm of x.
              w = [ min(gradu(:), old_u(:)); min(gradv(:), old_v(:)) ];
              criterionLCP = norm(w(:), inf);
              criterionLCP = criterionLCP / ...
                  max([1.0e-6, norm(old_u(:),inf), norm(old_v(:),inf)]);
              keep_going = (criterionLCP > tolA);
              if verbose
                  fprintf(1,'LCP = %e (target = %e)\n',criterionLCP,tolA)
              end
          case 4,
              % continue if not yeat reached target value tolA
              keep_going = (f > tolA);
              if verbose
                  fprintf(1,'Objective = %e (target = %e)\n',f,tolA)
              end
          case 5,
            % stopping criterion based on relative norm of step taken
            delta_x_criterion = sqrt(dd)/sqrt(x(:)'*x(:));
            keep_going = (delta_x_criterion > tolA);
            if verbose
                fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
                    delta_x_criterion,tolA)
            end
          otherwise,
              error('Unknown stopping criterion');
      end % end of the stopping criteria switch
      
      % take no less than miniter... 
      if iter<=miniter
	      keep_going = 1;
      elseif iter > maxiter %and no more than maxiter iterations  
	        keep_going = 0;
      end
      
    end % end of the main loop of the BB-QP algorithm

    % increment continuation loop counter
    cont_loop = cont_loop+1;
    
end % end of the continuation loop
% Print results

if verbose
   fprintf(1,'\nFinished the main algorithm!\nResults:\n')
   fprintf(1,'||A x - y ||_2^2 = %10.3e\n',resid(:)'*resid(:))
   fprintf(1,'||x||_1 = %10.3e\n',sum(abs(x(:))))
   fprintf(1,'Objective function = %10.3e\n',f);
   nz_x = (x~=0.0); num_nz_x = sum(nz_x(:));
   fprintf(1,'Number of non-zero components = %d\n',num_nz_x);
   fprintf(1,'CPU time so far = %10.3e\n', times(iter));
   fprintf(1,'\n');
end

end


