function [x,x_debias,objective,times,debias_start,mses,taus]= ...
    SpaRSA(y,A,tau,varargin)

% start the clock
t0 = cputime;
times(1) = cputime - t0;

% test for number of required parametres
if (nargin-length(varargin)) ~= 3
     error('Wrong number of required parameters');
end
G = MakeONFilter('Symmlet',8);
% Set the defaults for the optional parameters
stopCriterion = 2;
tolA = 0.01;
tolD = 0.0001;
debias = 0;
maxiter = 10000;
maxiter_debias = 200;
miniter = 5;
miniter_debias = 0;
init = 0;
bbVariant = 1;
bbCycle = 1;
enforceMonotone = 0;
enforceSafeguard = 0;
M = 5;
sigma = .01;
alphamin = 1e-30;
alphamax = 1e30;
compute_mse = 0;
AT = 0;
verbose = 1;
continuation = 0;
cont_steps = -1;
psi_ok = 0;
% amount by which to increase alpha after an unsuccessful step
eta = 2.0;
% amount by which to decrease alpha between iterations, if a
% Barzilai-Borwein rule is not used to make the initial guess at each
% iteration. 
alphaFactor = 0.8;
phi_l1 = 0;

% Set the defaults for outputs that may not be computed
debias_start = 0;
x_debias = [];
mses = [];

% Read the optional parameters
if (rem(length(varargin),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(varargin)-1)
    switch upper(varargin{i})
     case 'PSI'
       psi_function = varargin{i+1};
     case 'PHI'
       phi_function = varargin{i+1};
     case 'STOPCRITERION'
       stopCriterion = varargin{i+1};
     case 'TOLERANCEA'       
       tolA = varargin{i+1};
     case 'TOLERANCED'
       tolD = varargin{i+1};
     case 'DEBIAS'
       debias = varargin{i+1};
     case 'MAXITERA'
       maxiter = varargin{i+1};
     case 'MAXITERD'
       maxiter_debias = varargin{i+1};
     case 'MINITERA'
       miniter = varargin{i+1};
     case 'MINITERD'
       miniter_debias = varargin{i+1};
     case 'INITIALIZATION'
       if prod(size(varargin{i+1})) > 1   % we have an initial x
	 init = 33333;    % some flag to be used below
	 x = varargin{i+1};
       else 
	 init = varargin{i+1};
       end
     case 'BB_VARIANT'
       bbVariant = varargin{i+1};
     case 'BB_CYCLE'
       bbCycle = varargin{i+1};
     case 'MONOTONE'
       enforceMonotone = varargin{i+1};
     case 'SAFEGUARD'
       enforceSafeguard = varargin{i+1};
     case 'M'
       M = varargin{i+1};
     case 'SIGMA'
       sigma = varargin{i+1};
     case 'ETA'
       eta = varargin{i+1};
     case 'ALPHA_FACTOR'
       alphaFactor = varargin{i+1};
     case 'CONTINUATION'
       continuation = varargin{i+1};  
     case 'CONTINUATIONSTEPS' 
       cont_steps = varargin{i+1};
     case 'FIRSTTAUFACTOR'
       firstTauFactor = varargin{i+1};
     case 'TRUE_X'
       compute_mse = 1;
       true = varargin{i+1};
     case 'ALPHAMIN'
       alphamin = varargin{i+1};
     case 'ALPHAMAX'
       alphamax = varargin{i+1};
     case 'AT'
       AT = varargin{i+1};
     case 'VERBOSE'
       verbose = varargin{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' varargin{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

% if A is a matrix, we find out dimensions of y and x,
% and create function handles for multiplication by A and A',
% so that the code below doesn't have to distinguish between
% the handle/not-handle cases
if ~isa(A, 'function_handle')
   AT = @(x) A'*x;
   A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

psi_function = @(x,tau) soft(x,tau);
phi_function = @(x) sum(abs(x(:))); 
phi_l1 = 1;

% Initialization
switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(zeros(size(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = Aty; 
    otherwise
       error(['Unknown ''Initialization'' option']);
end
 
% if tau is large enough, in the case of phi = l1, thus psi = soft,
% the optimal solution is the zero vector
if phi_l1
   aux = AT(y);
   max_tau = max(abs(aux(:)));
   firstTauFactor = 0.8*max_tau / tau;
end

% define the indicator vector or matrix of nonzeros in x
nz_x = (x ~= 0.0);
num_nz_x = sum(nz_x(:));

% store given tau, because we're going to change it in the
% continuation procedure
final_tau = tau;
% if we choose to use adaptive continuation, need to reset tau to realmax to
% make things work (don't ask...)
if cont_steps == -1
  tau = realmax;
end

% store given stopping criterion and threshold, because we're going 
% to change them in the continuation procedure
final_stopCriterion = stopCriterion;
final_tolA = tolA;

% set continuation factors
cont_factors = 1;
cont_steps = 1;
keep_continuation = 1;
cont_loop = 1;
iter = 1;
taus = [];
G = MakeONFilter('Symmlet',8);  % Wavelet transformation basis

%%

% loop for continuation
while keep_continuation 
  
  % initialize the count of steps since last update of alpha 
  % (for use in cyclic BB)
  iterThisCycle = 0;
  
  % Compute the initial residual and gradient
  resid =  A(x) - y;

  gradq = AT(resid);
  
  if cont_steps == -1
     
     temp_tau = max(final_tau,0.2*max(abs(gradq(:))));
     
     if temp_tau > tau
        tau = final_tau;    
     else
        tau = temp_tau;
     end
     
     if tau == final_tau
        stopCriterion = final_stopCriterion;
        tolA = final_tolA;
        keep_continuation = 0;
     else
        stopCriterion = 1;
        tolA = 1e-5;
     end
  else
     tau = final_tau * cont_factors(cont_loop);
     if cont_loop == cont_steps
        stopCriterion = final_stopCriterion;
        tolA = final_tolA;
        keep_continuation = 0;
     else
        stopCriterion = 1;
        tolA = 1e-5;
     end
  end
  
  taus = [taus tau];
  
  if verbose
    fprintf('\n Regularization parameter tau = %10.6e\n',tau)
  end
  
  % compute and store initial value of the objective function 
  % for this tau
  alpha = 1; %1/eps;
  
  f = 0.5*(resid(:)'*resid(:)) + tau * phi_function(x);
  if enforceSafeguard
    f_lastM = f;
  end
 
  % initialization of alpha
  % alpha = 1/max(max(abs(du(:))),max(abs(dv(:))));
  % or just do a dumb initialization 
  %alphas(iter) = alpha;
  
  % control variable for the outer loop and iteration counter
  keep_going = 1;
 
  while keep_going
      
    % compute gradient
    gradq = AT(resid);
%     gradq = AT(resid);
    
    % save current values
    prev_x = x;
    prev_f = f;
    prev_resid = resid;
    % computation of step
    
    cont_inner = 1;
    while cont_inner
%     x = psi_function(prev_x - gradq*(1/alpha) ,tau/alpha);
   
    T = tau/alpha;
    x1_1 = prev_x - gradq*(1/alpha);
    if sum(abs(T(:)))==0
       x = x1_1;
    else
       x = max(abs(x1_1) - T, 0);
       x = x./(x+T) .* x1_1;
    end

      dx = x - prev_x;
      Adx = A(dx);
      resid = prev_resid + Adx;
      
    f = 0.5*(resid(:)'*resid(:)) + tau * phi_function(x);
      if enforceMonotone
	f_threshold = prev_f;
      elseif enforceSafeguard
	f_threshold = max(f_lastM) - 0.5*sigma*alpha*(dx(:)'*dx(:));
      else
	f_threshold = inf;
      end
       % f_threshold
      
      if f <= f_threshold
	cont_inner=0;
      else
	% not good enough, increase alpha and try again
	alpha = eta*alpha;
      end
    end   % of while cont_inner
    if enforceSafeguard
      if length(f_lastM)<M+1
	f_lastM = [f_lastM f];
      else
	f_lastM = [f_lastM(2:M+1) f];
      end
    end
    
    if bbVariant==1
      % standard BB choice of initial alpha for next step
      if iterThisCycle==0 | enforceMonotone==1
	dd  = dx(:)'*dx(:);  
	dGd = Adx(:)'*Adx(:);
	alpha = min(alphamax,max(alphamin,dGd/(realmin+dd)));
      end
    elseif bbVariant==2
      % alternative BB choice of initial alpha for next step
      if iterThisCycle==0 | enforceMonotone==1
	dd  = dx(:)'*dx(:);  
	dGd = Adx(:)'*Adx(:);
	ATAdx=AT(Adx);
	dGGd = ATAdx(:)'*ATAdx(:);
	alpha = min(alphamax,max(alphamin,dGGd/(realmin+dGd)));
      end
    else  
      % reduce current alpha to get initial alpha for next step
      alpha = alpha * alphaFactor;
    end

    % update iteration counts, store results and times
    iter=iter+1
    iterThisCycle=mod(iterThisCycle+1,bbCycle);
    objective(iter) = f;
    times(iter) = cputime-t0;
    % alphas(iter) = alpha;
    if compute_mse
      err = true - x;
      mses(iter) = (err(:)'*err(:));
    end
    
    % compute stopping criteria and test for termination
    switch stopCriterion
        case 0,
            % compute the stopping criterion based on the change
            % of the number of non-zero components of the estimate
            nz_x_prev = nz_x;
            nz_x = (abs(x)~=0.0);
            num_nz_x = sum(nz_x(:));
            num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
            if num_nz_x >= 1
                criterionActiveSet = num_changes_active / num_nz_x;
                keep_going = (criterionActiveSet > tolA);
            end
            if verbose
%                 fprintf(1,'Delta nz = %d (target = %e)\n',...
%                     criterionActiveSet , tolA)
            end
        case 1,
            % compute the stopping criterion based on the relative
            % variation of the objective function.
            criterionObjective = abs(f-prev_f)/(prev_f);
            keep_going =  (criterionObjective > tolA);
            if verbose
%                 fprintf(1,'Delta obj. = %e (target = %e)\n',...
%                     criterionObjective , tolA)
            end
        case 2,
            % compute the "duality" stopping criterion - actually based on the
            % iterate PRIOR to the step just taken. Make it relative to the primal
            % function value.
            scaleFactor = norm(gradq(:),inf);
            w = (tau*prev_resid(:)) / scaleFactor;
            criterionDuality = 0.5* (prev_resid(:)'*prev_resid(:)) + ...
                tau * phi_function(prev_x) + 0.5*w(:)'*w(:) + y(:)'*w(:);
            criterionDuality = criterionDuality / prev_f;
            keep_going = (criterionDuality > tolA);
            if verbose
%                 fprintf(1,'Duality = %e (target = %e)\n',...
%                     criterionDuality , tolA)
            end
        case 3,
            % compute the "LCP" stopping criterion - again based on the previous
            % iterate. Make it "relative" to the norm of x.
            w = [ min(tau + gradq(:), max(prev_x(:),0.0)); ...
                min(tau - gradq(:), max(-prev_x(:),0.0))];
            criterionLCP = norm(w(:), inf);
            criterionLCP = criterionLCP / max(1.0e-6, norm(prev_x(:),inf));
            keep_going = (criterionLCP > tolA);
%             if verbose
%                 fprintf(1,'LCP = %e (target = %e)\n',criterionLCP,tolA)
%             end
        case 4,
            % continue if not yeat reached target value tolA
            keep_going = (f > tolA);
%             if verbose
%                 fprintf(1,'Objective = %e (target = %e)\n',f,tolA)
%             end
        case 5,
            % stopping criterion based on relative norm of step taken
            delta_x_criterion = sqrt(dx(:)'*dx(:))/(x(:)'*x(:));
            keep_going = (delta_x_criterion > tolA);
%             if verbose
%                 fprintf(1,'Norm(delta x)/norm(x) = %e (target = %e)\n',...
%                     delta_x_criterion,tolA)
%             end
        otherwise,
            error(['Unknown stopping criterion']);
    end % end of the stopping criteria switch
    
    % overrule the stopping decision to ensure we take between miniter and
    % maxiter iterations
    if iter<=miniter
      % take no fewer than miniter... 
      keep_going = 1;
    elseif iter > maxiter
      % and no more than maxiter iterations  
      keep_going = 0;
    end
    
  end % end of the main loop of the GPBB algorithm (while keep_going)
  
  cont_loop = cont_loop + 1;
  
end % end of the continuation loop (while keep_continuation) 


%% 





% Print results
% if verbose
%   fprintf(1,'\nFinished the main algorithm!  Results:\n')
%   fprintf(1,'Number of iterations = %d\n',iter)
%   fprintf(1,'0.5*||A x - y ||_2^2 = %10.3e\n',0.5*resid(:)'*resid(:))
%   fprintf(1,'tau * Penalty = %10.3e\n',tau * phi_function(x))
%   fprintf(1,'Objective function = %10.3e\n',f);
%   fprintf(1,'Number of non-zero components = %d\n',sum(x(:)~=0));
%   fprintf(1,'CPU time so far = %10.3e\n', times(iter));
%   fprintf(1,'\n');
% end

% If the 'Debias' option is set to 1, we try to
% remove the bias from the l1 penalty, by applying CG to the 
% least-squares problem obtained by omitting the l1 term 
% and fixing the zero coefficients at zero.

if (debias & (sum(x(:)~=0)~=0))
  if verbose
    fprintf(1,'\nStarting the debiasing phase...\n\n')
  end
  
  x_debias = x;
  zeroind = (x_debias~=0); 
  cont_debias_cg = 1;
  debias_start = iter;
  
  % calculate initial residual
  resid = A(x_debias);
  resid = resid-y;
  prev_resid = eps*ones(size(resid));
  
  rvec = AT(resid);
  
  % mask out the zeros
  rvec = rvec .* zeroind;
  rTr_cg = rvec(:)'*rvec(:);
  
  % set convergence threshold for the residual || RW x_debias - y ||_2
  tol_debias = tolD * (rvec(:)'*rvec(:));
  
  % initialize pvec
  pvec = -rvec;
  
  % main loop
  while cont_debias_cg
    
    % calculate A*p = Wt * Rt * R * W * pvec
    RWpvec = A(pvec);      
    Apvec = AT(RWpvec);
    
    % mask out the zero terms
    Apvec = Apvec .* zeroind;
    
    % calculate alpha for CG
    alpha_cg = rTr_cg / (pvec(:)'* Apvec(:));
    
    % take the step
    x_debias = x_debias + alpha_cg * pvec;
    resid = resid + alpha_cg * RWpvec;
    rvec  = rvec  + alpha_cg * Apvec;
    
    rTr_cg_plus = rvec(:)'*rvec(:);
    beta_cg = rTr_cg_plus / rTr_cg;
    pvec = -rvec + beta_cg * pvec;
    
    rTr_cg = rTr_cg_plus;
    
    iter = iter+1;
    
    objective(iter) = 0.5*(resid(:)'*resid(:)) + ...
	                  tau * phi_function(x_debias(:));
    times(iter) = cputime - t0;
    
    if compute_mse
      err = true - x_debias;
      mses(iter) = (err(:)'*err(:));
    end
    
    % in the debiasing CG phase, always use convergence criterion
    % based on the residual (this is standard for CG)
    if verbose
       fprintf(1,'t = %5d, debias resid = %13.8e, convergence = %8.3e\n', ...
	   iter, resid(:)'*resid(:), rTr_cg / tol_debias);
    end
    cont_debias_cg = ...
     	(iter-debias_start <= miniter_debias )| ...
	    ((rTr_cg > tol_debias) & ...
	    (iter-debias_start <= maxiter_debias));
    
  end
  if verbose
  fprintf(1,'\nFinished the debiasing phase! Results:\n')
  fprintf(1,'Final number of iterations = %d\n',iter);
  fprintf(1,'0.5*||A x - y ||_2 = %10.3e\n',0.5*resid(:)'*resid(:))
  fprintf(1,'tau * penalty = %10.3e\n',tau * phi_function(x))
  fprintf(1,'Objective function = %10.3e\n',f);
  fprintf(1,'Number of non-zero components = %d\n',...
          sum((x_debias(:)~=0.0)));
  fprintf(1,'CPU time so far = %10.3e\n', times(iter));
  fprintf(1,'\n');
  end
end

if compute_mse
  mses = mses/length(true(:));
end

