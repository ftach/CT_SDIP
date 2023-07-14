function [ stopCriterion,tolA,tolD,debias,maxiter,...
    maxiter_debias,miniter,miniter_debias,enforceMonotone,alphamin,...
    alphamax,compute_mse,AT,verbose,continuation,cont_steps...
    ,debias_start,x_debias,mses,true,x] = VariablesIniciales(L1,A,y)


% flag for initial x (can take any values except 0,1,2)
Initial_X_supplied = 3333;
% Set the defaults for the optional parameters
stopCriterion = 3;
tolA = 0.01;
tolD = 0.0001;
debias = 0;
maxiter = 10000;
maxiter_debias = 500;
miniter = 2;
miniter_debias = 5;
init = 0;
enforceMonotone = 1;
alphamin = 1e-30;
alphamax = 1e30;
compute_mse = 0;
AT = 0;
verbose = 1;
continuation = 0;
cont_steps = -1;
firstTauFactorGiven = 0;
true = [];

% Set the defaults for outputs that may not be computed
debias_start = 0;
x_debias = [];
mses = [];

% Read the optional parameters
if (rem(length(L1),2)==1)
  error('Optional parameters should always go by pairs');
else
  for i=1:2:(length(L1)-1)
    switch upper(L1{i})
     case 'STOPCRITERION'
       stopCriterion = L1{i+1};
     case 'TOLERANCEA'       
       tolA = L1{i+1};
     case 'TOLERANCED'
       tolD = L1{i+1};
     case 'DEBIAS'
       debias = L1{i+1};
     case 'MAXITERA'
       maxiter = L1{i+1};
     case 'MAXITERD'
       maxiter_debias = L1{i+1};
     case 'MINITERA'
       miniter = L1{i+1};
     case 'MINITERD'
       miniter_debias = L1{i+1};
     case 'INITIALIZATION'
       if prod(size(L1{i+1})) > 1   % initial x supplied as array
	 init = Initial_X_supplied;    % flag to be used below
	 x = L1{i+1};
       else 
	 init = L1{i+1};
       end
     case 'MONOTONE'
       enforceMonotone = L1{i+1};
     case 'CONTINUATION'
       continuation = L1{i+1};  
     case 'CONTINUATIONSTEPS' 
       cont_steps = L1{i+1};
     case 'FIRSTTAUFACTOR'
       firstTauFactor = L1{i+1};
       firstTauFactorGiven = 1;
     case 'TRUE_X'
       compute_mse = 1;
       true = L1{i+1};
     case 'ALPHAMIN'
       alphamin = L1{i+1};
     case 'ALPHAMAX'
       alphamax = L1{i+1};
     case 'AT'
       AT = L1{i+1};
     case 'VERBOSE'
       verbose = L1{i+1};
     otherwise
      % Hmmm, something wrong with the parameter string
      error(['Unrecognized option: ''' L1{i} '''']);
    end;
  end;
end
%%%%%%%%%%%%%%

if (sum(stopCriterion == [0 1 2 3 4 5])==0)
  error(['Unknown stopping criterion']);
end

% if A is a function handle, we have to check presence of AT,
if isa(A, 'function_handle') & ~isa(AT,'function_handle')
  error(['The function handle for transpose of A is missing']);
end 

if ~isa(A, 'function_handle')
  AT = @(x) A'*x;
  A = @(x) A*x;
end
% from this point down, A and AT are always function handles.

% Precompute A'*y since it'll be used a lot
Aty = AT(y);

% Initialization

switch init
    case 0   % initialize at zero, using AT to find the size of x
       x = AT(zeros(size(y)));
    case 1   % initialize randomly, using AT to find the size of x
       x = randn(size(AT(zeros(size(y)))));
    case 2   % initialize x0 = A'*y
       x = Aty; 
    case Initial_X_supplied
       % initial x was given as a function argument; just check size
       if size(A(x)) ~= size(y)
          error('Size of initial x is not compatible with A'); 
       end
    otherwise
       error('Unknown ''Initialization'' option');
end

end

