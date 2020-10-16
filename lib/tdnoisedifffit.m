function [P,fval,Diagnostic] = tdnoisedifffit(x,varargin)
%TDNOISEDIFFFIT computes MLE parameters for the time-domain noise model
%
% Syntax:   P = tdnoisefit(x,Options)
%
% Description:
%
% TDNOISEFIT computes the noise parameters sigma and the underlying signal
% vector mu for the data matrix x, where the columns of x are each noisy
% measurements of mu.
%  
% Inputs:
%   x               Data matrix
%
% Optional inputs:
%   Options         Fit options
%
% Option fields:
%   v0              Initial guess, noise model parameters [3x1 double]
%   mu0             Initial guess, signal vector [Nx1 double]
%   A0              Initial guess, amplitude vector [Mx1 double]
%   eta0            Initial guess, delay vector [Mx1 double]
%   ts              Sampling time [scalar double]
%   Fix             Fixed variables [struct]
%   Ignore          Ignore variables [struct]
%
% Outputs:
%   P               Output parameter structure
%       .logv       Log of noise parameters
%       .mu         Signal vector
%       .A          Amplitude vector
%       .eta        Delay vector
%       .ts         Samling time
%   fval            Value of NLL cost function from FMINUNC
%   Diagnostic      Structure of diagnostic information
%       .exitflag	Exit flag from FMINUNC
%       .output     Output from FMINUNC
%       .grad     	NLL cost function gradient from FMINUNC
%       .hessian   	NLL cost function hessian from FMINUNC
%

% Parse inputs
[N,M] = size(x);
Input = inputParser;
addRequired(Input,'x', ...
    @(x) validateattributes(x,{'double'},{'2d'}));
addOptional(Input,'Options', struct, ...
    @(Options) validateattributes(Options,{'struct'},{'nonempty'}));
parse(Input,x,varargin{:})
Options = Input.Results.Options;

% Parse options
InputOptions = inputParser;
addParameter(InputOptions,'v0',mean(var(x,1,2))*ones(3,1), ...
    @(v0) validateattributes(v0,{'double'},{'vector','>',0}));
addParameter(InputOptions,'mu0',mean(x,2), ...
    @(mu0) validateattributes(mu0,{'double'},{'vector'}));
addParameter(InputOptions,'A0',ones(M,1), ...
    @(A0) validateattributes(A0,{'double'},{'vector'}));
addParameter(InputOptions,'eta0',zeros(M,1), ...
    @(eta0) validateattributes(eta0,{'double'},{'vector'}));
addParameter(InputOptions,'ts',1, ...
    @(ts) validateattributes(ts,{'numeric'},{'scalar','positive'}));
addParameter(InputOptions,'Fix', ...
    struct('logv',false,'mu',false,'A',true,'eta',true),...
    @(Fix) validateattributes(Fix,{'struct'},{'nonempty'}))
addParameter(InputOptions,'Ignore', ...
    struct('A',true,'eta',true),...
    @(Ignore) validateattributes(Ignore,{'struct'},{'nonempty'}))
parse(InputOptions,Options)

v0 = InputOptions.Results.v0(:);
mu0 = InputOptions.Results.mu0(:);
A0 = InputOptions.Results.A0(:);
eta0 = InputOptions.Results.eta0(:);
ts = InputOptions.Results.ts;
Fix = InputOptions.Results.Fix;
Ignore = InputOptions.Results.Ignore;

% Define optimization problem
MLE.solver = 'fminunc';
MLE.options = optimoptions('fminunc',...
    'SpecifyObjectiveGradient',false,...
    'Algorithm','quasi-newton',...
    'FiniteDifferenceType','central',...
    'FiniteDifferenceStepSize', 1e-7,...
    'StepTolerance',1e-8,...
    'OptimalityTolerance',1e-8,...
    'MaxFunctionEvaluations',1e6,...
    'Display','off',...
    'UseParallel',true,...
    'Diagnostics','off');
% ,...
%     'FiniteDifferenceType','central',...
%     'FiniteDifferenceStepSize', 1e-8,...
%     'SubproblemAlgorithm', 'cg', ...
%     'FiniteDifferenceType','central',...
%     'FiniteDifferenceStepSize', 1e-7,...
%     'CheckGradients',true,...
%     'StepTolerance',1e-8,...
%     'FunctionTolerance',1e-8,...
%     'OptimalityTolerance',1e-8,...
%     'MaxFunctionEvaluations',1e6,...
%     'UseParallel',true,...
MLE.x0 = [];
idxStart = 1;
if Fix.logv
    setPlogv = @(p) log(v0);
else
    MLE.x0 = [MLE.x0;log(v0)];
    idxEnd = idxStart+2;
    idxRange = idxStart:idxEnd;
    setPlogv = @(p) p(idxRange);
    idxStart = idxEnd + 1;
end
if Fix.mu
    setPmu = @(p) mu0;
else
    MLE.x0 = [MLE.x0;mu0];
    idxEnd = idxStart+N-1;
    idxRange = idxStart:idxEnd;
    setPmu = @(p) p(idxRange);
    idxStart = idxEnd + 1;
end
if Ignore.A
    setPA = @(p) [];
elseif Fix.A
    setPA = @(p) A0(:);
else
    MLE.x0 = [MLE.x0;A0(2:end)];
    idxEnd = idxStart+M-2;
    idxRange = idxStart:idxEnd;
    setPA = @(p) [A0(1);p(idxRange)];
    idxStart = idxEnd + 1;
end
if Ignore.eta
    setPeta = @(p) [];
elseif Fix.eta
    setPeta = @(p) eta0(:);
else
    MLE.x0 = [MLE.x0;eta0(2:end)];
    idxEnd = idxStart+M-2;
    idxRange = idxStart:idxEnd;
    setPeta = @(p) [eta0(1);p(idxRange)];
end

D = tdtf(@(theta,w) -1i*w,0,N,ts);

parseIn = @(p) struct('logv',setPlogv(p),'mu',setPmu(p),...
    'A',setPA(p),'eta',setPeta(p),'ts',ts,'D',D);
MLE.objective = @(p) tdnllfixedeffects(x,parseIn(p),Fix);

[pOut,fval,exitflag,output,grad,hessian] = fminunc(MLE);

% Parse output
idxStart = 1;
if Fix.logv
    P.var = v0;
else
    idxEnd = idxStart+2;
    idxRange = idxStart:idxEnd;
    idxStart = idxEnd + 1;
    P.var = exp(pOut(idxRange));
end
if Fix.mu
    P.mu = mu0;
else
    idxEnd = idxStart+N-1;
    idxRange = idxStart:idxEnd;
    idxStart = idxEnd + 1;
    P.mu = pOut(idxRange);
end
if Ignore.A || Fix.A
    P.A = A0;
else
    idxEnd = idxStart+M-2;
    idxRange = idxStart:idxEnd;
    idxStart = idxEnd + 1;
    P.A = [A0(1);pOut(idxRange)];
end
if Ignore.eta || Fix.eta
    P.eta = eta0;
else
    idxEnd = idxStart+M-2;
    idxRange = idxStart:idxEnd;
    P.eta = [eta0(1);pOut(idxRange)];
end

P.ts = ts;

if nargout > 2
    Diagnostic.exitflag = exitflag;
    Diagnostic.output = output;
    Diagnostic.grad = grad;
    Diagnostic.hessian = hessian;
    Diagnostic.Err = struct('var',[],'mu',[],'A',[],'eta',[]);
    
    paramTest = ~[Fix.logv;...
        Fix.mu;...
        (Fix.A || Ignore.A) ;...
        (Fix.eta || Ignore.eta)];
    
    Nparam = sum(paramTest.*[3;N;M-1;M-1]);
    V = speye(Nparam)/Diagnostic.hessian;
    err = sqrt(diag(V));
    
    idxStart = 1;
    if paramTest(1)
        Diagnostic.Err.var = ...
            sqrt(diag(diag(P.var)*V(1:3,1:3)*diag(P.var)));
        idxStart = idxStart + 3;
    end
    if paramTest(2)
        Diagnostic.Err.mu = err(idxStart+(0:N-1));
        idxStart = idxStart + N;
    end
    if paramTest(3)
        Diagnostic.Err.A = err(idxStart+(0:M-2));
        idxStart = idxStart + M - 1;
    end
    if paramTest(4)
        Diagnostic.Err.eta = err(idxStart+(0:M-2));
    end

end

%% TDNLLFIXEDEFFECTS
    function [nll,gradnll] = tdnllfixedeffects(x,P,Fix)
    % Strip first element of gradient vector wrt A and eta
        
        [nll,gradnll] = tdnll(x,P,Fix);
        idxStart = 1;
        if ~Fix.logv
            idxStart = idxStart + 3;
        end
        if ~Fix.mu
            idxStart = idxStart + N;
        end
        if ~Fix.A
            gradnll(idxStart) = [];
            idxStart = idxStart + M - 1;
        end
        if ~Fix.eta
            gradnll(idxStart) = [];
            idxStart = idxStart + M - 1;
        end        
    end

end