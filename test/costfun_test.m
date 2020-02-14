%% Test cost function
%

%% Preliminaries
% Set path
curpath = path;
curDir = fileparts(mfilename('fullpath'));
fs = strfind(curDir,filesep);
topDir = fullfile(curDir(1:fs(end)));
libDir = fullfile(curDir(1:fs(end)),'lib');
if ~contains(curpath,libDir)
    addpath(libDir);
end

%%
% Number of Monte Carlo runs
nMC = pow2(10);
Init = cell(nMC,1);

%%
% Initialize timer
tic

%%
% Initialize RNG
rng('default')

%% Set initial parameters
% Pulse measurement parameters
T=.05;              % sampling time [ps]
N=256;              % number of sampled points

%%
% Ideal pulse parameters
A=1;                % pulse amplitude [arb. units]
w=0.2;              % pulse width [ps]
t0=N*T/3;           % pulse center [ps]
alpha = 0;          % offset of the reference pulse
beta = .004;        % offset of the sample pulse

%%
% Noise parameters
sigma_alpha=1e-3;   % amplitude noise [units of time-domain peak]
sigma_beta=1e-2;    % multiplicative noise [dimensionless]
sigma_tau=1e-3;     % time base noise [ps]

%%
% Transfer function definition and parameters

tfun = @(theta,w) theta(1)*exp(1i*theta(2)*w*T);

A0 = 0.25;          % amplitude ratio between pulses
eta0 = 2;           % delay between pulses [T]
theta0 = [A0;eta0]; % Initial parameter vector
Np = length(theta0);

%% Generate time array and two ideal pulses, y1 and y2
% Use |THZGEN| to produce reference pulse |y1| and time vector |t|, then
% use |TDTF| to create a transfer matrix that shifts |y1| by |eta0| and
% rescales it by |A0| to give |y2|.
[y1,t]=thzgen(N,T,t0);

y2 = tdtf(tfun,theta0,N,T)*y1;

%% Generate sets of noisy pulses, yn1 and yn2

yn1 = zeros(N, nMC);
yn2 = zeros(N, nMC);

sigmay1 = sigmagen(y1, sigma_alpha, sigma_beta, sigma_tau, T);
sigmay2 = sigmagen(y2, sigma_alpha, sigma_beta, sigma_tau, T);

isigmay1 = isigmagen(y1, sigma_alpha, sigma_beta, sigma_tau, T);
isigmay2 = isigmagen(y2, sigma_alpha, sigma_beta, sigma_tau, T);

for jMC=1:nMC
    
    yn1(:,jMC) = ...
        mvnrnd(y1,sigmay1)';
    yn2(:,jMC) = ...
        mvnrnd(y2,sigmay2)';
    
end
%% Construct LSQ problem structure

LSQFit.x0 = theta0;
LSQFit.lb = [];
LSQFit.ub = [];
LSQFit.solver = 'lsqnonlin';
LSQFit.options = optimoptions('lsqnonlin',...
    'Display','off',...
    'UseParallel',true);

%% Construct MLE problem structure

MLEFit.lb = [];
MLEFit.ub = [];
MLEFit.solver = 'fminunc';
MLEFit.options = optimoptions('fminunc',...
    'Display','off',...
    'UseParallel',true);

%% LSQ parameter estimates

pLSQ = zeros(Np,nMC);
muLSQ = zeros(N,nMC);
resnormLSQ = zeros(1,nMC);
DiagnosticLSQ = struct('exitflag',Init,...
    'jacobian',Init);

for jMC=1:nMC
    
   LSQFit.objective = @(theta) ...
        costfunwofflsq(tfun,theta,yn1(:,jMC),yn2(:,jMC),...
        0,0,sigmay1,sigmay2,T);
    
    [p, resnormLSQ(jMC),~,...
        DiagnosticLSQ(jMC).exitflag,~,~,...
        DiagnosticLSQ(jMC).jacobian] = lsqnonlin(LSQFit);
    
    p = p(:);
    pLSQ(:,jMC) = p;
    
    H = tdtf(tfun,p,N,T);
    
    M1 = eye(N) + (sigmay1*H'*isigmay2*H);
    iM1 = eye(N)/M1;
    M2 = yn1(:,jMC) + sigmay1*H'*isigmay2*yn2(:,jMC);
    
    muLSQ(:,jMC) = iM1*M2;


    str = ['LSQ progress: ', num2str(100*jMC/nMC,'%05.1f')];
    bsp = repmat('\b',1,length(str));
    wht = repmat(' ',1,length(str));
    if jMC==1
        fprintf(1,wht)
    end
    fprintf(1, [bsp,str])
    if jMC==nMC
        fprintf(1,[bsp, 'Done with LSQ\n'])
    end

end

%% MLE parameter estimates

pMLE = zeros(Np,nMC);
muMLE = zeros(N,nMC);
FvalMLE = zeros(nMC,1);
resnormMLE = zeros(1,nMC);
DiagnosticMLE = struct('exitflag', Init,...
    'output', Init, 'grad', Init, 'hessian', Init);

for jMC=1:nMC
    
    MLEFit.x0 = [pLSQ(:,jMC); muLSQ(:,jMC)];

    MLEFit.objective = @(theta) ...
        costfun(tfun,theta(3:end),theta(1:2),yn1(:,jMC),yn2(:,jMC),...
        sigma_alpha, sigma_beta, sigma_tau, T);
    
    [p, FvalMLE(jMC), DiagnosticMLE(jMC).exitflag, ...
        DiagnosticMLE(jMC).output, DiagnosticMLE(jMC).grad] ...
        = fminunc(MLEFit);
    
    p = p(:);
    pMLE(:,jMC) = p(1:2);
    mu = p(3:end);
    muMLE(:,jMC) = mu;

    H = tdtf(tfun, pMLE(:,jMC), N, T);
    psi = H*mu;
    isigmamu = isigmagen(mu, sigma_alpha, sigma_beta, sigma_tau, T);
    isigmapsi = isigmagen(psi, sigma_alpha, sigma_beta, sigma_tau, T);
    
    resnormMLE(jMC) = (yn1(:,jMC) - mu)'*isigmamu*(yn1(:,jMC) - mu) ...
        + (yn2(:,jMC) - psi)'*isigmapsi*(yn2(:,jMC) - psi);

    str = ['MLE progress: ', num2str(100*jMC/nMC,'%05.1f')];
    bsp = repmat('\b',1,length(str));
    wht = repmat(' ',1,length(str));
    if jMC==1
        fprintf(1,wht)
    end
    fprintf(1, [bsp,str])
    if jMC==nMC
        fprintf(1,[bsp, 'Done with MLE\n'])
    end

end


%%
% Stop timer
toc
save('costfun_test_output')