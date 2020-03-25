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
nMC = pow2(9);
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
t0=N*T/3;           % pulse center [ps]

f = fftfreq(N, T);
w = 2*pi*f;

%%
% Noise parameters
sigma_alpha=1e-4;   % amplitude noise [units of time-domain peak]
sigma_beta=1e-2;    % multiplicative noise [dimensionless]
sigma_tau=1e-3;     % time base noise [ps]

sigma = [sigma_alpha; sigma_beta; sigma_tau];

%%
% Transfer function definition and parameters

tfun = @(theta,w) theta(1)*exp(1i*theta(2)*w*T);

% A0 = 0.25;          % amplitude ratio between pulses
% eta0 = 2;           % delay between pulses [T]
A0 = 1.;          % amplitude ratio between pulses
eta0 = 0.;           % delay between pulses [T]
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

Vy1 = diag(noisevar(sigma, y1, T));
Vy2 = diag(noisevar(sigma, y2, T));

for jMC=1:nMC
    
    yn1(:,jMC) = mvnrnd(y1,Vy1)';
    yn2(:,jMC) = mvnrnd(y2,Vy2)';
    
end

%% Construct LSQ problem structure

LSQFit.x0 = theta0;
LSQFit.lb = [];
LSQFit.ub = [];
LSQFit.solver = 'lsqnonlin';
LSQFit.options = optimoptions('lsqnonlin',...
    'Display','off',...
    'UseParallel',true);

%% LSQ parameter estimates

pLSQ = zeros(Np,nMC);
residualLSQ = zeros(N,nMC);
resnormLSQ = zeros(1,nMC);
DiagnosticLSQ = struct('exitflag',Init,...
    'jacobian',Init);

for jMC=1:nMC
    
    sigmayn1 = noiseamp(sigma, yn1(:,jMC), T);
    sigmayn2 = noiseamp(sigma, yn2(:,jMC), T);
    LSQFit.objective = @(theta) ...
        costfunlsq(tfun,theta,yn1(:,jMC),yn2(:,jMC),sigmayn1,sigmayn2,w);
    
    [p, resnormLSQ(jMC), residualLSQ(:,jMC),...
        DiagnosticLSQ(jMC).exitflag,~,~,...
        DiagnosticLSQ(jMC).jacobian] = lsqnonlin(LSQFit);
    
    p = p(:);
    pLSQ(:,jMC) = p;
    
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


%%
% Stop timer
toc
save('costfun_test_lsq_output')