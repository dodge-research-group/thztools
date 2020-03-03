%% Monte Carlo simulation of time-domain fit
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

%%
% Number of Monte Carlo runs
nMC = pow2(0);

%% Generate time array and two ideal pulses, y1 and y2
% Use |THZGEN| to produce reference pulse |y1| and time vector |t|, then
% use |TDTF| to create a transfer matrix that shifts |y1| by |eta0| and
% rescales it by |A0| to give |y2|.
[y1,t]=thzgen(N,T,t0);

y2 = tdtf(tfun,theta0,N,T)*y1;

%% Construct problem structure to determine the fit parameters
% In addition to the transfer function parameters, we need an additional
% offset parameter, which we set to zero for the initial guess.
TDFit.x0 = [theta0;0];
TDFit.lb = [];
TDFit.ub = [];
TDFit.solver = 'lsqnonlin';
TDFit.options = optimoptions('lsqnonlin',...
    'Display','Off');

Noise = struct('add',sigma_alpha,...
    'mult',sigma_beta,...
    'time',sigma_tau);

Vx = diag(Noise.add^2 + (Noise.mult*y1).^2 + ...
    (Noise.time*(gradient(y1,t))).^2);
Vy = diag(Noise.add^2 + (Noise.mult*y2).^2 + ...
    (Noise.time*(gradient(y2,t))).^2);

%% Run Monte Carlo simulations

Init = cell(nMC,1);
Np = length(theta0) + 1;
ParmOutTDnoise = zeros(nMC,Np);
resnormTDnoise = zeros(nMC,1);
DiagnosticTDnoise = struct('exitflag',Init,...
    'jacobian',Init);

for jMC=1:nMC
    
    % Generate noisy data
    yn1 = y1 + ...
        Noise.add*randn(N,1) + ...
        Noise.mult*y1.*randn(N,1) + ...
        Noise.time*(gradient(y1,t)).*randn(N,1) + alpha;
    yn2 = y2 +  ...
        Noise.add*randn(N,1) + ...
        Noise.mult*y2.*randn(N,1) + ...
        Noise.time*(gradient(y2,t)).*randn(N,1) + beta;
    
    TDFit.objective = @(theta) ...
        costfunwofflsq(tfun,theta(1:end-1),yn1,yn2,...
        alpha,theta(end),Vx,Vy,T);
    
    [ParmOutTDnoise(jMC,:),...
        resnormTDnoise(jMC),~,...
        DiagnosticTDnoise(jMC).exitflag,~,~,...
        DiagnosticTDnoise(jMC).jacobian] = lsqnonlin(TDFit);
    [Q,R]=qr(DiagnosticTDnoise(jMC).jacobian);
    DiagnosticTDnoise(jMC).covariance = ...
        full(eye(size(R,2))/(R'*R));
    
end

cvTDnoise = ...
    mean(reshape([DiagnosticTDnoise.covariance],Np,Np,nMC),3);

%% Display results

figure('Name','Delay histogram, time-domain fit');
histogram(ParmOutTDnoise(:,2)*T*1000)
xlabel('\eta (fs)')
ylabel('Occurence')

figure('Name','Delay normal probability plot, time-domain fit');
normplot(ParmOutTDnoise(:,2)*T*1000)
xlabel('\eta (fs)')

figure('Name','Amplitude histogram, time-domain fit');
histogram(ParmOutTDnoise(:,1))
xlabel('A')
ylabel('Occurence')

figure('Name','Amplitude normal probability plot, time-domain fit');
normplot(ParmOutTDnoise(:,1))
xlabel('A')

figure('Name','Cost-function histogram, time-domain fit');
histogram(resnormTDnoise)
xlabel('Norm of residuals')
ylabel('Occurence')

figure('Name','Cost-function cumulative distribution, time-domain fit');
ecdf(resnormTDnoise,'bounds','on');
hold on
r = floor(min(resnormTDnoise)):...
    ceil(max(resnormTDnoise));
nu = N-Np;
p = chi2cdf(r,nu);
plot(r,p,'k-')
legend('Empirical','LCB','UCB',...
    '\chi^2(r | N_t - N_p)',...
    'Location','SE')
xlabel('Norm of residuals')
hold off

fprintf('%-30s%.4g\n','nu:',nu)
fprintf('%-30s%.4g\n\n','mean(resnormTD):',mean(resnormTDnoise))

fprintf('%-30s%#.4g\n','mean(A)',mean(ParmOutTDnoise(:,1)))
fprintf('%-30s%#.2g\n',...
    'std(A), actual:',std(ParmOutTDnoise(:,1)))
fprintf('%-30s%#.2g\n\n','std(A), mean estimate:',sqrt(cvTDnoise(1,1)))

fprintf('%-30s%#.4g\n',...
    'mean(eta):',mean(ParmOutTDnoise(:,2))*T)
fprintf('%-30s%#.2g\n',...
    'std(eta), actual:',std(ParmOutTDnoise(:,2))*T)
fprintf('%-30s%#.2g\n\n',...
    'std(eta), mean estimate:',sqrt(cvTDnoise(2,2))*T)


%%
% Stop timer
toc