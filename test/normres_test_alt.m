%% Test procedure for producing normalized residuals
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
nMC = pow2(6);
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
f = fftfreq(N,T);   % discrete frequency vector
w = 2*pi*f;         % discrete angular frequency vector

%%
% Ideal pulse parameters
A=1;                % pulse amplitude [arb. units]
tw=0.2;             % pulse width [ps]
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

d = 20e-9;          % Film thickness [m]
n = 3.4;            % Substrate refractive index
sigma0 = 1e7;       % DC conductivity [1/(Ohm m)]
tau = 1;            % Momentum relaxation rate [ps]

gamma = d*SI.Z0/(n+1);

tfun = @(theta,w) 1./(1 + gamma*theta(1)./(1 -1i*w*theta(2)));

theta0 = [sigma0; tau]; % Initial parameter vector
Np = length(theta0);

%% Generate time array and two ideal pulses, y1 and y2
% Use |THZGEN| to produce reference pulse |y1| and time vector |t|, then
% use |TDTF| to create the ideal transformed pulse.
[y1,t]=thzgen(N,T,t0);

y2 = tdtf(tfun,theta0,N,T)*y1;

%% Generate sets of noisy pulses, yn1 and yn2

yn1 = zeros(N, nMC);
yn2 = zeros(N, nMC);

sigmay1 = sigmagen(y1, sigma_alpha, sigma_beta, sigma_tau, T);
sigmay2 = sigmagen(y2, sigma_alpha, sigma_beta, sigma_tau, T);

for jMC=1:nMC
    
    yn1(:,jMC) = mvnrnd(y1,sigmay1)';
    yn2(:,jMC) = mvnrnd(y2,sigmay2)';
    
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
muLSQ = zeros(N,nMC);
resnormLSQ = zeros(1,nMC);
residualLSQ = zeros(N,nMC);
DiagnosticLSQ = struct('exitflag',Init,...
    'jacobian',Init);

for jMC=1:nMC
    
    sigmayn1 = sqrt(diag(...
        sigmagen(yn1(:,jMC), sigma_alpha, sigma_beta, sigma_tau, T)));
    sigmayn2 = sqrt(diag(...
        sigmagen(yn2(:,jMC), sigma_alpha, sigma_beta, sigma_tau, T)));

    LSQFit.objective = @(theta) ...
        costfunlsq_alt(tfun,theta,yn1(:,jMC),yn2(:,jMC),...
        sigmayn1,sigmayn2,w);
    
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

% %% Show residuals
% 
% % Choose index to show
% k = 1;
% 
% % Construct covariance matrices for data
% Vx = sigmagen(yn1(:,k),sigma_alpha, sigma_beta, sigma_tau, T);
% Vy = sigmagen(yn2(:,k),sigma_alpha, sigma_beta, sigma_tau, T);
% 
% % Construct transfer matrix and transform Vy to Ux
% h = tdtf(tfun, pLSQ(:,k), N, T);
% hinv = eye(N)/h;
% Ux = hinv*Vy*transpose(hinv);
% 
% % Compute precision matrices
% Vxi = eye(N)/Vx;
% Vyi = eye(N)/Vy;
% Uxi = eye(N)/Ux;
% 
% % Compute basic residuals and plot
% rx = yn1(:,k) - muLSQ(:,k);
% ry = yn2(:,k) - h*muLSQ(:,k);
% 
% figure('Name','Basic residuals')
% plot(t, [rx ry])
% xlabel('Time (ps)')
% ylabel('Residual')
% legend('x - \mu', 'y - \psi')
% 
% % Normalize as in cost function
% rxnorm1 = sqrtm(Vxi)*rx;
% rynorm1 = sqrtm(Vyi)*ry;
% 
% figure('Name','Normed residuals, cost function')
% plot(t, [rxnorm1 rynorm1])
% xlabel('Time (ps)')
% ylabel('Normalized residual')
% legend('V_x^{-1/2}(x - \mu)', 'V_y^{-1/2}(y - \psi)')
% 
% % Normalize as in alternative cost function
% rxnorm2 = sqrtm(Vxi)*rx;
% rynorm2 = sqrtm(Uxi)*hinv*ry;
% 
% figure('Name','Normed residuals, alternative cost function')
% plot(t, [rxnorm2 rynorm2])
% xlabel('Time (ps)')
% ylabel('Normalized residual')
% legend('V_x^{-1/2}(x - \mu)', 'U_x^{-1/2}(h^{-1}y - \mu)')
% 
% % Normalize according to expected covariance
% VUi = eye(N)/(Vxi + Uxi);
% W = (eye(N) + Uxi*Vx)/(eye(N) + Vxi*Ux);
% 
% Vrx = VUi*W;
% Vry = VUi/W;
% 
% rxnorm3 = sqrtm(eye(N)/Vrx)*rx;
% rynorm3 = sqrtm(eye(N)/Vry)*hinv*ry;
% 
% figure('Name','Normed residuals, covariance')
% plot(t, [rxnorm3 rynorm3])
% xlabel('Time (ps)')
% ylabel('Normalized residual')
% legend('V_{rx}^{-1/2}(x - \mu)', 'V_{ry}^{-1/2}(h^{-1}y - \mu)')

%%
% Stop timer
toc