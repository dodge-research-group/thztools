%% Simulate marginalized HMC distributions for variance parameters

%% Start timer
tStart = tic;

%% Set initial parameters

T=.05;          % sampling time [ps]
N=256;          % number of sampled points
M=10;           % number of traces to compare
SNR=2e3;        % signal to noise ratio
w=0.2;          % pulse width [ps]
tc=N*T/3;       % pulse center [ps]
nMC = pow2(10); % number of Monte Carlo runs

% Generate ideal time-domain pulse and pulse derivative
t=T*(0:N-1);
t=t(:);

xfun = @(t,t0,w) (1-2*((t-t0)/w).^2).*exp(-((t-t0)/w).^2);

%% Simulate data

Noise.add = 1/SNR;
Noise.mult = 10/SNR;
Noise.time = 10*T/SNR;

sigma_alpha = Noise.add;
sigma_beta = Noise.mult;
sigma_tau = Noise.time;

sigma_A = 0.05;
sigma_eta = 0.1*T;

true_sigma = [sigma_alpha, sigma_beta, sigma_tau];
true_mu = xfun(t, tc, w);

rng('default')
mean_exp_logv = zeros(nMC,3);
exp_mean_logv = zeros(nMC,3);

parfor iMC = 1:nMC
    % Simulate data
    epsilon_alpha = sigma_alpha*randn(N,M);
    epsilon_beta = sigma_beta*randn(N,M);
    epsilon_tau = sigma_tau*randn(N,M);
    
    x = zeros(N,M);
    A = 1 + [0; sigma_A*randn(M-1,1)];
    eta = [0; sigma_eta*randn(M-1,1)];
    for jj = 1:M
        x(:,jj) = A(jj)...
            *xfun(t, tc + epsilon_tau(:,jj) + eta(jj), w);
    end
    xn = x + epsilon_alpha + epsilon_beta.*abs(x);
    
    %% Define prior distribution
    LogVAlphaPriorMean = -15;
    LogVAlphaPriorSigma = 15;
    LogVBetaPriorMean = -15;
    LogVBetaPriorSigma = 15;
    LogVTauPriorMean = -15;
    LogVTauPriorSigma = 15;
    MuPriorMean = mean(xn, 2);
    MuPriorSigma = 10*std(xn, 0, 2);
    LogAPriorMean = zeros(M-1,1);
    LogAPriorSigma = 5*ones(M-1,1);
    EtaPriorMean = zeros(M-1,1);
    EtaPriorSigma = 5*T*ones(M-1,1);
    
    % Define log posterior
    logpdf = @(P) tdlogposterior(P, xn, T, ...
        LogVAlphaPriorMean, LogVAlphaPriorSigma, ...
        LogVBetaPriorMean, LogVBetaPriorSigma, ...
        LogVTauPriorMean, LogVTauPriorSigma, ...
        MuPriorMean, MuPriorSigma, ...
        LogAPriorMean, LogAPriorSigma, ...
        EtaPriorMean, EtaPriorSigma);
    
    % Create HMC sampler
    startpoint = [2*log([sigma_alpha; sigma_beta; sigma_tau]); ...
        xfun(t, tc, w); log(A(2:end)); eta(2:end)];
    smp = hmcSampler(logpdf, startpoint, 'CheckGradient', false);
    
    % Estimate MAP point
    [MAPpars,fitInfo] = estimateMAP(smp,...
        'VerbosityLevel', 0, ...
        'IterationLimit', 200000);
        
    % Tune sampler
    [smp,tuneinfo] = tuneSampler(smp, ...
        'Start', MAPpars, ...
        'MassVectorTuningMethod', 'hessian', ...
        'VerbosityLevel', 0, ...
        'NumPrint', 10);
    
    % Draw samples
    NumChains = 4;
    chains = cell(NumChains,1);
    Burnin = 500;
    NumSamples = 1000;
    ThinSize = 1;
    level = 0;
    for c = 1:NumChains
        chains{c} = drawSamples(smp, ...
            'Start', MAPpars, ...
            'Burnin', Burnin, ...
            'NumSamples', NumSamples, ...
            'ThinSize', ThinSize, ...
            'VerbosityLevel', level, ...
            'NumPrint', 100);
    end
    
    concatenatedSamples = vertcat(chains{:});
    mean_exp_logv(iMC,:) = mean(exp(concatenatedSamples(:,1:3)));
    exp_mean_logv(iMC,:) = exp(mean(concatenatedSamples(:,1:3)));
    
end

%% Stop timer
toc(tStart)

%% Save results
strNow = char(datetime('now','Format','yyyy-MM-dd''T''HHmmss'));
save(['tdhmcnoisetestmc-' strNow])
disp(strNow)