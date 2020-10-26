%% Test HMC integration over nuisance parameters to obtain noise parameters

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
epsilon_alpha = sigma_alpha*randn(N,M);
epsilon_beta = sigma_beta*randn(N,M);
epsilon_tau = sigma_tau*randn(N,M);

sigma_A = 0.05;
sigma_eta = 0.1*T;

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
LogVAlphaPriorSigma = 5;
LogVBetaPriorMean = -15;
LogVBetaPriorSigma = 5;
LogVTauPriorMean = -15;
LogVTauPriorSigma = 5;
MuPriorMean = mean(xn, 2);
MuPriorSigma = std(xn, 0, 2);
LogAPriorMean = zeros(M-1,1);
LogAPriorSigma = 0.5*ones(M-1,1);
EtaPriorMean = zeros(M-1,1);
EtaPriorSigma = 0.5*T*ones(M-1,1);

%% Define log posterior
logpdf = @(P) tdlogposterior(P, xn, T, ...
    LogVAlphaPriorMean, LogVAlphaPriorSigma, ...
    LogVBetaPriorMean, LogVBetaPriorSigma, ...
    LogVTauPriorMean, LogVTauPriorSigma, ...
    MuPriorMean, MuPriorSigma, ...
    LogAPriorMean, LogAPriorSigma, ...
    EtaPriorMean, EtaPriorSigma);

%% Create HMC sampler
logSigmaEst = log(mean(std(xn, 0, 2)));
muEst = MuPriorMean;
% startpoint = [logSigmaEst*ones(3,1); muEst; zeros(2*(M-1),1)];
startpoint = [2*log([sigma_alpha; sigma_beta; sigma_tau]); ...
    xfun(t, tc, w); zeros(2*M-2,1)];
smp = hmcSampler(logpdf, startpoint, ...
    'StepSize', 1e-2, ...
    'NumSteps', 50, ...
    'CheckGradient', false);

%% Estimate MAP point

[MAPpars,fitInfo] = estimateMAP(smp,...
    'VerbosityLevel', 0, ...
    'IterationLimit', 100000);
MAPlogVarAlpha = MAPpars(1);
MAPlogVarBeta = MAPpars(2);
MAPlogVarTau = MAPpars(3);
MAPmu = MAPpars(3 + (1:N));
MAPlogA = MAPpars(3 + N + (1:M-1));
MAPeta = MAPpars(3 + N + M - 1 + (1:M-1));

plot(fitInfo.Iteration,fitInfo.Objective,'ro-');
xlabel('Iteration');
ylabel('Negative log density');

%% Tune sampler

[smp,tuneinfo] = tuneSampler(smp, 'Start', MAPpars,...
    'MassVectorTuningMethod','hessian',...
    'VerbosityLevel',1,'NumPrint',10,...
    'NumStepSizeTuningIterations',100);

figure;
plot(tuneinfo.StepSizeTuningInfo.StepSizeProfile);
xlabel('Iteration');
ylabel('Step size');

fprintf("accratio = %.4f\n", tuneinfo.StepSizeTuningInfo.AcceptanceRatio)

%% Draw samples

NumChains = 4;
chains = cell(NumChains,1);
Burnin = 2000;
NumSamples = 1000;
for c = 1:NumChains
    if (c == 1)
        level = 1;
    else
        level = 0;
    end
    chains{c} = drawSamples(smp,'Start',MAPpars + randn(size(MAPpars)), ...
        'Burnin',Burnin,'NumSamples',NumSamples, ...
        'VerbosityLevel',level,'NumPrint',100);
end

%% Examine convergence diagnostics

% diags = diagnostics(smp,chains);
% disp(diags)
% disp(2*log([sigma_alpha, sigma_beta, sigma_tau]))

%% Visualize samples

figure;
plot(chains{1}(:,1))
title("log(\sigma_{\alpha}^2), Chain 1")

figure;
plot(chains{1}(:,2))
title("log(\sigma_{\beta}^2), Chain 1")

figure;
plot(chains{1}(:,3))
title("log(\sigma_{\tau}^2), Chain 1")

