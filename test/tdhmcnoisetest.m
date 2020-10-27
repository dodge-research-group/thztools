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

rng('default')
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

%% Define log posterior
logpdf = @(P) tdlogposterior(P, xn, T, ...
    LogVAlphaPriorMean, LogVAlphaPriorSigma, ...
    LogVBetaPriorMean, LogVBetaPriorSigma, ...
    LogVTauPriorMean, LogVTauPriorSigma, ...
    MuPriorMean, MuPriorSigma, ...
    LogAPriorMean, LogAPriorSigma, ...
    EtaPriorMean, EtaPriorSigma);

%% Create HMC sampler

startpoint = [2*log([sigma_alpha; sigma_beta; sigma_tau]); ...
    xfun(t, tc, w); log(A(2:end)); eta(2:end)];
smp = hmcSampler(logpdf, startpoint, 'CheckGradient', false);

%% Estimate MAP point

[MAPpars,fitInfo] = estimateMAP(smp,...
    'VerbosityLevel', 0, ...
    'IterationLimit', 200000);
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

[smp,tuneinfo] = tuneSampler(smp, ...
    'Start', MAPpars, ...
    'MassVectorTuningMethod', 'hessian', ...
    'VerbosityLevel', 1, ...
    'NumPrint', 10);

disp(smp)
disp('')

figure;
plot(tuneinfo.StepSizeTuningInfo.StepSizeProfile);
xlabel('Iteration');
ylabel('Step size');

fprintf("accratio = %.4f\n", tuneinfo.StepSizeTuningInfo.AcceptanceRatio)

%% Draw samples

NumChains = 4;
chains = cell(NumChains,1);
Burnin = 500;
NumSamples = 1000;
ThinSize = 1;
for c = 1:NumChains
    if (c == 1)
        level = 1;
    else
        level = 0;
    end
    chains{c} = drawSamples(smp, ...
        'Start', MAPpars, ...
        'Burnin', Burnin, ...
        'NumSamples', NumSamples, ...
        'ThinSize', ThinSize, ...
        'VerbosityLevel', level, ...
        'NumPrint', 100);
end

%% Examine convergence diagnostics

diags = diagnostics(smp,chains);
disp(diags)

concatenatedSamples = vertcat(chains{:});
true_sigma = [sigma_alpha, sigma_beta, sigma_tau];
mean_exp = sqrt(mean(exp(concatenatedSamples(:,1:3))));
exp_mean = sqrt(exp(mean(concatenatedSamples(:,1:3))));
sigmaEst = [true_sigma; mean_exp; exp_mean]';

fprintf("%-15s%15s%15s%15s\n", "Parameter", "True", "<exp(logv)>", "exp(<logv>)")
fprintf("%-15s%15.4e%15.4e%15.4e\n", "sigma_alpha", sigmaEst(1,:))
fprintf("%-15s%15.4e%15.4e%15.4e\n", "sigma_beta", sigmaEst(2,:))
fprintf("%-15s%15.4e%15.4e%15.4e\n", "sigma_tau", sigmaEst(3,:))

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

figure('Name','sigma-squared (alpha)');
histogram(exp(concatenatedSamples(:,1)))
xline(sigma_alpha^2,'r-','LineWidth',1)
xline(mean(exp(concatenatedSamples(:,1))),'k-','LineWidth',1)
xlabel("\sigma_\alpha^2")
ylabel("Frequency")

figure('Name','sigma-squared (beta)');
histogram(exp(concatenatedSamples(:,2)))
xline(sigma_beta^2,'r-','LineWidth',1)
xline(mean(exp(concatenatedSamples(:,2))),'k-','LineWidth',1)
xlabel("\sigma_\beta^2")
ylabel("Frequency")

figure('Name','sigma-squared (tau)');
histogram(exp(concatenatedSamples(:,3)))
xline(sigma_tau^2,'r-','LineWidth',1)
xline(mean(exp(concatenatedSamples(:,3))),'k-','LineWidth',1)
xlabel("\sigma_\tau^2")
ylabel("Frequency")

figure('Name','logA');
ax = gobjects(M-1,1);
for i = 1:M-1
    ax(i) = subplot(3,3,i);
    histogram(concatenatedSamples(:,3+N+i))
    xline(log(A(i+1)),'r-','LineWidth',1)
    xline(mean(concatenatedSamples(:,3+N+i)),'k-','LineWidth',1)
    xlim(log(A(i+1))+[-.01 .01]);
    if i == 8
        ax(i).XTick = log(A(i+1))+[-.01 0 .01];
        ax(i).XTickLabel = {'-0.001', '0', '0.001'};
        xlabel(ax(i),'Difference from true log(A_k)');
    else
        ax(i).XTickLabel = {};
    end
    ax(i).YTickLabel = {};
    if i==2
        title("log(A_k)")
    end
end

figure('Name','eta');
ax = gobjects(M-1,1);
for i = 1:M-1
    ax(i) = subplot(3,3,i);
    histogram(concatenatedSamples(:,3+N+M-1+i))
    xline(eta(i+1),'r-','LineWidth',1)
    xline(mean(concatenatedSamples(:,3+N+M-1+i)),'k-','LineWidth',1)
    xlim(eta(i+1)+[-.001 .001]);
    if i == 8
        ax(i).XTick = eta(i+1)+[-.001 0 .001];
        ax(i).XTickLabel = {'-0.001', '0', '0.001'};
        xlabel(ax(i),'Difference from true \eta_k');
    else
        ax(i).XTickLabel = {};
    end
    ax(i).YTickLabel = {};
    if i==2
        title("\eta_k")
    end
end