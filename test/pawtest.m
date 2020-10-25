%% Test integration over nuisance parameters
% Based on Example 10.1 in Pawitan, _In All Likelihood_ , which in turn is
% based on Neyman & Scott (1948), |doi:10.2307/1914288|.
%
% Consider a set of samples $x_{i1}$ and $x_{i2}$ from $N(\mu_i,
% \sigma^2)$, $i = 1, 2, \ldots, N$, where the parameter of interest is
% $\sigma^2$. The maximum likelihood estimate is inconsistent, with
%
% $$
% \lim_{N\rightarrow\infty}\widehat{\sigma^2} = \sigma^2/2.
% $$
%
% Here, we use a Hamiltonian Monte Carlo sampler to integrate out the
% nuisance parameter vector $\mathbf{\mu}$ and obtain an accurate estimate
% of $\sigma^2$.

%% Preliminaries
% Read in data
Dat = csvread('pawdata.txt');
mu = Dat(:,2);
X = Dat(:,3:4);
xbar = Dat(:,5);
trueSigma = 1;

%% Define prior probability density
% We choose broad normal distributions as relatively uninformative priors
% for both $\log(\sigma^2)$ and $\mathbf{\mu}$.
LogVariancePriorMean = 0;
LogVariancePriorSigma = 2;
MeanPriorMean = 0;
MeanPriorSigma = 20;

%% Define log posterior
logpdf = @(P) logPosterior(P, X,...
LogVariancePriorMean, LogVariancePriorSigma, ...
MeanPriorMean, MeanPriorSigma);

%% Create HMC sampler
logSigmaEst = log(mean(std(X, 0, 2)));
muEst = xbar;
startpoint = [logSigmaEst; muEst];
smp = hmcSampler(logpdf, startpoint, 'NumSteps',50);

%% Estimate MAP point

[MAPpars,fitInfo] = estimateMAP(smp,'VerbosityLevel',0);
MAPlogSigma = MAPpars(1);
MAPmu = MAPpars(2:end);

plot(fitInfo.Iteration,fitInfo.Objective,'ro-');
xlabel('Iteration');
ylabel('Negative log density');

%% Tune sampler

[smp,tuneinfo] = tuneSampler(smp,'Start',MAPpars);

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
for c = 1:NumChains
    if (c == 1)
        level = 1;
    else
        level = 0;
    end
    chains{c} = drawSamples(smp,'Start',MAPpars + randn(size(MAPpars)), ...
        'Burnin',Burnin,'NumSamples',NumSamples, ...
        'VerbosityLevel',level,'NumPrint',300);
end

%% Examine convergence diagnostics

diags = diagnostics(smp,chains);
truePars = [log(trueSigma^2);mu];
disp(diags)
disp(truePars)

%% Visualize samples

figure;
plot(chains{1}(:,1))
title("log(\sigma^2), Chain 1")

figure;
plot(chains{1}(:,2))
title("\mu_1, Chain 1")

concatenatedSamples = vertcat(chains{:});
figure;
histogram(exp(concatenatedSamples(:,1)))
xline(trueSigma^2,'r-','LineWidth',1)
xline(mean(exp(concatenatedSamples(:,1))),'k-','LineWidth',1)
xlabel("\sigma^2")
ylabel("Frequency")

Fig = figure;
Fig.Position = Fig.Position.*[0.5 0.5 1.5 1.5];
N = length(mu);
ax = gobjects(N,1);
for i = 1:N
    ax(i) = subplot(4,5,i);
    histogram(concatenatedSamples(:,i+1))
    xline(mu(i),'r-','LineWidth',1)
    xline(mean(concatenatedSamples(:,i+1)),'k-','LineWidth',1)
    xlim([-10 10]);
    ax(i).XTickLabel = {};
    ax(i).YTickLabel = {};
end