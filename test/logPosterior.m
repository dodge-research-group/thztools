function [logpdf, gradlogpdf] = logPosterior(Parameters,X, ...
    LogVariancePriorMean,LogVariancePriorSigma, ...
    MeanPriorMean,MeanPriorSigma)

%% Preliminaries
% Unpack parameter vector
LogVariance     = Parameters(1);
Mu              = Parameters(2:end);

% Compute constants
log2pi = log(2*pi);

%% Compute log likelihood and gradient
% Auxiliary variables
Sigma                   = exp(LogVariance/2);
Z                       = (X - Mu)/Sigma;
% log likelihood
logLik                  = sum(-log2pi/2 -log(Sigma) - Z(:).^2/2);
% gradient
gradLogVariance1        = -sum(1 - Z(:).^2)/2;
gradMu1                 = sum(Z/Sigma,2);

%% Compute log prior and gradient
% Auxiliary variables
ZLogVar     = (LogVariance - LogVariancePriorMean)/LogVariancePriorSigma;
ZMu         = (Mu - MeanPriorMean)/MeanPriorSigma;
% log prior
LPLogVar    = -(log2pi + 2*log(LogVariancePriorSigma) + ZLogVar^2)/2;
LPMu        = -sum(log2pi + 2*log(MeanPriorSigma) + ZMu.^2)/2;
logPrior    = LPLogVar + LPMu;
% gradient
gradLogVariance2 = -ZLogVar/LogVariancePriorSigma;
gradMu2 = -ZMu/MeanPriorSigma;

%% Return the log posterior and its gradient
logpdf = logLik + logPrior;
gradlogpdf = [gradLogVariance1 + gradLogVariance2;gradMu1 + gradMu2];

end
