function [logpdf, gradlogpdf] = tdlogposterior(Parameters, X, T, ...
    LogVAlphaPriorMean, LogVAlphaPriorSigma, ...
    LogVBetaPriorMean, LogVBetaPriorSigma, ...
    LogVTauPriorMean, LogVTauPriorSigma, ...
    MuPriorMean, MuPriorSigma, ...
    LogAPriorMean, LogAPriorSigma, ...
    EtaPriorMean, EtaPriorSigma)

% Unpack parameter vector
[N, M] = size(X);
LogVAlpha   = Parameters(1);
LogVBeta    = Parameters(2);
LogVTau     = Parameters(3);
Mu          = Parameters(3 + (1:N));
LogA        = Parameters(3 + N + (1:M-1));
Eta         = Parameters(3 + N + M - 1 + (1:M-1));

% Repack parameter vector
P = [LogVAlpha; LogVBeta; LogVTau; Mu; 0; LogA; 0; Eta];

% Compute log likelihood and gradient
[logLik, gradLL] = tdloglikelihood(P, X, T);

% Compute log prior and gradient
[LPLogVAlpha, gradLPLogVAlpha] = normalPrior(LogVAlpha, ...
    LogVAlphaPriorMean, LogVAlphaPriorSigma);
[LPLogVBeta, gradLPLogVBeta] = normalPrior(LogVBeta, ...
    LogVBetaPriorMean, LogVBetaPriorSigma);
[LPLogVTau, gradLPLogVTau] = normalPrior(LogVTau, ...
    LogVTauPriorMean, LogVTauPriorSigma);
[LPMu, gradLPMu] = normalPrior(Mu, MuPriorMean, MuPriorSigma);
[LPLogA, gradLPLogA] = normalPrior(LogA, LogAPriorMean, LogAPriorSigma);
[LPEta, gradLPEta] = normalPrior(Eta, EtaPriorMean, EtaPriorSigma);

logPrior = LPLogVAlpha + LPLogVBeta + LPLogVTau + LPMu + LPLogA + LPEta;
gradPrior = [gradLPLogVAlpha; gradLPLogVBeta; gradLPLogVTau; ...
    gradLPMu; gradLPLogA; gradLPEta];

% Return the log posterior and its gradient
logpdf = logLik + logPrior;
gradlogpdf = [gradLL(1:3); gradLL(3 + (1:N)); gradLL(3 + N + (2:M)); ...
    gradLL(3 + N + M + (2:M))] + gradPrior;

end

function [lognormpdf,gradlognormpdf] = normalPrior(P,Mu,Sigma)
Z          = (P - Mu)./Sigma;
lognormpdf     = sum(-log(Sigma) - .5*log(2*pi) - .5*(Z.^2));
gradlognormpdf = -Z./Sigma;
end
