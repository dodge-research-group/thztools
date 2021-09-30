function [loglik,gradll] = tdloglikelihood(Param, x, ts)
%TDLOGLIKELIHOOD computes log-likelihood for the time-domain noise model
%
% Syntax:   loglik = tdloglikelihood(Param, x, ts)
%
% Description:
%
% TDLOGLIKELIHOOD computes the log-likelihood function for obtaining the
% data matrix x, given the parameter vector Param.
%  
% Inputs:
%   x       Data matrix                 [NxM double]
%   Param   Parameter vector, including:
%       logv   Log of noise parameters	[3x1 double]
%       mu     Signal vector            [Nx1 double]
%       logA   Log of amplitude vector  [Mx1 double]
%       eta    Delay vector             [Mx1 double]
%   ts      Sampling time               [1x1 double]
%
% Outputs: 
%   loglik     log-likelihood function
%   gradll     Gradient of the log-likelihood function
%

% Parse function inputs
[N,M] = size(x);
validateattributes(x, {'double'}, {'2d'})
validateattributes(Param, {'double'}, {'vector'})
validateattributes(ts, {'double'}, {'scalar'})

Param = Param(:);
logv = Param(1:3);
v = exp(logv);
mu = Param(3 + (1:N));
logA = Param(3 + N + (1:M));
A = exp(logA);
eta = Param(3 + N + M + (1:M));

% Compute frequency vector and Fourier coefficients of mu
f =  fftfreq(N,ts);
w = 2*pi*f;
mu_f = fft(mu);

% Compute zeta
exp_iweta = exp(1i*w(:,ones(1,M)).*eta(:,ones(1,N))');
zeta_f = A(:,ones(1,N))'.*conj(exp_iweta).*mu_f(:,ones(1,M));
zeta = real(ifft(zeta_f));

% Compute log-likelihood and gradient

% Compute residuals and their squares for subsequent computations
res = x - zeta;
ressq = res.^2;


Dzeta = real(ifft(1i*w(:,ones(1,M)).*zeta_f));

valpha = v(1);
vbeta = v(2)*zeta.^2;
vtau = v(3)*(Dzeta).^2;
vtot = valpha + vbeta + vtau;

resnormsq = ressq./vtot;
loglik = -M*N*log(2*pi)/2 - sum(log(vtot(:)))/2 - sum(resnormsq(:))/2;

% Compute gradient if requested
if nargout > 1
    Ngrad = sum([3;N;M;M]);
    gradll = zeros(Ngrad,1);

    reswt = res./vtot;
    dvar = (vtot - ressq)./vtot.^2;
    
    % Gradient wrt logv
    gradll(1) = -(1/2)*sum(dvar(:))*v(1);
    gradll(2) = -(1/2)*sum(zeta(:).^2.*dvar(:))*v(2);
    gradll(3) = -(1/2)*sum(Dzeta(:).^2.*dvar(:))*v(3);
    
    % Gradient wrt mu
    P = fft(v(2)*dvar.*zeta - reswt) ...
        -1i*v(3)*w.*fft(dvar.*Dzeta);
    gradll(3 + (1:N)) = ...
        -sum(A'.*real(ifft(exp_iweta.*P)),2);
    
    % Gradient wrt logA
    term = ((vtot - valpha).*dvar - reswt.*zeta);
    gradll(3 + N + (1:M)) = -sum(term,1)';
    
    % Gradient wrt eta
    DDzeta = ...
        real(ifft(-w(:,ones(1,M)).^2.*zeta_f));
    gradll(3 + N + M + (1:M)) = ...
        sum(dvar.*(zeta.*Dzeta*v(2) + Dzeta.*DDzeta*v(3)) ...
        - reswt.*Dzeta);
end


end