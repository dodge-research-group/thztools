function [nll,gradnll] = tdnll(x,Param,varargin)
%TDNLL computes negative log-likelihood for the time-domain noise model
%
% Syntax:   nll = tdnll(x,Param,Fix)
%
% Description:
%
% TDNLL computes the negative log-likelihood function for obtaining the
% data matrix x, given the parameter structure Param.
%  
% Inputs:
%   x       Data matrix
%   Param   Parameter structure, including:
%       .logv   Log of noise parameters	[3x1 double]
%       .mu     Signal vector           [Nx1 double]
%       .A      Amplitude vector        [Mx1 double]
%       .eta    Delay vector            [Mx1 double]
%       .ts     Sampling time           [double]
%       .D      Derivative matrix       [NxN double]
%   Fix     Variables to fix for gradient calculation
%       .logv   Log of noise parameters [logical]
%       .mu     Signal vector           [logical]
%       .A      Amplitude vector        [logical]
%       .eta    Delay vector            [logical]
%
% Outputs: 
%   nll     Negative log-likelihood function
%   gradnll Gradient of the negative log-likelihood function
%

% Parse function inputs
[N,M] = size(x);
validateattributes(x,{'double'},{'2d'})
validateattributes(Param,{'struct'},{'nonempty'})
if nargin > 2
    Fix = varargin{1};
    validateattributes(Fix,{'struct'},{'nonempty'})
else
    Fix = struct('logv',false,'mu',false,'A',false,'eta',false);
end

% Parse parameter structure
Pfields = fieldnames(Param);
if ismember('logv',Pfields)
    v = exp(Param.logv(:));
    validateattributes(v,{'double'},{'vector','numel',3})
else
    error('TDNLL requires Param structure with logv field')
end
if ismember('mu',Pfields)
    mu = Param.mu(:);
    validateattributes(mu,{'double'},{'vector','numel',N})
else
    error('TDNLL requires Param structure with mu field')
end
if ismember('A',Pfields) && ~isempty(Param.A)
    A = Param.A(:);
    validateattributes(A,{'double'},{'vector','numel',M})
    Ignore.A = false;
else
    A = ones(M,1);
    Ignore.A = true;
end
if ismember('eta',Pfields) && ~isempty(Param.eta)
    eta = Param.eta(:);
    validateattributes(eta,{'double'},{'vector','numel',M})
    Ignore.eta = false;
else
    eta = zeros(M,1);
    Ignore.eta = true;
end
if ismember('ts',Pfields)
    ts = Param.ts;
    validateattributes(ts,{'double'},{'scalar'})
else
    ts = 1;
    warning('TDNLL received Param structure without ts field; set to one')
end
if ismember('D',Pfields)
    D = Param.D;
    validateattributes(D,{'double'},{'size',[N N]})
else
    % Compute derivative matrix
    fun = @(theta,w) -1i*w;
    D = tdtf(fun,0,N,ts);
end

% Compute frequency vector and Fourier coefficients of mu
f =  fftfreq(N,ts);
w = 2*pi*f;
mu_f = fft(mu);

gradcalc = ~[Fix.logv;...
    Fix.mu;...
    (Fix.A || Ignore.A) ;...
    (Fix.eta || Ignore.eta)];

if Ignore.eta
    zeta = mu*A';
    zeta_f = fft(zeta);
else
    exp_iweta = exp(1i*w(:,ones(1,M)).*eta(:,ones(1,N))');
    zeta_f = A(:,ones(1,N))'.*conj(exp_iweta).*mu_f(:,ones(1,M));
    zeta = real(ifft(zeta_f));
end

% Compute negative-log likelihood and gradient

% Compute residuals and their squares for subsequent computations
res = x - zeta;
ressq = res.^2;

% Simplest case: just variance and signal parameters, A and eta fixed at
% defaults
if Ignore.A && Ignore.eta
    
    Dmu = real(ifft(1i*w.*mu_f));
    valpha = v(1);
    vbeta = v(2)*mu.^2;
    vtau = v(3)*Dmu.^2;
    vtot = valpha + vbeta + vtau;
    
    resnormsq = ressq./vtot(:,ones(1,M));
    nll = M*N*log(2*pi)/2 + (M/2)*sum(log(vtot)) + sum(resnormsq(:))/2;
    
    % Compute gradient if requested
    if nargout > 1
        Ngrad = sum(gradcalc(1:2).*[3;N]);
        gradnll = zeros(Ngrad,1);
        nStart = 1;
        dvar = (vtot - mean(ressq,2))./vtot.^2;
        if gradcalc(1)
            gradnll(nStart) = (M/2)*sum(dvar)*v(1);
            gradnll(nStart+1) = (M/2)*sum(mu.^2.*dvar)*v(2);
            gradnll(nStart+2) = (M/2)*sum(Dmu.^2.*dvar)*v(3);
            nStart = nStart + 3;
        end
        if gradcalc(2)
            gradnll(nStart+(0:N-1)) = M*(v(2)*mu.*dvar ...
                + v(3)*transpose(D)*(Dmu.*dvar) ...
                - mean(res,2)./vtot);
        end
    end
    
% Alternative case: A, eta, or both are not set to defaults
else
    
    Dzeta = real(ifft(1i*w(:,ones(1,M)).*zeta_f));
 
    valpha = v(1);
    vbeta = v(2)*zeta.^2;
    vtau = v(3)*(Dzeta).^2;
    vtot = valpha + vbeta + vtau;
    
    resnormsq = ressq./vtot;
    nll = M*N*log(2*pi)/2 + sum(log(vtot(:)))/2 + sum(resnormsq(:))/2;
    
    % Compute gradient if requested
    if nargout > 1
        Ngrad = sum(gradcalc.*[3;N;M;M]);
        gradnll = zeros(Ngrad,1);
        nStart = 1;
        reswt = res./vtot;
        dvar = (vtot - ressq)./vtot.^2;
        if gradcalc(1)
            % Gradient wrt logv
            gradnll(nStart) = (1/2)*sum(dvar(:))*v(1);
            gradnll(nStart+1) = (1/2)*sum(zeta(:).^2.*dvar(:))*v(2);
            gradnll(nStart+2) = (1/2)*sum(Dzeta(:).^2.*dvar(:))*v(3);
            nStart = nStart + 3;
        end
        if gradcalc(2)
            % Gradient wrt mu
            P = fft(v(2)*dvar.*zeta - reswt) ...
                -1i*v(3)*w.*fft(dvar.*Dzeta);
            gradnll(nStart:nStart+N-1) = ...
                sum(A'.*real(ifft(exp_iweta.*P)),2);
            nStart = nStart + N;
        end
        if gradcalc(3)
            % Gradient wrt A
            term = ((vtot - valpha).*dvar - reswt.*zeta);
            gradnll(nStart + (0:M-1)) = sum(term,1)'./A;
            if ~Fix.mu
                gradnll(nStart) = [];
                nStart = nStart + M - 1;
            else
                nStart = nStart + M;
            end
        end
        if gradcalc(4)
            % Gradient wrt eta
            DDzeta = ...
                real(ifft(-w(:,ones(1,M)).^2.*zeta_f));
            gradnll(nStart + (0:M-1)) = ...
                -sum(dvar.*(zeta.*Dzeta*v(2) + Dzeta.*DDzeta*v(3))...
                - reswt.*Dzeta);
            if ~Fix.mu
                gradnll(nStart) = [];
            end
        end
    end
end

end