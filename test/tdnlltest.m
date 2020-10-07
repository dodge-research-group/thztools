%% Test TDNLL function
%

%% Set initial parameters

T=.05;          % sampling time [ps]
N=256;          % number of sampled points
M=10;            % number of traces to compare
SNR=2e3;        % signal to noise ratio
w=0.2;          % pulse width [ps]
tc=N*T/3;       % pulse center [ps]

% Generate ideal time-domain pulse and pulse derivative
t=T*(0:N-1);
t=t(:);

xfun = @(t,t0,w) (1-2*((t-t0)/w).^2).*exp(-((t-t0)/w).^2);

%% Generate noisy pulse set
Noise.add = 1/SNR;
Noise.mult = 10/SNR;
Noise.time = 10*T/SNR;
rng('default')

x = zeros(N,M);
eps_alpha = randn(N,M);
eps_beta = randn(N,M);
eps_eta = randn(N,M);
for mm = 1:M
    x(:,mm)=xfun(t+Noise.time*eps_eta(:,mm),tc,w);
end
xn = x + Noise.add*eps_alpha + Noise.mult*x.*eps_beta;

%% Compute negative-log likelihood cost function and its gradient
P.logv = log([Noise.add;Noise.mult;Noise.time].^2);
P.mu = xfun(t,tc,w);
P.A = 1 + 0.1*randn(M,1);
P.eta = 0.1*T*randn(M,1);
P.ts = T;

Fix.logv = false;
Fix.mu = false;
Fix.A = false;
Fix.eta = false;

[nll,gradnll] = tdnll(xn,P,Fix);

%% Compute gradient numerically
delta = 1e-6;
gradnum = zeros(size(gradnll));
for ii=1:3
    Pp = P;
    Pp.logv(ii) = P.logv(ii) + delta;
    Pm = P;
    Pm.logv(ii) = P.logv(ii) - delta;
    gradnum(ii) = (tdnll(xn,Pp)-tdnll(xn,Pm))/(2*delta);
end
for ii=1:N
    Pp = P;
    Pp.mu(ii) = P.mu(ii) + delta;
    Pm = P;
    Pm.mu(ii) = P.mu(ii) - delta;
    gradnum(3+ii) = (tdnll(xn,Pp)-tdnll(xn,Pm))/(2*delta);
end
for m=1:M
    Pp = P;
    Pp.A(m) = P.A(m) + delta;
    Pm = P;
    Pm.A(m) = P.A(m) - delta;
    gradnum(3+N+m) = (tdnll(xn,Pp)-tdnll(xn,Pm))/(2*delta);
end
for m=1:M
    Pp = P;
    Pp.eta(m) = P.eta(m) + delta;
    Pm = P;
    Pm.eta(m) = P.eta(m) - delta;
    gradnum(3+N+M+m) = (tdnll(xn,Pp)-tdnll(xn,Pm))/(2*delta);
end

disp('Value of nll:')
disp(nll)

vRange = 1:3;
disp('Gradient with respect to variance parameters:')
disp([gradnll(vRange), gradnum(vRange)])

disp('Norm of differences wrt logv:')
disp(norm(gradnll(vRange)-gradnum(vRange))/...
    norm(gradnll(vRange)+gradnum(vRange)))

muRange = 3+(1:N);
plot(t,gradnll(muRange),'-',t,gradnum(muRange),'--')
xlabel('t (ps)')
ylabel('\nabla_\muC')
title('Gradient with respect to signal vector')

disp('Norm of differences wrt mu:')
disp(norm(gradnll(muRange)-gradnum(muRange))/...
    norm(gradnll(muRange)+gradnum(muRange)))

ARange = 3+N+(1:M);
disp('Gradient with respect to amplitudes:')
disp([gradnll(ARange), gradnum(ARange)])

disp('Norm of differences wrt A:')
disp(norm(gradnll(ARange)-gradnum(ARange))/...
    norm(gradnll(ARange)+gradnum(ARange)))

etaRange = 3+N+M+(1:M);
disp('Gradient with respect to delays:')
disp([gradnll(etaRange), gradnum(etaRange)])

disp('Norm of differences wrt eta:')
disp(norm(gradnll(etaRange)-gradnum(etaRange))/...
    norm(gradnll(etaRange)+gradnum(etaRange)))
