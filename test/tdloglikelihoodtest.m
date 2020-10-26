%% Test TDLOGLIKELIHOOD function
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
P = zeros(3+N+2*M,1);
P(1:3) = log([Noise.add;Noise.mult;Noise.time].^2);
P(3 + (1:N)) = xfun(t,tc,w);
P(3 + N + (1:M)) = log(1 + 0.1*randn(M,1));
P(3 + N + M + (1:M)) = 0.1*T*randn(M,1);

[loglik,gradll] = tdloglikelihood(P, xn, T);

%% Compute gradient numerically
delta = 1e-6;
gradnum = zeros(size(gradll));
for ii=1:3
    Pp = P;
    Pp(ii) = P(ii) + delta;
    Pm = P;
    Pm(ii) = P(ii) - delta;
    gradnum(ii) = (tdloglikelihood(Pp, xn, T) ...
        - tdloglikelihood(Pm, xn, T))/(2*delta);
end
for ii=1:N
    Pp = P;
    Pp(3 + ii) = P(3 + ii) + delta;
    Pm = P;
    Pm(3 + ii) = P(3 + ii) - delta;
    gradnum(3 + ii) = (tdloglikelihood(Pp, xn, T) ...
        - tdloglikelihood(Pm, xn, T))/(2*delta);
end
for m=1:M
    Pp = P;
    Pp(3 + N + m) = P(3 + N + m) + delta;
    Pm = P;
    Pm(3 + N + m) = P(3 + N + m) - delta;
    gradnum(3 + N + m) = (tdloglikelihood(Pp, xn, T) ...
        - tdloglikelihood(Pm, xn, T))/(2*delta);
end
for m=1:M
    Pp = P;
    Pp(3 + N + M + m) = P(3 + N + M + m) + delta;
    Pm = P;
    Pm(3 + N + M + m) = P(3 + N + M + m) - delta;
    gradnum(3 + N + M + m) = (tdloglikelihood(Pp, xn, T) ...
        - tdloglikelihood(Pm, xn, T))/(2*delta);
end

disp('Value of loglik:')
disp(loglik)

vRange = 1:3;
disp('Gradient with respect to variance parameters:')
disp([gradll(vRange), gradnum(vRange)])

disp('Norm of differences wrt logv:')
disp(norm(gradll(vRange)-gradnum(vRange))/...
    norm(gradll(vRange)+gradnum(vRange)))

muRange = 3 + (1:N);
plot(t, gradll(muRange), '-', t, gradnum(muRange), '--')
xlabel('t (ps)')
ylabel('\nabla_\muC')
title('Gradient with respect to signal vector')

disp('Norm of differences wrt mu:')
disp(norm(gradll(muRange) - gradnum(muRange))/...
    norm(gradll(muRange) + gradnum(muRange)))

ARange = 3 + N + (1:M);
disp('Gradient with respect to amplitudes:')
disp([gradll(ARange), gradnum(ARange)])

disp('Norm of differences wrt logA:')
disp(norm(gradll(ARange) - gradnum(ARange))/...
    norm(gradll(ARange) + gradnum(ARange)))

etaRange = 3 + N + M + (1:M);
disp('Gradient with respect to delays:')
disp([gradll(etaRange), gradnum(etaRange)])

disp('Norm of differences wrt eta:')
disp(norm(gradll(etaRange) - gradnum(etaRange))/...
    norm(gradll(etaRange) + gradnum(etaRange)))
