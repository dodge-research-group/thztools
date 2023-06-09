%% costfunlsq
fun = @(theta, wfft) theta(1) * exp(1i * theta(2) * wfft);
theta = [1 2];
N = 100;
ampSigma = 1e-5;

xx = [linspace(0, 10, N)'];
yy = [thzgen(N, 1, 1)]; + ampSigma*rand(N, 1);
y1 = thzgen(N, 1, 1);


wfft = 2*pi*[fftfreq(N, 1)];
sigmax = ampSigma*rand(N, 1);
sigmay = ampSigma*rand(N, 1);

outcostunflsq = costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft);

% ============================================
%% tdtf
theta = [3.0 4.0];
N = 5;
ts = 0.2;
fun = @(theta, w) theta(1)*exp(-1i*theta(2)*w);

out_tdtf = tdtf(fun, theta, N, ts);
%=============================================

%% shiftmtx
tau = 1.0;
n= 256;
ts = 1.0;

out_shiftmtx = shiftmtx(tau, n, ts);
%===========================================

%% tdnll
N = 10;
M = 8;
xm = ones(N, M);

Param =[struct()];
varargin = [struct()];
for i = 1:1:1
    Param(i).logv = ones(3,1);
    Param(i).mu = ones(N,1);
    Param(i).A = ones(M,1);
    Param(i).eta = ones(M,1);
    Param(i).ts = 0.5;
    Param(i).D = tdtf(@(theta,w) -1i*w, 0, N, Param(i).ts);
    
    % Set optional inputs
    varargin(i).logv = 0;
    varargin(i).mu = 0;
    varargin(i).A = 0;
    varargin(i).eta = 0;


end

[nll, gradnll] = tdnll(xm, Param, varargin)

