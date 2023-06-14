%% Generate test data for all functions in thztools

rng('default')

%% FFTFREQ
% Set inputs
N = [4, 5];
T = [0.1, 1];

% Define combinations
args = combinations(N, T);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.fftfreq = struct('args', Init, 'out', Init);
for i = 1:n_test
    N = args.N(i);
    T = args.T(i);
    Set.fftfreq(i).args = {int64(N), T};
    Set.fftfreq(i).out = {fftfreq(N, T)};
end

%% EPSWATER
% Set inputs
f = [1, 2];
T = [20, 30];

% Define combinations
args = combinations(f, T);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.epswater = struct('args', Init, 'out', Init);
for i = 1:n_test
    f = args.f(i);
    T = args.T(i);
    Set.epswater(i).args = {f, T};
    Set.epswater(i).out = {epswater(f, T)};
end

%% COSTFUNLSQ
% Set required inputs
N = 100;
t0 = 1;
fun = @(theta, wfft) theta(1) * exp(-1i * theta(2) * wfft);
sigma = {[1e-5, 0, 0], [1e-5, 1e-2, 0], [1e-5, 1e-2, 1e-3]};
theta = mat2cell(rand(2, 3), 2, ones(1,3));
ts = [0.1, 1];

% Define combinations
args = combinations(sigma, theta, ts);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.costfunlsq = struct('args', Init, 'out', Init);
for i = 1:n_test
    sigma = args.sigma{i};
    theta = args.theta{i};
    ts = args.ts(i);
    wfft = 2*pi*fftfreq(N, ts);
    x = thzgen(N, t0, ts);
    y = tdtf(fun, theta, N, ts)*x;
    sigmax = noiseamp(sigma, x, ts);
    sigmay = noiseamp(sigma, y, ts);
    xx = x + sigmax.*randn(N, 1);
    yy = y + sigmay.*randn(N, 1);
    Set.costfunlsq(i).args = {theta, xx, yy, sigmax, sigmay, ts};
    Set.costfunlsq(i).out = {costfunlsq(fun, theta, xx, yy, sigmax, ...
        sigmay, wfft)};
end


%% TDNLL
% Set required inputs
N = 10;
M = 8;
x = randn(N, M, 4);
x = squeeze(mat2cell(x, N, M, ones(1,4)));

% Set optional inputs
varargin.logv = 0;
varargin.mu = 0;
varargin.A = 0;
varargin.eta = 0;

Param = [struct() struct() struct()];
for i = 1:1:3
    Param(i).logv = rand(3,1);
    Param(i).mu = rand(N,1);
    Param(i).A = rand(M,1);
    Param(i).eta = rand(M,1);
    Param(i).ts = rand();
    Param(i).D = tdtf(@(theta,w) -1i*w, 0, N, Param(i).ts);
end

% Define combinations
args = combinations(x, Param);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.tdnll = struct('args', Init, 'out', Init);
for i = 1:n_test
    x = args.x{i};
    Param = args.Param(i);
    Set.tdnll(i).args = {x, Param.logv, Param.mu, Param.A, Param.eta,...
        Param.ts, Param.D};
    [Set.tdnll(i).out{1:2}] = tdnll(x, Param, varargin);
end


%% THZGEN
% Set required inputs
N = [256, 257];
T = [0.1, 1];
t0 = [2.0, 3.0];

% Define combinations
args = combinations(N, T, t0);
n_test = height(args);

% Set optional inputs
taur = 0.6;
tauc = 0.2;
taul = 0.025/sqrt(2*log(2));

% Generate output
Init = cell(n_test, 1);
Set.thzgen = struct('args', Init, 'out', Init);
for i = 1:n_test
    N = args.N(i);
    T = args.T(i);
    t0 = args.t0(i);
    Set.thzgen(i).args = {int64(N), T, t0};
    Set.thzgen(i).out = {thzgen(N, T, t0)};
end

%% NOISEVAR
% Set required inputs
sigma = {[0.5  1.0  1.5]; [6.0 7.0 8.0]; [9.0 5.0 3.0]};
mu = {[1.0 2.0 3.0 4.0]; [3.0 4.0 5.0 6.0]};
T = [0.1, 0.2];

% Define combinations
args = combinations(sigma, mu, T);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.noisevar = struct('args', Init, 'out', Init);
for i = 1:n_test
    sigma = args.sigma{i}(:);
    mu = args.mu{i}(:);
    T = args.T(i);
    Set.noisevar(i).args = {sigma, mu, T};
    Set.noisevar(i).out = {noisevar(sigma, mu, T)};
end

%% NOISEAMP
% Set required inputs
sigma = {[0.5  1.0  1.5]; [6.0 7.0 8.0]; [9.0 5.0 3.0]};
mu = {[1.0 2.0 3.0 4.0]; [3.0 4.0 5.0 6.0]};
T = [0.1, 0.2];

% Define combinations
args = combinations(sigma, mu, T);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.noiseamp = struct('args', Init, 'out', Init);
for i = 1:n_test
    sigma = args.sigma{i}(:);
    mu = args.mu{i}(:);
    T = args.T(i);
    Set.noiseamp(i).args = {sigma, mu, T};
    Set.noiseamp(i).out = {noiseamp(sigma, mu, T)};
end

%% SHIFTMTX
% Set required inputs
tau = [1.0, 2.0];
n = [256, 257];
ts = [1.0, 2.0];

% Define combinations
args = combinations(tau, n, ts);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.shiftmtx = struct('args', Init, 'out', Init);
for i = 1:n_test
    tau = args.tau(i);
    n = args.n(i);
    ts = args.ts(i);
    Set.shiftmtx(i).args = {tau, int64(n), ts};
    Set.shiftmtx(i).out = {shiftmtx(tau, n, ts)};
end

%% TDTF
% Set required inputs
theta = {[3.0 4.0];  [5.0 6.0]};
N = [5, 6];
ts = [0.2, 0.3];

% Define combinations
args = combinations(theta, N, ts);
n_test = height(args);

fun = @(theta, w) theta(1)*exp(-1i*theta(2)*w);

% Generate output
Init = cell(n_test, 1);
Set.tdtf = struct('args', Init, 'out', Init);
for i = 1:n_test
    theta = args.theta{i};
    N = args.N(i);
    ts = args.ts(i);
    Set.tdtf(i).args = {theta, int64(N), ts};
    Set.tdtf(i).out = {tdtf(fun, theta, N, ts)};
end


%% TDNOISEFIT    
% Set required inputs
N = 100;
M = 50;
ts = 0.1;

x1 = thzgen(N, ts, N*ts/2);
xIdeal = x1(:, ones(M,1));

sAlpha = 1e-4;
sBeta = 1e-2;
sTau = 1e-3;
noise = noiseamp([sAlpha, sBeta, sTau], xIdeal, ts).*randn(N,M);

x = xIdeal + noise;

% Set optional inputs
% Fix structure
Fix = struct();
Fix.logv = 0;
Fix.mu = 0;
Fix.A = 0;
Fix.eta = 0;

% Ignore structure
Ignore = struct();
Ignore.A = 1;
Ignore.eta = 0;

varargin = [struct() struct()];
v0 = [[sAlpha; sBeta; sTau].^2, [sAlpha; 0.1 * sBeta; 0.1 * sTau].^2];

for i = 1:1:length(varargin)
    % Initial parameters
    varargin(i).v0 = v0(:,i);
    varargin(i).mu0 = x(:, 1);
    varargin(i).A0 = ones(M,1);
    varargin(i).eta0 = zeros(M,1);
    varargin(i).ts = ts;
    varargin(i).Fix = Fix;
    varargin(i).Ignore = Ignore;
end

% Define combinations
args = combinations(varargin);
n_test = height(args);

% Generate output
Init = cell(n_test, 1);
Set.tdnoisefit = struct('args', Init, 'out', Init);
for i = 1:n_test
    varargin = args.varargin(i);
    Set.tdnoisefit(i).args = {x, varargin.v0, varargin.mu0, varargin.A0,...
        varargin.eta0, varargin.ts};
    P = tdnoisefit(x, varargin);
    Set.tdnoisefit(i).out = {P.var, P.mu, P.A, P.eta};
end

save thztools_test_data -v7.3 Set