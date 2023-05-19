%% Generate test data for all functions in thztools

% Add path
curdir = split(fileparts(mfilename('fullpath')), filesep);
mldir = fullfile(filesep, curdir{1:end-1}, 'matlab');
oldpath = addpath(mldir);

%% FFTFREQ
% Set inputs
N = [4, 5];
T = [0.1, 1];

% Generate output
Init = cell(length(N), length(T));
Set.fftfreq = struct('N', Init, 'T', Init, 'f', Init);
for i = 1:length(N)
    for j = 1:length(T)
        Set.fftfreq(i,j).N = N(i);
        Set.fftfreq(i,j).T = T(j);
        Set.fftfreq(i,j).f = fftfreq(N(i), T(j));
    end
end

path(oldpath)

save fftfreq_test_data.mat -v7.3 Set

%% EPSWATER
% Set inputs
f = [1, 2];
T = [20, 30];

% Generate output
Init = cell(length(f), length(T));
Set.epswater = struct('f', Init, 'T', Init, 'epsR', Init, 'epsI', Init);
for i = 1:length(f)
    for j = 1:length(T)
        Set.epswater(i,j).f = f(i);
        Set.epswater(i,j).T = T(j);
        Set.epswater(i,j).epsR = real(epswater(f(i), T(j)));
        Set.epswater(i,j).epsI = imag(epswater(f(i), T(j)));
    end
end

path(oldpath)

save epswater_test_data.mat -v7.3 Set

%% COSTFUNLSQ
% Set required inputs
fun = @(theta, wfft) theta(1) * exp(1i * theta(2) * wfft);
theta = rand(2, 3);
N = 100;
xx = [linspace(0, 10, N)' linspace(0, 10, N)' linspace(0, 10, N)'];
yy = [thzgen(N, 1, 1) thzgen(N, 2, 2) thzgen(N, 3, 3)] + ampSigma*rand(N, 3);
wfft = 2*pi*[fftfreq(N, 1) fftfreq(N, 2) fftfreq(N, 3)];
ampSigma = 1e-5;
sigmax = ampSigma*rand(N, 3);
sigmay = ampSigma*rand(N, 3);

% Set optional inputs

% Generate output
Init = cell(size(theta,2), size(xx,2), size(yy,2), size(sigmax,2), size(sigmay,2), size(wfft,2));
Set.costfunlsq = struct('theta', Init, 'xx', Init, 'yy', Init, 'sigmax', Init, ...
    'sigmay', Init, 'wfft', Init, 'res', Init);

for i = 1:size(theta,2)
    for j = 1:size(xx,2)
        for k = 1:size(yy,2)
            for l = 1:size(sigmax,2)
                for m = 1:size(sigmay,2)
                    for n = 1:size(wfft,2)
                        Set.costfunlsq(i,j,k,l,m,n).theta = theta(:, i);
                        Set.costfunlsq(i,j,k,l,m,n).xx = xx(:, j);
                        Set.costfunlsq(i,j,k,l,m,n).yy = yy(:, k);
                        Set.costfunlsq(i,j,k,l,m,n).sigmax = sigmax(:, l);
                        Set.costfunlsq(i,j,k,l,m,n).sigmay = sigmay(:, m);
                        Set.costfunlsq(i,j,k,l,m,n).wfft = wfft(:, n);
                        Set.costfunlsq(i,j,k,l,m,n).res = costfunlsq(fun, theta(:, i), xx(:, j), yy(:, k), ...
                            sigmax(:, l), sigmay(:, m), wfft(:, n));
                    end
                end
            end
        end
    end
end

path(oldpath)

save costfunlsq_test_data.mat -v7.3 Set

%% TDNLL
% Set required inputs
clc
N = 10;
M = 8;
x = rand(N, M, 4);
Param = [struct() struct() struct()];
varargin = [struct() struct() struct()];
for i = 1:1:3
    Param(i).logv = rand(3,1);
    Param(i).mu = rand(N,1);
    Param(i).A = rand(M,1);
    Param(i).eta = rand(M,1);
    Param(i).ts = rand();
    Param(i).D = tdtf(@(theta,w) -1i*w, 0, N, Param(i).ts);
    
    % Set optional inputs
    varargin(i).logv = 0;
    varargin(i).mu = 0;
    varargin(i).A = 0;
    varargin(i).eta = 0;
end

% Generate output
Init = cell(size(x,3), size(Param,2), size(varargin,2));
Set.tdnll = struct('x', Init, 'Param', Init, 'varargin', Init, 'nll', Init, 'gradnll', Init);

for i = 1:size(x,3)
    for j = 1:size(Param,2)
        for k = 1:size(varargin,2)
            Set.tdnll(i,j,k).x = x(:, :, i);
            Set.tdnll(i,j,k).Param = Param(j);
            Set.tdnll(i,j,k).varargin = varargin(k);
            [Set.tdnll(i,j,k).nll, Set.tdnll(i,j,k).gradnll] = tdnll(x(:, :, i),Param(j),varargin(k));
            disp(Set.tdnll(i,j,k).nll)
        end
    end
end

path(oldpath)

save tdnll_test_data.mat -v7.3 Set


%% THZGEN
% Set required inputs
N = [256, 257];
T = [0.1, 1];
t0 = [2.0, 3.0];

% Set optional inputs
A = 2;
taur = 0.6;
tauc = 0.2;
taul = 0.025/sqrt(2*log(2));

% Generate output
Init = cell(length(N), length(T), length(t0));
Set.thzgen = struct('N', Init, 'T', Init, 't0', Init, 'y', Init);
for i = 1:length(N)
    for j = 1:length(T)
        for k = 1:length(t0)
            Set.thzgen(i,j,k).N = N(i);
            Set.thzgen(i,j,k).T = T(j);
            Set.thzgen(i,j,k).t0 = t0(k);
            Set.thzgen(i,j,k).y = thzgen(N(i), T(j), t0(k));
        end
    end
end

path(oldpath)

save thzgen_test_data.mat -v7.3 Set

%% NOISEVAR
% Set required inputs
sigma = ([0.5  1.0  1.5; 6.0 7.0 8.0; 9.0 5.0 3.0])';
mu = ([1.0 2.0 3.0 4.0; 3.0 4.0 5.0 6.0])';
T = [0.1, 0.2];

% Generate output
Init = cell(size(sigma, 2), size(mu, 2), length(T));
Set.noisevar = struct('sigma', Init, 'mu', Init, 'T', Init, 'Vmu', Init);
for i = 1:size(sigma, 2)
    for j = 1:size(mu, 2)
        for k = 1:length(T)
            Set.noisevar(i,j,k).sigma = sigma(:, i);
            Set.noisevar(i,j,k).mu = mu(:, j);
            Set.noisevar(i,j,k).T = T(k);
            Set.noisevar(i,j,k).Vmu = noisevar(sigma(:, i), mu(:, j), T(k));
        end
    end
end

path(oldpath)


save noisevar_test_data.mat -v7.3 Set



%% PULSEGEN
% Set required inputs
N = [256, 257];
t0 = [2.0, 3.0];
w = [1];
A = [1];
T = [1];

% Generate output
Init = cell(length(N), length(t0), length(w),  length(A),  length(T));
Set.pulsegen = struct('N', Init, 't0', Init, 'w', Init, 'A', Init, 'T', Init, 'y', Init);
for i = 1:length(N)
    for j = 1:length(t0)
        for k = 1:length(w)
            for r = 1:length(A)
                for s = 1:length(T)
                    Set.pulsegen(i,j,k,r,s).N = N(i);
                    Set.pulsegen(i,j,k,r,s).t0 = t0(j);
                    Set.pulsegen(i,j,k,r,s).w = w(k);
                    Set.pulsegen(i,j,k,r,s).A = A(r);
                    Set.pulsegen(i,j,k,r,s).T = T(s);
                    Set.pulsegen(i,j,k,r,s).y = pulsegen(N(i), t0(j), w(k), A(r), T(s));
                end
            end
        end
    end
end

path(oldpath)

save pulsegen_test_data.mat -v7.3 Set

%% SHIFTMTX
% Set required inputs
tau = [1.0, 2.0];
n = [256, 257];
ts = [1.0, 2.0];

% Generate output
Init = cell(length(tau), length(n), length(ts));
Set.shiftmtx = struct('tau', Init, 'n', Init, 'ts', Init, 'h', Init);
for i = 1:length(N)
    for j = 1:length(t0)
        for k = 1:length(w)
            for r = 1:length(A)
                for s = 1:length(T)
                    Set.shiftmtx(i,j,k).tau = tau(i);
                    Set.shiftmtx(i,j,k).n = n(j);
                    Set.shiftmtx(i,j,k).ts = ts(k);
                    Set.shiftmtx(i,j,k).h = shiftmtx(tau(i), n(j), ts(k));
                end
            end
        end
    end
end

path(oldpath)

save shiftmtx_test_data.mat -v7.3 Set

%% TDTF
% Set required inputs
theta = ([3.0 4.0;  5.0 6.0])';
N = [5, 6];
ts = [0.2, 0.3];
fun = @(theta, w) theta(1)*exp(-1i*theta(2)*w);


% Generate output
Init = cell(size(theta, 2), length(N), length(ts));
Set.tdtf = struct('theta', Init, 'N', Init, 'ts', Init, 'h', Init); 
for i = 1:size(theta, 2)
    for j = 1:length(N)
       for k = 1:length(ts)
            Set.tdtf(i, j, k).theta = theta(:, i);
            Set.tdtf(i, j, k).N = N(j);
            Set.tdtf(i, j, k).ts = ts(k);
            Set.tdtf(i, j, k).h = tdtf(fun, theta(:, i), N(j), ts(k));
        end
   end
end


save tdtf_test_data.mat -v7.3 Set

%% TDNOISEFIT    
% Set required inputs
clc
N = 100;
M = 50;
ts = 0.1;

x1 = thzgen(N, ts, N*ts/2);
xIdeal = x1(:, ones(M,1));

sAlpha = 1e-1;
sBeta = 1e-2;
sTau = 1e-3;
noise = sqrt(sAlpha^2 + (sBeta .* xIdeal).^2 + sTau).*randn(N,M);

x = xIdeal + noise;

varargin = [struct() struct()];
paramForPy = [struct() struct()];
fixForPy = [struct() struct()];
ignoreForPy = [struct() struct()];
for i = 1:1:length(varargin)
    % Initial parameters
    varargin(i).v0 = rand(3,1);
    varargin(i).mu0 = x(:, 1);
    varargin(i).A0 = 1 - 1e-10*ones(M,1);
    varargin(i).eta0 = 1e-10*ones(M,1);
    varargin(i).ts = ts;

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
    
    varargin(i).Fix = Fix;
    varargin(i).Ignore = Ignore;

    paramForPy(i).v0 = varargin(i).v0;
    paramForPy(i).mu0 = varargin(i).mu0;
    paramForPy(i).A0 = varargin(i).A0;
    paramForPy(i).eta0 = varargin(i).eta0;
    paramForPy(i).ts = varargin(i).ts;

    fixForPy(i).logv = Fix.logv;
    fixForPy(i).mu = Fix.mu;
    fixForPy(i).A = Fix.A;
    fixForPy(i).eta = Fix.eta;
    ignoreForPy(i).A = Ignore.A;
    ignoreForPy(i).eta = Ignore.eta;
end

% Generate output
Init = cell(size(x,3), size(varargin,2));
Set.tdnoisefit = struct('x', Init, 'paramForPy', Init,  'fixForPy', Init, 'ignoreForPy', Init, 'P', Init, 'fval', Init, 'Diagnostic', Init);

for i = 1:size(x,3)
    for j = 1:size(varargin,2)
        Set.tdnoisefit(i,j).x = x(:, :, i);
        Set.tdnoisefit(i,j).paramForPy = paramForPy(j);
        Set.tdnoisefit(i,j).fixForPy = fixForPy(j);
        Set.tdnoisefit(i,j).ignoreForPy = ignoreForPy(j);
        [Set.tdnoisefit(i,j).P, Set.tdnoisefit(i,j).fval, Set.tdnoisefit(i,j).Diagnostic] = tdnoisefitNew(x(:, :, i),varargin(j));
    end
end

path(oldpath)

save tdnoisefit_test_data.mat -v7.3 Set

plot(xIdeal(:,1))
hold on
plot(x(:,1))
plot(Set.tdnoisefit(1,1).P.mu)
legend('Ideal', 'Ideal + Noise', 'Fit')

%% Noisefitshow (using dataToptica)

% Generate Output

Data = dataToptica;
nDir = length(Data);

%Run loop backwards to preallocate structure array upon first iteration
Data = Data(nDir:-1:1);
for iDir = length(Data):-1:1
    t = Data(iDir).t;
    X = Data(iDir).X;
    [~, dirName] = fileparts(Data(iDir).Dir);
    fprintf("%s\n", repmat('-',1,50));
    fprintf("%s\n", repmat('-',1,50));
    fprintf('Fit noise model to data in %s\n', dirName)
    [Output(iDir), Options, P] = noisefitshow(t, X, replace(dirName, '_', ' '));
end


% data 50 ps, 2 avgs


   eta2 = Output(2).P(4).eta;
   A2 = Output(2).P(4).A;
   mu2 = Output(2).P(4).mu; 
   v2 = Output(2).P(4).var;
   Xadjusted2 = Output(2).Xadjusted;
   sigmaTotStar2 = Output(2).sigmaTotStar;
   X2 = Output(2).X;

%%%%%%%%%%%%%%%%%%
% data, 50 ps, 100 avgs


   eta1 = Output(1).P(4).eta;
   A1 = Output(1).P(4).A;
   mu1 = Output(1).P(4).mu; 
   v1 = Output(1).P(4).var;
   Xadjusted1 = Output(1).Xadjusted;
   sigmaTotStar1 = Output(1).sigmaTotStar;
   X1 = Output(1).X;


 
    dataToptica = struct('X1', X1,  'eta1', eta1, 'mu1', mu1, 'A1', A1, 'v1', v1, 'Xadjusted1', Xadjusted1, ...
         'sigmaTotstar1', sigmaTotStar1, 'X2', X2,  'eta2', eta2, 'mu2', mu2, 'A2', A2, 'v2', v2, ...
         'Xadjusted2', Xadjusted2, 'sigmaTotstar2', sigmaTotStar2, 't', t);
 
   
 
   save dataToptica.mat -v7.3 dataToptica;

