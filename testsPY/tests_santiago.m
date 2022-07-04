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

%% TDNOISEFIT    
% Set required inputs
clc
N = 10;
M = 8;
x = rand(N, M, 4);
varargin = [struct() struct() struct()];
paramForPy = [struct() struct() struct()];
fixForPy = [struct() struct() struct()];
ignoreForPy = [struct() struct() struct()];
for i = 1:1:length(varargin)
    % Initial parameters
    varargin(i).v0 = rand(3,1);
    varargin(i).mu0 = rand(N,1);
    varargin(i).A0 = rand(M,1);
    varargin(i).eta0 = rand(M,1);
    varargin(i).ts = rand();

    % Fix structure
    Fix = struct();
    Fix.logv = 0;
    Fix.mu = 0;
    Fix.A = 1;
    Fix.eta = 1;

    % Ignore structure
    Ignore = struct();
    Ignore.A = 1;
    Ignore.eta = 1;
    
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
        [Set.tdnoisefit(i,j).P, Set.tdnoisefit(i,j).fval, Set.tdnoisefit(i,j).Diagnostic] = tdnoisefit(x(:, :, i),varargin(j));
    end
end

path(oldpath)

save tdnoisefit_test_data.mat -v7.3 Set