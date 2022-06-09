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
sigma = [0.5, 1.0];
mu = [1.0, 2.0];
T = [0.1, 1.0];

% Generate output
Init = cell(length(sigma), length(mu), length(T));
Set.thzgen = struct('sigma', Init, 'mu', Init, 'T', Init, 'Vmu', Init);
for i = 1:length(sigma)
    for j = 1:length(mu)
        for k = 1:length(T)
            Set.thzgen(i,j,k).sigma = sigma(i);
            Set.thzgen(i,j,k).mu = T(j);
            Set.thzgen(i,j,k).T = t0(k);
            Set.thzgen(i,j,k).Vmu = noisevar(sigma(i), mu(j), T(k));
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
Set.thzgen = struct('N', Init, 't0', Init, 'w', Init, 'A', Init, 'T', Init, 'y', Init);
for i = 1:length(N)
    for j = 1:length(t0)
        for k = 1:length(w)
            for r = 1:length(A)
                for s = 1:length(T)
                    Set.thzgen(i,j,k,r,s).N = N(i);
                    Set.thzgen(i,j,k,r,s).t0 = t0(j);
                    Set.thzgen(i,j,k,r,s).w = w(k);
                    Set.thzgen(i,j,k,r,s).A = A(r);
                    Set.thzgen(i,j,k,r,s).T = T(s);
                    Set.thzgen(i,j,k,r,s).y = pulsegen(N(i), t0(j), w(k), A(r), T(s));
                end
            end
        end
    end
end

path(oldpath)

save pulsegen_test_data.mat -v7.3 Set

