%% Generate test data for all functions in thztools

% Add path
curdir = split(fileparts(mfilename('fullpath')), filesep);
mldir = fullfile(filesep, curdir{1:end-1}, 'matlab');
%oldpath = addpath(mldir);

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
%        Set.fftfreq(i,j).f = fftfreq(N(i), T(j));
    end
end

%% THZGEN
% Set required inputs
N = [256, 257];
T = 0.1;
t0 = 3.0;

% Set optional inputs
A = 2;
taur = 0.6;
tauc = 0.2;
taul = 0.025/sqrt(2*log(2));

% Generate output
Init = cell(length(N));
Set.thzgen = struct('N', Init, 't', Init, 'y', Init);
Set.thzgen.T = T;
Set.thzgen.t0 = t0;

path(oldpath)

save test_data.mat -v7.3 Set