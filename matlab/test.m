%% Test FFTFREQ

% Set inputs
N = [4, 5];
T = [0.1, 1];

Init = cell(length(N), length(T));
Set = struct('N', Init, 'T', Init, 'f', Init);
for i = 1:length(N)
    for j = 1:length(T)
        Set(i,j).N = N(i);
        Set(i,j).T = T(j);
        Set(i,j).f = fftfreq(N(i), T(j));
    end
end

save test_data.mat -v7.3 Set