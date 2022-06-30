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


save tdtf_test_data1.mat -v7.3 Set




