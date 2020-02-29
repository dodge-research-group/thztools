function f =  fftfreq(N,ts)
%FFTFREQ computes the positive and negative frequencies sampled in the FFT
%
% Syntax:   f = fftfreq(N,ts)
%
% Description:
%
% FFTFREQ computes a column vector of N frequencies with sampling frequency
% 1/ts, with circular frequencies mapped to positive and negative
% frequencies that are centered at zero.
%
% Inputs:
%   N   Number of time samples
%   ts  Sampling time
%
% Outputs:
%   f   Frequency vector (1/ts)
%

validateattributes(N,{'numeric'},{'scalar','positive'})
validateattributes(ts,{'numeric'},{'scalar','positive'})

fs = 1/(ts*N);

% Use CIRCSHIFT to shift zero frequency to first position
%
% For even N,
%   kcirc = [0, 1, ... , N/2, -N/2 + 1, -N/2 + 2, ... , -1]
%
% For odd N,
%   kcirc = [0, 1, ... , (N - 1)/2, -(N - 1)/2, -(N - 1)/2 + 1, ..., -1]
%
kcirc = circshift((1:N) - ceil(N/2), floor(N/2)+1);

f = kcirc*fs;
f = f(:);
    
end