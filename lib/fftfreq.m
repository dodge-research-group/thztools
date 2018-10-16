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

% Use IFFTSHIFT to shift zero frequency from center to first position.
kcyclic = ifftshift((0:N-1)-floor(N/2));

f = kcyclic*fs;
f = f(:);
    
end