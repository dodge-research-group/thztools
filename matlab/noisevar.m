function Vmu = noisevar(sigma, mu, T)
%NOISEVAR computes the time-domain noise amplitudes
%
% Syntax:   Vmu = noisevar(sigma,mu,T)
%
% Description:
%
% NOISEVAR computes the time-domain noise variance for noise parameters
% sigma, with a signal mu and sampling interval T. There are three noise
% parameters: the first corresponds to amplitude noise, in signal units
% (ie, the same units as mu); the second corresponds to multiplicative
% noise, which is dimensionless; and the third corresponds to timebase
% noise, in units of signal/time, where the units for time are the same as
% for T. The output, Vmu, is given in units that are the square of the
% signal units.
% 
%
% Inputs:
%   sigma   Noise parameters    [3x1 double]
%   mu      Signal vector       [Nx1 double]
%   T       Sampling time       [1x1 double]
%
% Outputs:
%   Vmu     Noise variance      [Nx1 double]


    [N,~] = size(mu);
    w = 2*pi*fftfreq(N,T);
    mudot = real(ifft(1i*w.*fft(mu)));
    
    Vmu = sigma(1).^2+(sigma(2)*mu).^2+(sigma(3)*mudot).^2;
end