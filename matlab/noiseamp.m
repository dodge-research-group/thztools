function sigmamu = noiseamp(sigma, mu, T)
%NOISEAMP computes the time-domain noise amplitudes
%
% Syntax:   sigmamu = noiseamp(sigma,mu,T)
%
% Description:
%
% NOISEAMP computes the time-domain noise amplitudes for noise parameters
% sigma, with a signal mu and sampling interval T. There are three noise
% parameters: the first corresponds to amplitude noise, in signal units
% (ie, the same units as mu); the second corresponds to multiplicative
% noise, which is dimensionless; and the third corresponds to timebase
% noise, in units of signal/time, where the units for time are the same as
% for T. The output, sigmamu, is given in signal units.
% 
%
% Inputs:
%   sigma   Noise parameters    [3x1 double]
%   mu      Signal vector       [Nx1 double]
%   T       Sampling time       [1x1 double]
%
% Outputs:
%   sigmamu Noise amplitude     [Nx1 double]

sigmamu = sqrt(noisevar(sigma,mu,T));

end