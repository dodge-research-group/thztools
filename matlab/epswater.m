function epsilonr = epswater(f,varargin)
%EPSWATER Complex relative permittivity of water
%   Interpolation function taken from W. J. Ellison, doi:10.1063/1.2360986
%
% EPSWATER(F) computes the complex relative permittivity at frequency F
% (THz) and temperature T = 25 deg C
%
% EPSWATER(F,T) computes the complex relative permittivity at frequency F
% (THz) and temperature T (deg C)

Input = inputParser;
addRequired(Input,'f',@isreal);
addOptional(Input,'T',25,@isreal)

parse(Input,f,varargin{:});
t
f = Input.Results.f*1e12;
T = Input.Results.T;

% Define relaxation parameters
a = [79.23882, 3.815866, 1.634967];
b = [0.004300598, 0.01117295, 0.006841548];
c = [1.382264e-13, 3.510354e-16, 6.30035e-15];
d = [652.7648, 1249.533, 405.5169];
Tc = 133.1383;

% Define resonance parameters
p0 = 0.8379692;
p = [-0.006118594, -0.000012936798, 4235901000000, -14260880000, ...
    273815700, -1246943, 9.618642e-14, 1.795786e-16, -9.310017e-18, ...
    1.655473e-19,...
    0.6165532, 0.007238532, -0.00009523366, 15983170000000, ...
    -74413570000, 497448000, 2.882476e-14, -3.142118e-16, 3.528051e-18];

% Compute temperature-dependent functions

eps0 = 87.9144 - 0.404399*T + 9.58726e-4*T.^2 - 1.32802e-6*T.^3;
Delta = a.*exp(-b*T);
tau = c.*exp(d/(T+Tc));

Delta4 = p0 + p(1)*T + p(2)*T.^2;
f0 = p(3) + p(4)*T + p(5)*T.^2 + p(6)*T.^3;
tau4 = p(7) + p(8)*T + p(9)*T.^2 + p(10)*T.^3;
Delta5 = p(11) + p(12)*T + p(13)*T.^2;
f1 = p(14) + p(15)*T + p(16)*T.^2;
tau5 = p(17) + p(18)*T + p(19)*T.^2;

% Put it all together
epsilonr = eps0 + 2i*pi*f.*(Delta(1)*tau(1)./(1-2i*pi*f*tau(1)) ...
    + Delta(2)*tau(2)./(1-2i*pi*f*tau(2)) ...
    + Delta(3)*tau(3)./(1-2i*pi*f*tau(3))) ...
    + 1i*pi*f.*(Delta4*tau4./(1-2i*pi*tau4*(f0+f)) ...
    + Delta4*tau4./(1+2i*pi*tau4*(f0-f))) ...
    + 1i*pi*f.*(Delta5*tau5./(1-2i*pi*tau5*(f1+f)) ...
    + Delta5*tau5./(1+2i*pi*tau5*(f1-f)));

end

