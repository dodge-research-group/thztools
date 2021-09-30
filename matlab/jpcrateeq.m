function dthetadt = jpcrateeq(t,theta)
%JPCRATEEQ Rate equations for photoconducting antenna current

n = theta(1);
v = theta(2);
p = theta(3);

taus = 0.1;         % Drude scattering lifetime [ps]
tauc = 0.4;         % Carrier capture time [ps]
taur = 100;         % Carrier recombination time [ps]
taul = 0.05/sqrt(2*log(2));        % Laser pulse 1/e width [ps]
mstar = 0.063;
epsilonr = 12.9;
eta = 3;
g0 = 2e8;
Ebias = 100/400;    % Bias field (V/µm)

e_over_m = 1e-12*SI.qe/SI.me;
e_over_eps0 = 1e6*SI.qe/SI.eps0;

dvdt = -v/(taus) - e_over_m*Ebias/mstar ...
    + e_over_m*e_over_eps0*p/(mstar*eta*epsilonr*taur);
dpdt = -p/taur - n*v;
dndt = -n/tauc + g0*exp(-(t/taul).^2/2);

dthetadt = [dndt;dvdt;dpdt];
end

