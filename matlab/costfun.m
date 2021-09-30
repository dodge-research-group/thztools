function K = costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, ...
    sigma_tau, ts)
%COSTFUN Computes the MLE cost function

xx = xx(:);
yy = yy(:);
mu = mu(:);
sigma = [sigma_alpha; sigma_beta; sigma_tau];

n = length(xx);
H = tdtf(fun, theta, n, ts);

psi = H*mu;

Vmu = diag(noisevar(sigma, mu, ts));
Vpsi = diag(noisevar(sigma, psi, ts));

iVmu = diag(1./noisevar(sigma, mu, ts));
iVpsi = diag(1./noisevar(sigma, psi, ts));

% Compute the inverse covariance matrices for xx and yy
iVx = diag(1./noisevar(sigma, xx, ts));
iVy = diag(1./noisevar(sigma, yy, ts));

% Compute cost function
% Note that sigmamu and sigmapsi both have determinants below the numerical
% precision, so we multiply them by the constant matrices isigmaxx and
% isigmayy to improve numerical stability
K = log(det(iVx*Vmu)) + log(det(iVy*Vpsi)) + ...
    (xx - mu).'*iVmu*(xx - mu) + (yy - psi).'*iVpsi*(yy - psi);

end