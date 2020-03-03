function K = costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, ...
    sigma_tau, ts)
<<<<<<< HEAD
=======
%COSTFUN Computes the MLE cost function
>>>>>>> 6e744793b12ed83024b140af8f64e3d754a27ec9

xx = xx(:);
yy = yy(:);
mu = mu(:);

n = length(xx);
H = tdtf(fun, theta, n, ts);

psi = H*mu;
<<<<<<< HEAD
isigmamu = sigmagen(mu, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);
isigmapsi = sigmagen(psi, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);

K = (xx - mu).'*isigmamu*(xx - mu) + (yy - psi).'*isigmapsi*(yy - psi);
=======
sigmamu = sigmagen(mu, sigma_alpha, sigma_beta, sigma_tau, ts);
sigmapsi = sigmagen(psi, sigma_alpha, sigma_beta, sigma_tau, ts);

isigmamu = sigmagen(mu, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);
isigmapsi = sigmagen(psi, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);

% Compute the inverse covariance matrices for xx and yy
isigmaxx = sigmagen(xx, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);
isigmayy = sigmagen(yy, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);

% Compute cost function
% Note that sigmamu and sigmapsi both have determinants below the numerical
% precision, so we multiply them by the constant matrices isigmaxx and
% isigmayy to improve numerical stability
K = log(det(isigmaxx*sigmamu)) + log(det(isigmayy*sigmapsi)) + ...
    (xx - mu).'*isigmamu*(xx - mu) + (yy - psi).'*isigmapsi*(yy - psi);
>>>>>>> 6e744793b12ed83024b140af8f64e3d754a27ec9

end