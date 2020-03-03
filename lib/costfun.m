function K = costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, ...
    sigma_tau, ts)

xx = xx(:);
yy = yy(:);
mu = mu(:);

n = length(xx);
H = tdtf(fun, theta, n, ts);

psi = H*mu;
isigmamu = sigmagen(mu, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);
isigmapsi = sigmagen(psi, 1/sigma_alpha, 1/sigma_beta, 1/sigma_tau, ts);

K = (xx - mu).'*isigmamu*(xx - mu) + (yy - psi).'*isigmapsi*(yy - psi);

end