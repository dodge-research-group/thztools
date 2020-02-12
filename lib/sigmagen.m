function sigma = sigmagen(mu, sigma_alpha, sigma_beta, sigma_tau, ts)

mu = mu(:);

t = (0:length(mu)-1)*ts;
t = t(:);

sigma = diag(sigma_alpha^2 + (sigma_beta*mu).^2 + ...
    (sigma_tau*(gradient(mu,t))).^2);

end