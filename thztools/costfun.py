import numpy as np 
def costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, sigma_tau, ts):

    sigma = [sigma_alpha, sigma_beta, sigma_tau]
    
    n = len(xx)
    H = tdtf(fun, theta, n, ts)
    
    psi = H*mu
    
    Vmu = np.diag(noisevar(sigma, mu, ts))
    Vpsi = np.diag(noisevar(sigma, psi, ts))
    
    iVmu = np.diag(1/noisevar(sigma, mu,ts))
    iVpsi = np.diag(1/noisevar(sigma, psi, ts))
    
    
    
    iVx = diag(1/noisevar(sigma, xx, ts))
    iVy = diag(1/noisevar(sigma, yy, ts))
    
    
    K = np.log(np.linalg.det(iVx*iVmu)) + np.log(np.linalg.det(iVy*Vpsi)) + (xx-mu)*iVmu*(xx-mu) + (yy-psi)*(iVpsi)*(yy-psi)    
    