import numpy as np
from thztools.thztoolsPY.noisevar import noisevar

def costfun(fun,mu ,theta, xx, yy, sigma_alpha, sigma_beta, sigma_tau, ts):

    n = len(xx)
    pass