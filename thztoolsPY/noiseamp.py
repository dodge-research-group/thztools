import numpy as np

from thztoolsPY.noisevar import noisevar


def sigmamu(sigma, mu, t):
    return np.sqrt(noisevar(sigma, mu, t))
