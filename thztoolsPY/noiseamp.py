import numpy as np
from thztools.thztoolsPY.noisevar import noisevar


def sigmamu(sigma, mu, T):
    return np.sqrt(noisevar(sigma, mu, T))