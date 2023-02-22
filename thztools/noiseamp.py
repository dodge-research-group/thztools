import numpy as np
from thztoolsPY.noisevar import noisevar


def sigmamu(sigma, mu, t):
    """
    noiseamp computes the time-domain noise amplitudes for noise parameters sigma, with a signal mu and sampling
    interval t.  There are three noise parameters: the first corresponds to amplitude noise, in signal units
    (ie, the same units as mu); the second corresponds to multiplicative noise, which is dimensionless; and the
    third corresponds to timebase noise, in units of signal/time, where the units for time are the same as
    for t. The output, sigmamu, is given in signal units.

    Parameters
    ----------

    sigma : ndarray
        (3, ) array  containing noise parameters

    mu :  signal vector
        (n, ) array  containing  signal vector

    t : float 
        sampling time
        
    Returns
    -------
    sigmamu : ndarray
        (n, ) array containing noise amplitude

    """

    return np.sqrt(noisevar(sigma, mu, t))
