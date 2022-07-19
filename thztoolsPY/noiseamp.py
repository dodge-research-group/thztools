import numpy as np

from thztools.thztoolsPY.noisevar import noisevar


def noiseamp(sigma, mu, t):
    """
    NOISEAMP computes the time-domain noise amplitudes

    Syntax:   sigmamu = noiseamp(sigma, mu, T)

    Description:

    NOISEAMP computes the time-domain noise amplitudes for noise parameters
    sigma, with a signal mu and sampling interval t. There are three noise
    parameters: the first corresponds to amplitude noise, in signal units
    (ie, the same units as mu); the second corresponds to multiplicative
    noise, which is dimensionless; and the third corresponds to timebase
    noise, in units of signal/time, where the units for time are the same as
    for t. The output, sigmamu, is given in signal units.

    Parameters
    ----------
        sigma   Noise parameters    [1x3 double]
        mu      Signal vector       [1xN double]
        t       Sampling time       double

    Returns
    -------
        sigmamu Noise amplitude     1xN numpy array
    """
    return np.sqrt(noisevar(sigma, mu, t))
