import numpy as np
from numpy.fft import fftfreq


def noisevar(sigma, mu, t):
    """
    NOISEVAR computes the time-domain noise amplitudes

    Syntax:   Vmu = noisevar(sigma, mu, t)

    Description:

    NOISEVAR computes the time-domain noise amplitudes for noise parameters
    sigma, with a signal mu and sampling interval t. There are three noise
    parameters: the first corresponds to amplitude noise, in signal units
    (ie, the same units as mu); the second corresponds to multiplicative
    noise, which is dimensionless; and the third corresponds to timebase
    noise, in units of signal/time, where the units for time are the same as
    for t. The output, Vmu, is given in units that are the square of the
    signal units.

    Parameters
    ----------
        sigma   Noise parameters    [1x3 double]
        mu      Signal vector       [1xN double]
        t       Sampling time       double

    Returns
    -------
        Vmu     Noise variance      [1xN double]
    """

    n = len(mu)
    w = 2 * np.pi * fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j * w * np.fft.fft(mu)))

    return sigma[0]**2 + (sigma[1] * mu)**2 + (sigma[2] * mudot)**2