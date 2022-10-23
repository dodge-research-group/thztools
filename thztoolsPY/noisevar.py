import numpy as np
from numpy.fft import fftfreq


def noisevar(sigma, mu, t):

    """
 Noisevar computes the time-domain noise variance for noise parameters sigma, with a signal mu and sampling interval T.
 There are three noise parameters: the first corresponds to amplitude noise, in signal units (i.e, the same units as mu)
 ;  the second corresponds to multiplicative noise, which is dimensionless; and the third corresponds to timebase noise,
 in units of signal/time, where the units for time are the same as for T. The output, Vmu, is given in units that are
 the square of the signal units.



 Parameters
 ----------

   sigma : ndarray
    (3, ) array  containing noise parameters

   mu : ndarray
       (n, ) array  containing  signal vector

   t : float
    Sampling time

 Returns
 --------
   vmu : ndarray
        Noise variance


    """

    n = len(mu)
    w = 2 * np.pi * fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j * w * np.fft.fft(mu)))

    return sigma[0]**2 + (sigma[1] * mu)**2 + (sigma[2] * mudot)**2
