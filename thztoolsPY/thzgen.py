import numpy as np


def thzgen(n, t, t0, varargin):

    """
    Generate a terahertz pulse with n points at sampling interval t and centered at t0.

    Parameters
    ----------

    n : int
        number of sampled points.

    t : float
        sampling time

    t0 : float
        pulse center

    Returns
    -------

    y : ndarray
        signal (u.a)

    t : float
        timebase

    """

    default_a = 1
    default_taur = 0.3
    default_tauc = 0.1
    default_taul = 0.05 / np.sqrt(2 * np.log(2))

    a = default_a
    taur = default_taur
    tauc = default_tauc
    taul = default_taul

    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n / 2)] = -f[int(n / 2)]

    w = 2 * np.pi * f
    l = np.exp(-(w * taul) ** 2 / 2) / np.sqrt(2 * np.pi * taul ** 2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (l * r) ** 2 * np.exp(1j * w * t0)

    t2 = t * np.arange(n)

    y = np.real(np.fft.ifft(np.conj(s)))
    y = a * y / np.max(y)

    return [y, t2]
