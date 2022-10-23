import numpy as np


def pulsegen(n, t0, w, a, t):

    """ Pulsegen generates a short pulse of temporal width w centered at t0 for use in tests of time-domain analysis

    Parameters
    ----------

    n : int
        number of sampled points

    t0 : float
        pulse center

    w : float
        pulse width

    a : float
        pulse amplitude

    t : float
        sampling time


    Returns
    -------

    y : ndarray
        signal (u.a)

    t : float
        timebase

    """

    t2 = t*np.arange(n)
    tt = (t2-t0)/w

    y = a*(1-2*tt**2)*np.exp(-tt**2)

    return [y, t]
