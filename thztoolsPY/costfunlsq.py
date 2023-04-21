import numpy as np
import scipy.linalg


def costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft):
    """

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        mu : ndarray
            Signal vector of size (n,).

        theta : ndarray
            Input parameters for the function.

        xx : ndarray
            Measured input signal.

        yy : ndarray
            Measured output signal.

        sigmax : ndarray or matrix
            Noise covariance matrix of the input signal.

        sigmay : nadarray
            Noise covariance matrix of the output signal.

        wfft :


    Returns
    -------
    res : callable


    """
    wfft1 = np.array([wfft])
    n = len(sigmax)
    #n =np.shape(wfft)[0]
    h = np.conj(fun(theta, wfft))
    if n % 2 == 0:
        kny = n // 2
        h[kny] = np.real(h[kny])

    ry = yy - np.real(np.fft.ifft(np.fft.fft(xx) * h))
    vy = np.diag(sigmay**2)

    htilde = np.fft.ifft(h)

    uy = np.zeros((n, n))
    for k in np.arange(n):
        a = np.reshape(np.roll(htilde, k), (n, 1))
        b = np.reshape(np.conj(np.roll(htilde, k)), (1, n))
        uy = uy + np.real(np.dot(a, b)) * sigmax[k]**2
        #uy = uy + np.real(np.roll(htilde, k-1) @ np.roll(htilde, k-1).T) @ sigmax[k]**2

    w = np.dot(np.eye(n), scipy.linalg.inv(scipy.linalg.sqrtm(uy + vy)))
    res = np.dot(w, ry)

    return res
