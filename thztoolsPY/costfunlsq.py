import numpy as np
import scipy


def costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft):
    """

    Parameters
    ----------
    fun
    theta
    xx
    yy
    sigmax
    sigmay
    wfft

    Returns
    -------

    """
    N = len(sigmax)
    H = np.conj(fun(theta, wfft))
    if N % 2 == 0:
        kNy = N/2
        H[kNy] = np.real(H(kNy+1))

    ry = yy - np.real(np.fft.ifft(np.fft.fft(xx)*H))
    Vy = np.diag(sigmay**2)

    Htilde = np.fft.ifft(H)

    Uy = np.zeros((N, N))
    for k in np.arange(N):
        Uy = Uy + np.real(np.roll(Htilde, k) * np.roll(Htilde, k).H) * sigmax[k] ** 2

    W = np.eye(N) / scipy.linalg.sqrtm(Uy + Vy)
    res = W*ry

    return res
