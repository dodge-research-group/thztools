import numpy as np

def tdtf(fun,theta,N,ts):
    ts = .5
    N = 30
    fs = 1/(ts*N)
    p = np.arange((N-1)/2)
    fp = fs*p
    wp = 2*np.pi*fp



    tfunp = fun(theta,wp)
    tpsh = tfunp[1:end]
    fliptpsh = np.flipud(tpsh)
    flipconj = np.conj(fliptpsh)

    if N % 2 != 0:
        tfun  = np.concatenate(tfunp,fliptpsh)
    else:
        wny = np.pi*N*fs
    twny = fun(theta,wNy)
    conjtwnh = np.concatenate(twny,fliptsh)
    fliptwny = np.concatenate(twny, fliptpsh)

    tfun = []