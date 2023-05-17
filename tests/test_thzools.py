import h5py
import numpy as np
import pandas as pd
import unittest
import os
import pathlib
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thztools.thztools import fftfreq, thzgen, noisevar, epswater, costfunlsq, tdtf, shiftmtx, tdnll, tdnoisefit


class Name(unittest.TestCase):
    # ==================================================================
    def test_fftfreq(self):
        dt = 0.5

        # Odd Case

        f_odd_true = [0., 0.4, 0.8, -0.8, -0.4]
        f_odd = fftfreq(len(f_odd_true), dt)
        assert_array_almost_equal(f_odd_true, f_odd, decimal=12)

        # Even Case
        f_even_true = [0., 0.25, 0.5, 0.75, 1., -0.75, -0.5, -0.25]
        f_even = fftfreq(len(f_even_true), dt)
        assert_array_almost_equal(f_even, f_even_true, decimal=12)

    # ==================================================================
    def test_noisevar(self):
        sigma_alpha = 1e-4,  # Additive noise amplitude [relative to peak]
        sigma_beta = 0.01,  # Multiplicative noise amplitude [-]
        sigma_tau = 1e-3,  # Time base noise amplitude [ps]
        sigma_vec = np.array([sigma_alpha, sigma_beta, sigma_tau])

        y, t = thzgen(n=20, ts=0.05, t0=2.5)
        vmu = noisevar(sigma_vec, y, 2.5)
        vmu_true = np.array(
            [4.36242890663103e-6, 1.18964333287937e-5, 1.95293889001674e-5, 2.55422643627414e-5, 2.93161838588929e-5,
             3.09039003252619e-5, 3.06822632940482e-5, 2.90574034275636e-5, 2.57635783418052e-5, 1.82432209755600e-5,
             4.91505996620606e-6, 2.38368994086406e-6, 3.48676583001915e-5, 8.17878674468416e-5, 0.000100010020280983,
             8.17871774183767e-5, 4.81773804791099e-05, 1.93749781259091e-5, 3.62858609090918e-6, 1.05084791456656e-07])

        assert_array_almost_equal(vmu, vmu_true, decimal=12)

    # ==================================================================

    def test_epswater(self):
        epsr_true = 4.1783 + 2.4381j
        epsr = epswater(1, t=25)
        np.testing.assert_allclose(np.round(epsr, 4), epsr_true)

    # ==================================================================

    def test_thzgen(self):
        cur_path = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'test_files' / 'thzgen_out.csv'

        thzgen_true = pd.read_csv(new_path, header=None)
        thzgen_true = np.array(thzgen_true[0])

        thz = thzgen(n=256, ts=0.05, t0=2.5)[0]
        np.testing.assert_allclose(thz, thzgen_true, atol=1e-5)

    # ==================================================================

    def test_cosfunlsq(self):
        def fun(theta, wfft):
            return theta[0] * np.exp(1j * theta[1] * wfft)

        theta = np.array([1, 2])
        n = 100
        ampSigma = 1e-5
        xx = np.column_stack([np.linspace(0, 10, n)])
        xx = np.squeeze(xx)
        yy = thzgen(n, 1, 1)[0] + ampSigma * np.ones(n)

        wfft = 2 * np.pi * np.fft.fftfreq(n, 1)
        sigmax = ampSigma * np.ones(n)
        sigmay = ampSigma * np.ones(n)

        # read data from matlab
        cur_path = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'test_files' / 'out_costfunlsq_rand.csv'
        costfunlsq_true_rand = pd.read_csv(new_path, header=None)
        cosfunlsq_true_rand = np.array(costfunlsq_true_rand[0])


        yy = thzgen(n, 1, 1)[0] + ampSigma * np.random.rand(n)

        wfft_r = 2 * np.pi * np.fft.fftfreq(n)
        sigmax_r = ampSigma * np.random.rand(n)
        sigmay_r = ampSigma * np.random.rand(n)
        cosfunlsq_rand = costfunlsq(fun, theta, xx, yy, sigmax_r, sigmay_r, 1)
        np.testing.assert_allclose(cosfunlsq_true_rand, cosfunlsq_rand, rtol=1e2, atol=1e2)

    # ==================================================================

    def test_tdtf(self):
        theta = np.array([3., 4.])
        n = 5
        ts = 0.2

        def fun(theta, wfft):
            return theta[0] * np.exp(1j * theta[1] * wfft)

        tdft_true = np.array([[3., -2.50e-15, 1.54e-15, -1.54e-15, 2.50e-15],
                              [2.50e-15, 3., -2.50e-15, 1.54e-15, -1.54e-15],
                              [-1.54e-15, 2.50e-15, 3., -2.50e-15, 1.54e-15],
                              [1.54e-15, -1.54e-15, 2.50e-15, 3., -2.50e-15],
                              [-2.50e-15, 1.54e-15, -1.54e-15, 2.50e-15, 3.]])

        tdt_p = tdtf(fun, theta, n, ts)
        assert_array_almost_equal(tdft_true, tdt_p)

    # ==================================================================

    def test_shiftmtx(self):
        tau = 1.0
        n = 256
        ts = 1
        cur_path = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'test_files' / 'shiftmtx_out.csv'

        shiftmtx_true = pd.read_csv(new_path, header=None)
        shiftmtx_true = np.array(shiftmtx_true)

        shiftmtx_p = shiftmtx(tau, n, ts)
        assert_array_almost_equal(shiftmtx_p, shiftmtx_true)

    # ==================================================================

    def test_tdnll(self):
        n = 10
        m = 8
        x = np.ones((n, m))

        # read gradient obtanied in python
        cur_path = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'test_files' / 'tdnll_out.csv'
        gradnll_true = pd.read_csv(new_path, header=None)
        gradnll_true = np.array(gradnll_true[0])

        # define parameters
        logv = np.ones(3)
        mu = np.ones(n)
        a = np.ones(m)
        eta = np.ones(m)
        ts = 0.5

        theta = np.array([0, 0])

        def fun(theta, w):
            return -1j * w

        d = tdtf(fun, theta, n, ts)

        param = {'logv': logv, 'mu': mu, 'a': a, 'eta': eta, 'ts': ts, 'd': d}
        fix = {'logv': 0, 'mu': 0, 'a': 0, 'eta': 0}

        [nll, gradnll] = tdnll(x, param, fix)
        assert_array_almost_equal(gradnll, gradnll_true)

    def test_tdnoisefit(self):
        cur_path = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'test_files' / 'tdnoisefit_test_python.mat'
        with h5py.File(new_path, 'r') as file:
            Set = file['Set']
            x = Set['tdnoisefit']['x'][0]
            param_test = Set['tdnoisefit']['paramForPy']
            fix_test = Set['tdnoisefit']['fixForPy']
            ignore_test = Set['tdnoisefit']['ignoreForPy']
            p_test = Set['tdnoisefit']['P']
            diagnostic = Set['tdnoisefit']['Diagnostic']
            dfx = np.array(x)
            dfParam = np.array(param_test)

            x = np.array(file[Set['tdnoisefit']['x'][0, 0]]).T
            param = {'v0': np.array(file[param_test[0, 0]]['v0'])[0],
                     'mu0': np.array(file[param_test[0, 0]]['mu0'])[0],
                     'a0': np.array(file[param_test[0, 0]]['A0'])[0],
                     'eta0': np.array(file[param_test[0, 0]]['eta0'])[0],
                     'ts': np.array(file[param_test[0, 0]]['ts'])[0]}
            fix = {'logv': bool(np.array(file[fix_test[0, 0]]['logv'])),
                   'mu': bool(np.array(file[fix_test[0, 0]]['mu'])),
                   'a': bool(np.array(file[fix_test[0, 0]]['A'])),
                   'eta': bool(np.array(file[fix_test[0, 0]]['eta']))}
            ignore = {'a': bool(np.array(file[ignore_test[0, 0]]['A'])),
                      'eta': bool(np.array(file[ignore_test[0, 0]]['eta']))}
            p = {'var': np.array(file[p_test[0, 0]]['var'])[0],
                 'mu': np.array(file[p_test[0, 0]]['mu'])[0],
                 'a': np.array(file[p_test[0, 0]]['A'])[0],
                 'eta': np.array(file[p_test[0, 0]]['eta'])[0],
                 'ts': np.array(file[p_test[0, 0]]['ts'])[0]}
            fun = np.array(file[Set['tdnoisefit']['fval'][0, 0]])[0]
            diagnostic = np.array(file[Set['tdnoisefit']['Diagnostic'][0, 0]])[0]

            [p_out, fun_out, diagnosticPy] = tdnoisefit(x, param, fix, ignore)

            assert_array_almost_equal(fun_out, fun, decimal=3)
            assert_array_almost_equal(p_out['var'], p['var'], decimal=3)
            assert_array_almost_equal(p_out['mu'], p['mu'], decimal=3)
            assert_array_almost_equal(p_out['a'], p['a'], decimal=3)
            assert_array_almost_equal(p_out['eta'], p['eta'], decimal=3)
            assert_array_almost_equal(p_out['ts'], p['ts'], decimal=3)



if __name__ == '_main_':
    unittest.main()
