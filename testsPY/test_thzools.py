import h5py
import numpy as np
import pandas as pd
import unittest
import os
import pathlib
from numpy.testing import assert_array_almost_equal, assert_array_equal
from thztoolsPY.costfunlsq import costfunlsq
from thztoolsPY.epswater import epswater
from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.pulsegen import pulsegen
from thztoolsPY.noisevar import noisevar
from thztoolsPY.shiftmtx import shiftmtx
from thztoolsPY.tdtf import tdtf
from thztoolsPY.thzgen import thzgen


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
    def test_pulsegen(self):
        n = 20
        t0 = 1
        w = 0.2
        a = 5
        t = 0.1

        y = [-0.0000, -0.0000, -0.0000, -0.0006, -0.0105, -0.1110, -0.6410, -1.8445, -1.8394,
             1.9470, 5.0000, 1.9470, -1.8394, -1.8445, -0.6410, -0.1110, -0.0105, -0.0006, -0.0000, -0.0000]
        y_out, t_out = pulsegen(n, t0, w, a, t)

        assert_array_almost_equal(y_out, y, decimal=4)
        assert_array_almost_equal(t_out, t * np.arange(len(y)), decimal=12)

    # ==================================================================
    def test_noisevar(self):
        sigma_alpha = 1e-4,  # Additive noise amplitude [relative to peak]
        sigma_beta = 0.01,  # Multiplicative noise amplitude [-]
        sigma_tau = 1e-3,  # Time base noise amplitude [ps]
        sigma_vec = np.array([sigma_alpha, sigma_beta, sigma_tau])

        y, t = thzgen(n=20, t=0.05, t0=2.5)
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
        thzgen_true = pd.read_csv('thzgen_out.csv', header=None)
        thzgen_true = np.array(thzgen_true[0])

        thz = thzgen(n=256, t=0.05, t0=2.5)[0]
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

        #print(os.path.dirname(__file__))
        #fname = os.path.join(os.path.dirname(__file__), 'out_costfunlsq_ones.csv')


        #pd.read_csv('C:\\Users\\jonap\\thz_project\\thztools\\testsPY\\out_cosfunlsq_ones.csv')
        #pd.read_csv(r'C:\Users\jonap\thz_project\thztools\testsPY\out_cosfunlsq_ones.csv', header=None)


        #cosfunlsq_true_ones = np.array(cosfunlsq_true[0])
        #aa = os.path.join(os.path.dirname(__file__), 'out_costfunlsq_rand.csv')
        #a1 = pd.read_csv('aa', header=None)
        #a2 = np.array(a1[0])
        #print('a', aa)
        #cosfunlsq = costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft)
        #np.testing.assert_allclose(cosfunlsq_true, cosfunlsq, rtol=1e2, atol=1e2)

        # =====================================================================
        # random noise
        cur_path  = pathlib.Path(__file__).parent.resolve()
        new_path = cur_path / 'out_costfunlsq_rand.csv'


        costfunlsq_true_rand = pd.read_csv(new_path, header=None)
        costfunlsq_true_rand = pd.read_csv(r'C:\Users\jonap\thz_project\thztools\testsPY\out_cosfunlsq_ones.csv', header=None)
        cosfunlsq_true_rand = np.array(costfunlsq_true_rand[0])
        yy = thzgen(n, 1, 1)[0] + ampSigma * np.random.rand(n)

        wfft_r = 2 * np.pi * np.fft.fftfreq(n)
        sigmax_r = ampSigma * np.random.rand(n)
        sigmay_r = ampSigma * np.random.rand(n)
        cosfunlsq_rand = costfunlsq(fun, theta, xx, yy, sigmax_r, sigmay_r, wfft_r)
        np.testing.assert_allclose(cosfunlsq_true_rand, cosfunlsq_rand, rtol=1e2, atol=1e2)


        # ======================================================================
        # random noise
        #costfunlsq_true_rand = pd.read_csv('out_costfunlsq_rand.csv', header=None)
        #costfunlsq_true_rand = pd.read_csv(r'C:\Users\jonap\thz_project\thztools\testsPY\out_cosfunlsq_ones.csv', header=None)
        #cosfunlsq_true_rand = np.array(costfunlsq_true_rand[0])
        #yy = thzgen(n, 1, 1)[0] + ampSigma * np.random.rand(n)

        #wfft_r = 2 * np.pi * np.fft.fftfreq(n)
        #sigmax_r = ampSigma * np.random.rand(n)
        #sigmay_r = ampSigma * np.random.rand(n)
        #cosfunlsq_rand = costfunlsq(fun, theta, xx, yy, sigmax_r, sigmay_r, wfft_r)
        #np.testing.assert_allclose(cosfunlsq_true_rand, cosfunlsq_rand, rtol=1e2, atol=1e2)

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

        shiftmtx_true = pd.read_csv('shiftmtx_out.csv', header=None)
        shiftmtx_true = np.array(shiftmtx_true)

        shiftmtx_p = shiftmtx(tau, n, ts)
        assert_array_almost_equal(shiftmtx_p, shiftmtx_true)

    # ==================================================================

    def test_tdnll(self):
        n = 10
        m = 8


if __name__ == '_main_':
    unittest.main()
