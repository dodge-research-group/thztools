from __future__ import annotations

import numpy as np
import pytest
import scipy
from numpy import pi
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike

from thztools.thztools import (
    NoiseModel,
    _costfuntls,
    _tdnll_scaled,
    fit,
    noiseamp,
    noisevar,
    scaleshift,
    tdnoisefit,
    thzgen,
    transfer_out,
)

atol = 1e-8
rtol = 1e-5


def tfun(p, w):
    return p[0] * np.exp(1j * p[1] * w)


def tfun1(w, p):
    return p[0] * np.exp(1j * p[1] * w)


def tfun2(w, p0, p1):
    return p0 * np.exp(1j * p1 * w)


def jac_fun(p, w):
    exp_ipw = np.exp(1j * p[1] * w)
    return np.stack((exp_ipw, 1j * w * p[0] * exp_ipw)).T


class TestNoiseModel:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    mu_dot = -2 * pi * np.sin(2 * pi * t)

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, np.ones(n)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, np.ones((2, n))),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, np.ones((n, 2))),
            (0, 1, 0, mu, dt, -1, mu**2),
            (0, 0, 1, mu, dt, -1, mu_dot**2),
        ],
    )
    def test_var_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: ArrayLike,
        dt: float,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        noise_model = NoiseModel(alpha, beta, tau, dt)
        assert_allclose(
            noise_model.variance(mu, dt=dt, axis=axis),  # type: ignore
            expected,  # type: ignore
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, np.ones(n)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, np.ones((2, n))),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, np.ones((n, 2))),
            (0, 1, 0, mu, dt, -1, np.abs(mu)),
            (0, 0, 1, mu, dt, -1, np.abs(mu_dot)),
        ],
    )
    def test_amp_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: ArrayLike,
        dt: float,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        noise_model = NoiseModel(alpha, beta, tau, dt)
        assert_allclose(
            noise_model.amplitude(mu, dt=dt, axis=axis),  # type: ignore
            expected,  # type: ignore
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, (n,)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, (2, n)),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, (n, 2)),
        ],
    )
    def test_noise_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: ArrayLike,
        dt: float,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        noise_model = NoiseModel(alpha, beta, tau, dt)
        assert noise_model.noise(mu, dt=dt, axis=axis).shape == expected


class TestTransferOut:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)

    @pytest.mark.parametrize("fft_sign", [True, False])
    @pytest.mark.parametrize(
        "t_fun, x, p, expected",
        [
            [tfun1, mu, [1.0, 0.0], mu],
            [tfun2, mu, (1.0, 0.0), mu],
        ],
    )
    def test_inputs(self, t_fun, x, fft_sign, p, expected):
        ts = self.dt
        assert_allclose(
            transfer_out(t_fun, x, dt=ts, fft_sign=fft_sign, args=p),
            expected,
        )

    @pytest.mark.parametrize("x", [np.ones((n, n))])
    def test_error(self, x):
        dt = self.dt
        with pytest.raises(ValueError):
            _ = transfer_out(x, tfun1, dt=dt, args=[1.0, 0.0])


class TestNoise:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    mu_dot = -2 * pi * np.sin(2 * pi * t)

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [
            ([1, 0, 0], mu, dt, np.ones(n)),
            ([0, 1, 0], mu, dt, mu**2),
            ([0, 0, 1], mu, dt, mu_dot**2),
        ],
    )
    def test_var_definition(
        self, sigma: ArrayLike, mu: ArrayLike, dt: float, expected: ArrayLike
    ) -> None:
        assert_allclose(
            noisevar(sigma, mu, dt),  # type: ignore
            expected,  # type: ignore
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [
            ([1, 0, 0], mu, dt, np.ones(n)),
            ([0, 1, 0], mu, dt, np.abs(mu)),
            ([0, 0, 1], mu, dt, np.abs(mu_dot)),
        ],
    )
    def test_amp_definition(
        self, sigma: ArrayLike, mu: ArrayLike, dt: float, expected: ArrayLike
    ) -> None:
        assert_allclose(
            noiseamp(sigma, mu, dt),  # type: ignore
            expected,  # type: ignore
            atol=atol,
            rtol=rtol,
        )


class TestTHzGen:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"a": 1.0},
            {"taur": 0.3},
            {"tauc": 0.1},
            {"fwhm": 0.05},
        ],
    )
    def test_inputs(self, kwargs: dict) -> None:
        n = 8
        dt = 1.0
        t0 = 2.0
        y_expected = np.array(
            [
                0.05651348,
                -0.13522073,
                1.0,
                -0.65804181,
                -0.28067975,
                0.05182924,
                -0.01837401,
                -0.01602642,
            ]
        )
        t_expected = np.arange(n)
        assert_allclose(
            thzgen(n, dt, t0, **kwargs),  # type: ignore
            (y_expected, t_expected),  # type: ignore
            atol=atol,
            rtol=rtol,
        )


class TestScaleShift:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    x = np.cos(2 * pi * t)
    x_2 = np.stack((x, x))

    @pytest.mark.parametrize(
        "x, kwargs, expected",
        [
            [[], {}, np.empty((0,))],
            [x, {}, x],
            [x, {"a": 2}, 2 * x],
            [x, {"eta": 1}, np.cos(2 * pi * (t - dt))],
            [x, {"a": 2, "eta": 1}, 2 * np.cos(2 * pi * (t - dt))],
            [x, {"a": 2, "eta": dt, "dt": dt}, 2 * np.cos(2 * pi * (t - dt))],
            [x_2, {"a": [2, 0.5]}, np.stack((2 * x, 0.5 * x))],
            [
                x_2,
                {"eta": [1, -1]},
                np.stack(
                    (np.cos(2 * pi * (t - dt)), np.cos(2 * pi * (t + dt)))
                ),
            ],
            [
                x_2,
                {"eta": [dt, -dt], "dt": dt},
                np.stack(
                    (np.cos(2 * pi * (t - dt)), np.cos(2 * pi * (t + dt)))
                ),
            ],
            [
                x_2,
                {"a": [2, 0.5], "eta": [1, -1]},
                np.stack(
                    (
                        2 * np.cos(2 * pi * (t - dt)),
                        0.5 * np.cos(2 * pi * (t + dt)),
                    )
                ),
            ],
            [
                x_2,
                {"a": [2, 0.5], "eta": [dt, -dt], "dt": dt},
                np.stack(
                    (
                        2 * np.cos(2 * pi * (t - dt)),
                        0.5 * np.cos(2 * pi * (t + dt)),
                    )
                ),
            ],
            [x_2.T, {"a": [2, 0.5], "axis": 0}, np.stack((2 * x, 0.5 * x)).T],
        ],
    )
    def test_inputs(
        self, x: ArrayLike, kwargs: dict, expected: ArrayLike
    ) -> None:
        assert_allclose(
            scaleshift(x, **kwargs),  # type: ignore
            expected,  # type: ignore
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "x, kwargs", [[x, {"a": [2, 0.5]}], [x, {"eta": [1, -1]}]]
    )
    def test_errors(self, x: ArrayLike, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            scaleshift(x, **kwargs)


class TestCostFunTLS:
    theta = (1, 0)
    mu = np.arange(8)
    xx = mu
    yy = xx
    sigmax = np.ones_like(xx)
    sigmay = sigmax
    dt = 1.0

    assert_allclose(
        _costfuntls(tfun, theta, mu, xx, yy, sigmax, sigmay, dt),
        np.concatenate((np.zeros_like(xx), np.zeros_like(xx))),
    )


class TestTDNLL:
    m = 2
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    x = np.tile(mu, [m, 1])
    logv = (0, -np.inf, -np.inf)
    delta = np.zeros(n)
    alpha = np.zeros(m - 1)
    eta = np.zeros(m - 1)
    desired_nll = x.size * np.log(2 * pi) / 2

    @pytest.mark.parametrize(
        "fix_logv, desired_gradnll_logv",
        [
            [
                True,
                [],
            ],
            [
                False,
                [16.0, 0.0, 0.0],
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fix_delta, desired_gradnll_delta",
        [
            [
                True,
                [],
            ],
            [
                False,
                np.zeros(n),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fix_alpha, desired_gradnll_alpha",
        [
            [
                True,
                [],
            ],
            [
                False,
                np.zeros(m - 1),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fix_eta, desired_gradnll_eta",
        [
            [
                True,
                [],
            ],
            [
                False,
                np.zeros(m - 1),
            ],
        ],
    )
    def test_gradnll_calc(
        self,
        fix_logv,
        fix_delta,
        fix_alpha,
        fix_eta,
        desired_gradnll_logv,
        desired_gradnll_delta,
        desired_gradnll_alpha,
        desired_gradnll_eta,
    ):
        n = self.n
        m = self.m
        x = self.x
        logv = self.logv
        delta = self.delta
        alpha = self.alpha
        eta = self.eta
        dt = self.dt
        desired_gradnll = np.concatenate(
            (
                desired_gradnll_logv,
                desired_gradnll_delta,
                desired_gradnll_alpha,
                desired_gradnll_eta,
            )
        )
        _, gradnll = _tdnll_scaled(
            x,
            logv,
            delta,
            alpha,
            eta,
            dt,
            fix_logv=fix_logv,
            fix_delta=fix_delta,
            fix_alpha=fix_alpha,
            fix_eta=fix_eta,
            scale_logv=np.ones(3),
            scale_delta=np.ones(n),
            scale_alpha=np.ones(m - 1),
            scale_eta=np.ones(m - 1),
            scale_v=1.0,
        )
        assert_allclose(
            gradnll, desired_gradnll, atol=10 * np.finfo(float).eps
        )


class TestTDNoiseFit:
    rng = np.random.default_rng(0)
    n = 256
    m = 64
    dt = 0.05
    t = np.arange(n) * dt
    mu, _ = thzgen(n, dt=dt, t0=n * dt / 3)
    sigma = np.array([1e-5, 1e-2, 1e-3])
    noise = noiseamp(sigma, mu, dt) * rng.standard_normal((m, n))
    x = np.array(mu + noise)
    a = np.ones(m)
    eta = np.zeros(m)

    @pytest.mark.parametrize("x", [x, x[:, 0]])
    @pytest.mark.parametrize("v0", [None, sigma**2, []])
    @pytest.mark.parametrize("mu0", [None, mu, []])
    @pytest.mark.parametrize("a0", [None, a, []])
    @pytest.mark.parametrize("eta0", [None, eta, []])
    @pytest.mark.parametrize("fix_v", [True, False])
    @pytest.mark.parametrize("fix_mu", [True, False])
    @pytest.mark.parametrize("fix_a", [True, False])
    @pytest.mark.parametrize("fix_eta", [True, False])
    def test_inputs(self, x, v0, mu0, a0, eta0, fix_v, fix_mu, fix_a, fix_eta):
        print(f"{scipy.__version__=}")
        n = self.n
        m = self.m
        if (
            x.ndim < 2
            or (v0 is not None and len(v0) != 3)
            or (mu0 is not None and len(mu0) != n)
            or (a0 is not None and len(a0) != m)
            or (eta0 is not None and len(eta0) != m)
            or (fix_v and fix_mu and fix_a and fix_eta)
        ):
            with pytest.raises(ValueError):
                _, _, _ = tdnoisefit(
                    x.T,
                    v0=v0,
                    mu0=mu0,
                    a0=a0,
                    eta0=eta0,
                    fix_v=fix_v,
                    fix_mu=fix_mu,
                    fix_a=fix_a,
                    fix_eta=fix_eta,
                )
        else:
            p, fval, diagnostic = tdnoisefit(
                x.T,
                v0=v0,
                mu0=mu0,
                a0=a0,
                eta0=eta0,
                fix_v=fix_v,
                fix_mu=fix_mu,
                fix_a=fix_a,
                fix_eta=fix_eta,
            )
            assert diagnostic["status"] == 0


class TestFit:
    rng = np.random.default_rng(0)
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    p0 = (1, 0)
    psi = mu
    sigma = np.array([1e-5, 0, 0])
    noise_amp = noiseamp(sigma, mu, dt)
    x = mu + noise_amp * rng.standard_normal(n)
    y = psi + noise_amp * rng.standard_normal(n)

    @pytest.mark.parametrize("noise_parms", [(1, 0, 0), sigma**2])
    @pytest.mark.parametrize("p_bounds", [None, ((0, -1), (2, 1))])
    @pytest.mark.parametrize("jac", [None, jac_fun])
    @pytest.mark.parametrize("kwargs", [None, {}])
    def test_inputs(self, noise_parms, p_bounds, jac, kwargs):
        p0 = self.p0
        x = self.x
        y = self.y
        dt = self.dt
        p = fit(
            tfun,
            p0,
            x,
            y,
            dt=dt,
            sigma_parms=noise_parms,
            p_bounds=p_bounds,
            jac=jac,
            kwargs=kwargs,
        )
        assert_allclose(p["p_opt"], p0, atol=1e-6)

    def test_errors(self):
        p0 = self.p0
        x = self.x
        y = self.y
        dt = self.dt
        with pytest.raises(ValueError):
            _ = fit(tfun, p0, x, y, dt=dt, p_bounds=())
