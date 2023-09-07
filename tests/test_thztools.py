from __future__ import annotations

import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike

from thztools.thztools import (
    costfunlsq,
    costfuntls,
    noiseamp,
    noisevar,
    scaleshift,
    tdnll,
    tdnoisefit,
    thzgen,
)

atol = 1e-8
rtol = 1e-5


def tfun(p, w):
    return p[0] * np.exp(1j * p[1] * w)


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
        ts = 1.0
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
            thzgen(n, ts, t0, **kwargs),  # type: ignore
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
            [x, {"a": 2, "eta": dt, "ts": dt}, 2 * np.cos(2 * pi * (t - dt))],
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
                {"eta": [dt, -dt], "ts": dt},
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
                {"a": [2, 0.5], "eta": [dt, -dt], "ts": dt},
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


class TestCostFunLSQ:
    theta = [1, 0]
    xx = np.arange(8)
    yy = xx
    sigmax = np.ones_like(xx)
    sigmay = sigmax
    ts = 1.0

    assert_allclose(
        costfunlsq(tfun, theta, xx, yy, sigmax, sigmay, ts),
        np.zeros_like(xx),
    )


class TestCostFunTLS:
    theta = [1, 0]
    mu = np.arange(8)
    xx = mu
    yy = xx
    sigmax = np.ones_like(xx)
    sigmay = sigmax
    ts = 1.0

    assert_allclose(
        costfuntls(tfun, theta, mu, xx, yy, sigmax, sigmay, ts),
        np.concatenate((np.zeros_like(xx), np.zeros_like(xx))),
    )


class TestTDNLL:
    m = 2
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    x = np.tile(mu, [m, 1])
    logv = [0, -np.inf, -np.inf]
    a = np.ones(m)
    eta = np.zeros(m)
    ts = dt
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
        "fix_mu, desired_gradnll_mu",
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
        "fix_a, desired_gradnll_a",
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
        fix_mu,
        fix_a,
        fix_eta,
        desired_gradnll_logv,
        desired_gradnll_mu,
        desired_gradnll_a,
        desired_gradnll_eta,
    ):
        x = self.x
        mu = self.mu
        logv = self.logv
        a = self.a
        eta = self.eta
        ts = self.ts
        desired_gradnll = np.concatenate(
            (
                desired_gradnll_logv,
                desired_gradnll_mu,
                desired_gradnll_a,
                desired_gradnll_eta,
            )
        )
        _, gradnll = tdnll(
            x,
            mu,
            logv,
            a,
            eta,
            ts,
            fix_logv=fix_logv,
            fix_mu=fix_mu,
            fix_a=fix_a,
            fix_eta=fix_eta,
        )
        assert_allclose(
            gradnll, desired_gradnll, atol=10 * np.finfo(float).eps
        )


class TestTDNoiseFit:
    rng = np.random.default_rng(0)
    n = 64
    m = 8
    ts = 1.0 / n
    t = np.arange(n) * ts
    mu, _ = thzgen(n, ts=ts, t0=n * ts / 2)
    sigma = np.array([1e-5, 0, 0])
    noise = noiseamp(sigma, mu, ts) * rng.standard_normal((m, n))
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
    def test_inputs_new(
        self, x, v0, mu0, a0, eta0, fix_v, fix_mu, fix_a, fix_eta
    ):
        m = self.m
        n = self.n
        sigma = self.sigma
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
            assert_allclose(
                p["var"] * m / (m - 1), sigma**2, rtol=1e-8, atol=1e-10
            )
