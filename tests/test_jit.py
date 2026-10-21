"""Reference tests for numba kernels against pure-numpy implementations."""

import numpy as np
import pytest

from multigarch._jit import dcc_cl_loglik, dcc_loglik_loop, garch_loglik, garch_loglik_grad


def _numpy_dcc_nll(std_resid, Q_bar, a, b):
    """Straightforward numpy DCC negative log-likelihood (0.5*(logdet+quad), t>=1)."""
    T, n = std_resid.shape
    Q = Q_bar.copy()
    nll = 0.0
    for t in range(1, T):
        e = std_resid[t - 1]
        Q = (1.0 - a - b) * Q_bar + a * np.outer(e, e) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        _, logdet = np.linalg.slogdet(R)
        x = std_resid[t]
        nll += logdet + x @ np.linalg.solve(R, x)
    return 0.5 * nll


@pytest.mark.parametrize("n", [2, 4, 7])
@pytest.mark.parametrize("ab", [(0.05, 0.90), (0.02, 0.97), (0.20, 0.50)])
def test_full_loglik_matches_numpy_reference(n, ab):
    a, b = ab
    rng = np.random.default_rng(2)
    e = rng.standard_normal((60, n))
    Q_bar = np.corrcoef(e.T)
    val = dcc_loglik_loop(e, Q_bar, a, b)
    ref = _numpy_dcc_nll(e, Q_bar, a, b)
    assert val == pytest.approx(ref, rel=1e-9)


def _numpy_cl_nll(e, Q_bar, a, b):
    """Pairwise composite NLL over contiguous pairs, pure numpy."""
    T, n = e.shape
    total = 0.0
    for i in range(n - 1):
        j = i + 1
        qii = qjj = 1.0
        qij = Q_bar[i, j]
        for t in range(1, T):
            qii = (1 - a - b) + a * e[t - 1, i] ** 2 + b * qii
            qjj = (1 - a - b) + a * e[t - 1, j] ** 2 + b * qjj
            qij = (1 - a - b) * Q_bar[i, j] + a * e[t - 1, i] * e[t - 1, j] + b * qij
            rho = qij / np.sqrt(qii * qjj)
            om = 1.0 - rho * rho
            total += np.log(om) + (e[t, i] ** 2 + e[t, j] ** 2 - 2 * rho * e[t, i] * e[t, j]) / om
    return 0.5 * total


@pytest.mark.parametrize("n", [2, 5, 12])
def test_cl_loglik_matches_numpy_reference(n):
    rng = np.random.default_rng(3)
    e = rng.standard_normal((80, n))
    Q_bar = np.corrcoef(e.T)
    val = dcc_cl_loglik(e, Q_bar, 0.05, 0.90)
    ref = _numpy_cl_nll(e, Q_bar, 0.05, 0.90)
    assert val == pytest.approx(ref, rel=1e-9)


def test_cl_loglik_single_asset_is_zero():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((50, 1))
    assert dcc_cl_loglik(e, np.array([[1.0]]), 0.05, 0.90) == 0.0


@pytest.mark.parametrize("pq", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_garch_grad_matches_value_and_finite_differences(pq):
    p, q = pq
    rng = np.random.default_rng(5)
    r = rng.standard_normal(300)
    var0 = float(np.var(r))
    x = np.concatenate([[0.08], np.full(p, 0.06 / p), np.full(q, 0.85 / q)])

    def f(z):
        return garch_loglik(r, z[0], z[1 : 1 + p], z[1 + p :], var0)

    val, grad = garch_loglik_grad(r, x, p, q, var0)
    assert val == pytest.approx(f(x), rel=1e-12)

    h = 1e-6
    fd = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)
