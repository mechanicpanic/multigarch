"""Reference tests for numba kernels against pure-numpy implementations."""

import numpy as np
import pytest

from multigarch._jit import dcc_cl_loglik, dcc_loglik_loop


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
