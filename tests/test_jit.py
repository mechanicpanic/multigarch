"""Reference tests for numba kernels against pure-numpy implementations."""

import numpy as np
import pytest

from multigarch._jit import (
    _MAX_PERSISTENCE,
    _projected_gradient_norm,
    dcc_cl_loglik,
    dcc_covariance_loop,
    dcc_final_covariance,
    dcc_loglik_loop,
    garch_fit_projected,
    garch_loglik,
    garch_loglik_grad,
    garch_project,
)


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


def test_garch_project_matches_brute_force_qp():
    from scipy.optimize import minimize

    rng = np.random.default_rng(7)
    for _ in range(50):
        k1 = int(rng.integers(2, 6))
        x = rng.normal(0.3, 0.6, k1)
        out = np.empty(k1)
        garch_project(x, out)
        ref = minimize(
            lambda y, x=x: ((y - x) ** 2).sum(),
            np.full(k1, 0.01),
            method="SLSQP",
            bounds=[(1e-12, None)] + [(0.0, None)] * (k1 - 1),
            constraints=[{"type": "ineq", "fun": lambda y: _MAX_PERSISTENCE - y[1:].sum()}],
            options={"ftol": 1e-14},
        )
        np.testing.assert_allclose(out, ref.x, atol=1e-6)
        assert out[1:].min() >= 0.0
        assert out[1:].sum() <= _MAX_PERSISTENCE + 1e-12


def _fit_projected(r, x0, p, q):
    z = r / r.std()
    x, f, ok = garch_fit_projected(z, np.asarray(x0, dtype=np.float64), p, q, 1.0, 500, 1e-7, 1e-10)
    g = garch_loglik_grad(z, x, p, q, 1.0)[1] / len(z)
    return x, ok, _projected_gradient_norm(x, g, np.empty(len(x)))


def test_projected_fit_reaches_alpha_boundary_exactly():
    # For this white-noise draw the MLE has alpha = 0 on the boundary,
    # which the projection reaches exactly
    r = np.random.default_rng(8).standard_normal(3000)
    x, ok, pg = _fit_projected(r, [0.05, 0.05, 0.90], 1, 1)
    assert ok
    assert x[1] == 0.0
    assert pg < 1e-4


def test_projected_fit_respects_persistence_cap():
    # Integrated-variance data push persistence to the cap
    rng = np.random.default_rng(9)
    T = 3000
    r = np.empty(T)
    s2 = 1.0
    for t in range(T):
        r[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = 0.002 + 0.12 * r[t] ** 2 + 0.8799 * s2
    x, ok, pg = _fit_projected(r, [0.05, 0.05, 0.90], 1, 1)
    assert ok
    assert x[1:].sum() == pytest.approx(_MAX_PERSISTENCE, abs=1e-9)
    assert pg < 1e-4


@pytest.mark.parametrize("n", [1, 2, 5, 8, 33])
def test_dcc_final_covariance_matches_path_last_step(n):
    # Parallel row-paired kernel vs the sequential full-path kernel
    rng = np.random.default_rng(10)
    e = rng.standard_normal((120, n))
    sig = np.abs(rng.standard_normal((120, n))) + 0.1
    Q_bar = np.corrcoef(e.T) if n > 1 else np.ones((1, 1))
    R, H, Q = dcc_final_covariance(e, sig, Q_bar, 0.04, 0.93)
    R_path, H_path, Q_path = dcc_covariance_loop(e, sig, Q_bar, 0.04, 0.93)
    np.testing.assert_allclose(Q, Q_path, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(R, R_path[-1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(H, H_path[-1], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(Q, Q.T)


@pytest.mark.parametrize("n", [1, 3, 8])
@pytest.mark.parametrize("T", [2, 3, 70, 401])
def test_full_loglik_independent_of_thread_count(n, T):
    import numba

    e = np.random.default_rng(11).standard_normal((T, n))
    Q_bar = np.corrcoef(e.T) if n > 1 and T > n else np.eye(n)
    prev = numba.get_num_threads()
    try:
        vals = set()
        for k in sorted({1, 2, prev}):
            numba.set_num_threads(k)
            vals.add(dcc_loglik_loop(e, Q_bar, 0.05, 0.90))
    finally:
        numba.set_num_threads(prev)
    assert len(vals) == 1


def test_full_loglik_finite_for_singular_target():
    rng = np.random.default_rng(12)
    e = rng.standard_normal((200, 5))
    e[:, 4] = e[:, 3]  # rank-deficient Q_bar: the Cholesky pivot floor engages
    val = dcc_loglik_loop(e, np.corrcoef(e.T), 0.05, 0.90)
    assert np.isfinite(val)
