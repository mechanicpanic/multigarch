"""Reference tests for numba kernels against pure-numpy implementations."""

import numpy as np
import pytest

from multigarch._jit import (
    _ETA_MAX,
    _ETA_MIN,
    _MAX_PERSISTENCE,
    DIST_NORMAL,
    DIST_T,
    _projected_gradient_norm,
    dcc_cl_loglik,
    dcc_covariance_loop,
    dcc_final_covariance,
    dcc_loglik_loop,
    digamma,
    garch_fit_projected,
    garch_loglik,
    garch_loglik_grad,
    garch_project,
    garch_variance_loop,
)


def _neg_moment(e):
    neg = np.minimum(e, 0.0)
    return neg.T @ neg / len(e)


def _numpy_dcc_nll(std_resid, Q_bar, a, b, N_bar=None, g=0.0):
    """Straightforward numpy (A)DCC negative log-likelihood (0.5*(logdet+quad), t>=1)."""
    T, n = std_resid.shape
    N_bar = np.zeros((n, n)) if N_bar is None else N_bar
    Q = Q_bar.copy()
    nll = 0.0
    for t in range(1, T):
        e = std_resid[t - 1]
        ne = np.minimum(e, 0.0)
        Q = (1.0 - a - b) * Q_bar - g * N_bar + a * np.outer(e, e) + g * np.outer(ne, ne) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        _, logdet = np.linalg.slogdet(R)
        x = std_resid[t]
        nll += logdet + x @ np.linalg.solve(R, x)
    return 0.5 * nll


@pytest.mark.parametrize("n", [2, 4, 7])
@pytest.mark.parametrize(
    "abg", [(0.05, 0.90, 0.0), (0.02, 0.97, 0.0), (0.20, 0.50, 0.0), (0.03, 0.90, 0.04)]
)
def test_full_loglik_matches_numpy_reference(n, abg):
    a, b, g = abg
    rng = np.random.default_rng(2)
    e = rng.standard_normal((60, n))
    Q_bar = np.corrcoef(e.T)
    N_bar = _neg_moment(e)
    val = dcc_loglik_loop(e, Q_bar, N_bar, a, b, g)
    ref = _numpy_dcc_nll(e, Q_bar, a, b, N_bar, g)
    assert val == pytest.approx(ref, rel=1e-9)


def _numpy_cl_nll(e, Q_bar, a, b, N_bar=None, g=0.0):
    """Pairwise composite (A)DCC NLL over contiguous pairs, pure numpy."""
    T, n = e.shape
    N_bar = np.zeros((n, n)) if N_bar is None else N_bar
    ne = np.minimum(e, 0.0)
    total = 0.0
    for i in range(n - 1):
        j = i + 1
        qii = qjj = 1.0
        qij = Q_bar[i, j]
        for t in range(1, T):
            qii = (
                (1 - a - b)
                - g * N_bar[i, i]
                + a * e[t - 1, i] ** 2
                + g * ne[t - 1, i] ** 2
                + b * qii
            )
            qjj = (
                (1 - a - b)
                - g * N_bar[j, j]
                + a * e[t - 1, j] ** 2
                + g * ne[t - 1, j] ** 2
                + b * qjj
            )
            qij = (
                (1 - a - b) * Q_bar[i, j]
                - g * N_bar[i, j]
                + a * e[t - 1, i] * e[t - 1, j]
                + g * ne[t - 1, i] * ne[t - 1, j]
                + b * qij
            )
            rho = qij / np.sqrt(qii * qjj)
            om = 1.0 - rho * rho
            total += np.log(om) + (e[t, i] ** 2 + e[t, j] ** 2 - 2 * rho * e[t, i] * e[t, j]) / om
    return 0.5 * total


@pytest.mark.parametrize("n", [2, 5, 12])
@pytest.mark.parametrize("g", [0.0, 0.04])
def test_cl_loglik_matches_numpy_reference(n, g):
    rng = np.random.default_rng(3)
    e = rng.standard_normal((80, n))
    Q_bar = np.corrcoef(e.T)
    N_bar = _neg_moment(e)
    val = dcc_cl_loglik(e, Q_bar, N_bar, 0.05, 0.88, g)
    ref = _numpy_cl_nll(e, Q_bar, 0.05, 0.88, N_bar, g)
    assert val == pytest.approx(ref, rel=1e-9)


def test_cl_loglik_single_asset_is_zero():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((50, 1))
    assert dcc_cl_loglik(e, np.array([[1.0]]), np.zeros((1, 1)), 0.05, 0.90, 0.0) == 0.0


SPECS = [
    (1, 0, 1, DIST_NORMAL),
    (2, 0, 1, DIST_NORMAL),
    (1, 0, 2, DIST_NORMAL),
    (2, 0, 2, DIST_NORMAL),
    (1, 1, 1, DIST_NORMAL),
    (1, 0, 1, DIST_T),
    (1, 1, 1, DIST_T),
    (2, 1, 1, DIST_T),
    (1, 2, 2, DIST_T),
]


def _params(p, o, q, dist):
    x = [[0.08], np.full(p, 0.04 / p), np.full(o, 0.03 / max(o, 1))[:o], np.full(q, 0.85 / q)]
    if dist == DIST_T:
        x.append([1.0 / 7.0])
    return np.concatenate(x)


@pytest.mark.parametrize("spec", SPECS)
def test_garch_grad_matches_value_and_finite_differences(spec):
    p, o, q, dist = spec
    rng = np.random.default_rng(5)
    r = rng.standard_normal(300)
    var0 = float(np.var(r))
    x = _params(p, o, q, dist)

    def f(z):
        return garch_loglik(r, z, p, o, q, dist, var0)

    val, grad = garch_loglik_grad(r, x, p, o, q, dist, var0)
    assert val == pytest.approx(f(x), rel=1e-12)

    h = 1e-6
    fd = np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(len(x))])
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_gjr_variance_matches_explicit_recursion():
    r = np.random.default_rng(13).standard_normal(400)
    x = np.array([0.06, 0.04, 0.05, 0.85])  # g = gamma / 2 = 0.05
    s = np.empty(400)
    s[0] = 1.0
    for t in range(1, 400):
        s[t] = (
            x[0]
            + x[1] * r[t - 1] ** 2
            + 2 * x[2] * r[t - 1] ** 2 * (r[t - 1] < 0)
            + x[3] * s[t - 1]
        )
    np.testing.assert_allclose(garch_variance_loop(r, x, 1, 1, 1, 1.0), s, rtol=1e-14)


def test_t_loglik_matches_scipy():
    from scipy import stats

    r = np.random.default_rng(14).standard_normal(400) * 1.3
    nu = 6.0
    x = np.array([0.06, 0.04, 0.05, 0.85, 1.0 / nu])
    s2 = garch_variance_loop(r, x, 1, 1, 1, 1.0)
    ref = -stats.t.logpdf(r, df=nu, scale=np.sqrt(s2 * (nu - 2) / nu)).sum()
    assert garch_loglik(r, x, 1, 1, 1, DIST_T, 1.0) == pytest.approx(ref, rel=1e-12)


def test_digamma_matches_scipy():
    from scipy import special

    for v in np.r_[np.linspace(0.05, 3.0, 40), np.linspace(3.0, 300.0, 40)]:
        assert digamma(v) == pytest.approx(special.digamma(v), rel=1e-11, abs=1e-12)


def test_garch_project_matches_brute_force_qp():
    from scipy.optimize import minimize

    rng = np.random.default_rng(7)
    for _ in range(50):
        k1 = int(rng.integers(2, 6))
        x = rng.normal(0.3, 0.6, k1)
        out = np.empty(k1)
        garch_project(x, out, k1 - 1)
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


def test_garch_project_clips_eta_and_leaves_it_out_of_the_sum():
    out = np.empty(5)
    garch_project(np.array([0.1, 0.6, 0.3, 0.4, 0.9]), out, 3)
    assert out[1:4].sum() == pytest.approx(_MAX_PERSISTENCE)
    assert out[4] == _ETA_MAX
    garch_project(np.array([0.1, 0.1, 0.1, 0.1, -1.0]), out, 3)
    assert out[4] == _ETA_MIN


def _fit_projected(r, x0, p, q):
    z = r / r.std()
    x0 = np.asarray(x0, dtype=np.float64)
    x, f, ok = garch_fit_projected(z, x0, p, 0, q, DIST_NORMAL, 1.0, 500, 1e-7, 1e-10)
    g = garch_loglik_grad(z, x, p, 0, q, DIST_NORMAL, 1.0)[1] / len(z)
    return x, ok, _projected_gradient_norm(x, g, np.empty(len(x)), p + q)


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
    N_bar = _neg_moment(e)
    R, H, Q = dcc_final_covariance(e, sig, Q_bar, N_bar, 0.04, 0.90, 0.03)
    R_path, H_path, Q_path = dcc_covariance_loop(e, sig, Q_bar, N_bar, 0.04, 0.90, 0.03)
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
            vals.add(dcc_loglik_loop(e, Q_bar, _neg_moment(e), 0.05, 0.88, 0.03))
    finally:
        numba.set_num_threads(prev)
    assert len(vals) == 1


def test_full_loglik_finite_for_singular_target():
    rng = np.random.default_rng(12)
    e = rng.standard_normal((200, 5))
    e[:, 4] = e[:, 3]  # rank-deficient Q_bar: the Cholesky pivot floor engages
    val = dcc_loglik_loop(e, np.corrcoef(e.T), np.zeros((5, 5)), 0.05, 0.90, 0.0)
    assert np.isfinite(val)
