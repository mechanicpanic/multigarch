import numpy as np
import pytest

from multigarch import DCC


def simulate_dcc_returns(T, n, a, b, seed):
    """Simulate returns with DCC correlation dynamics and constant unit-ish vol."""
    rng = np.random.default_rng(seed)
    Q_bar = 0.5 * np.ones((n, n)) + 0.5 * np.eye(n)
    Q = Q_bar.copy()
    z = rng.standard_normal((T, n))
    eps = np.zeros((T, n))
    e_prev = np.zeros(n)
    for t in range(T):
        Q = (1 - a - b) * Q_bar + a * np.outer(e_prev, e_prev) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        L = np.linalg.cholesky(R)
        eps[t] = L @ z[t]
        e_prev = eps[t]
    return eps * 0.01


def test_dcc_forecast_uses_q_expectation():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((80, 3)) * 0.02

    model = DCC(p=1, q=1, n_jobs=1, low_memory=False).fit(returns)
    forecast_cov = model.forecast(1)[0]

    assert model.Q_last is not None
    assert model.Q_bar is not None

    # Expected next-step Q uses E[εε'] = Q̄
    Q_next = (1.0 - model.b) * model.Q_bar + model.b * model.Q_last
    inv = np.diag(1.0 / np.sqrt(np.diag(Q_next)))
    expected_R = inv @ Q_next @ inv

    vars_pred = np.diag(forecast_cov)
    R_pred = forecast_cov / np.sqrt(np.outer(vars_pred, vars_pred))

    assert np.allclose(R_pred, expected_R, atol=1e-3, rtol=1e-3)


def test_cl_agrees_with_full_on_small_n():
    returns = simulate_dcc_returns(1500, 3, a=0.06, b=0.90, seed=5)
    full = DCC(n_jobs=1, method="full").fit(returns)
    cl = DCC(n_jobs=1, method="cl").fit(returns)
    assert abs(full.a - cl.a) < 0.05
    assert abs(full.b - cl.b) < 0.10


def test_auto_dispatch():
    small = simulate_dcc_returns(300, 3, a=0.05, b=0.90, seed=6)
    m_small = DCC(n_jobs=1).fit(small)
    assert m_small.method_ == "full"

    large = simulate_dcc_returns(200, 30, a=0.05, b=0.90, seed=7)
    m_large = DCC(n_jobs=1).fit(large)
    assert m_large.method_ == "cl"
    assert m_large.a + m_large.b < 0.999 + 1e-8


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        DCC(method="exact")


def test_dcc_shrinkage_applies_to_q_bar():
    returns = simulate_dcc_returns(300, 3, a=0.05, b=0.90, seed=9)
    m = DCC(n_jobs=1, shrinkage=1.0).fit(returns)
    np.testing.assert_allclose(m.Q_bar, np.eye(3), atol=1e-12)


def test_dcc_constraint_respected():
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((400, 4)) * 0.01
    m = DCC(n_jobs=1).fit(returns)
    assert m.a >= 0 and m.b >= 0
    assert m.a + m.b <= 0.999 + 1e-8


def test_dcc_low_memory_matches_final_step():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=10)
    full = DCC(n_jobs=1, low_memory=False).fit(returns)
    lm = DCC(n_jobs=1, low_memory=True).fit(returns)
    np.testing.assert_allclose(lm.H, full.H[-1], rtol=1e-8)
    np.testing.assert_allclose(lm.R, full.R[-1], rtol=1e-8)
    np.testing.assert_allclose(lm.Q_last, full.Q_last, rtol=1e-8)


def test_dcc_invariants():
    returns = simulate_dcc_returns(400, 4, a=0.05, b=0.90, seed=11)
    m = DCC(n_jobs=1).fit(returns)
    for t in [0, 200, 399]:
        np.testing.assert_allclose(np.diag(m.R[t]), 1.0, atol=1e-10)
        np.testing.assert_allclose(m.H[t], m.H[t].T, atol=1e-14)
        assert np.abs(m.R[t]).max() <= 1.0 + 1e-10
        assert np.linalg.eigvalsh(m.H[t]).min() > -1e-12


def test_dcc_single_asset():
    rng = np.random.default_rng(12)
    m = DCC(n_jobs=1).fit(rng.standard_normal(200) * 0.01)
    assert m.H.shape == (200, 1, 1)
    assert np.all(np.isfinite(m.H))


def test_cl_no_silent_stall_at_large_n():
    # Regression: with the raw summed CL objective (|f| ~ 3e5 here), SLSQP
    # returned x0=(0.05, 0.90) unchanged after 3 function evaluations while
    # reporting success. The data below has constant correlation and no
    # dynamics, so a genuine fit must move `a` well off the 0.05 start.
    rng = np.random.default_rng(21)
    common = rng.standard_normal((4000, 1))
    idio = rng.standard_normal((4000, 500))
    returns = (np.sqrt(0.6) * common + np.sqrt(0.4) * idio) * 0.01
    m = DCC(method="cl", low_memory=True).fit(returns)
    assert m.a < 0.04


def test_dcc_diagnostics():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=14)
    m = DCC(n_jobs=1).fit(returns)
    assert np.isfinite(m.loglik_)
    k = 3 * 3 + 2  # univariate params + (a, b)
    assert m.aic == pytest.approx(2 * k - 2 * m.loglik_)
    assert m.bic == pytest.approx(k * np.log(400) - 2 * m.loglik_)
    se = m.std_err
    assert se.shape == (2,)
    assert np.all((se > 0) | np.isnan(se))
    s = m.summary()
    assert "DCC" in s and "method" in s


def test_dcc_loglik_consistent_across_methods():
    # loglik_ always reports the full-likelihood value, so CL and full fits
    # on the same data should report similar (not identical) values
    returns = simulate_dcc_returns(800, 3, a=0.06, b=0.90, seed=15)
    full = DCC(n_jobs=1, method="full").fit(returns)
    cl = DCC(n_jobs=1, method="cl").fit(returns)
    assert abs(full.loglik_ - cl.loglik_) / abs(full.loglik_) < 0.01


def test_dcc_forecast_psd():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=13)
    m = DCC(n_jobs=1).fit(returns)
    f = m.forecast(10)
    assert f.shape == (10, 3, 3)
    for h in range(10):
        assert np.linalg.eigvalsh(f[h]).min() > 0


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_parallel_fit_matches_serial(n_jobs):
    # Threads are only used on a free-threaded build; results must not
    # depend on n_jobs or on column-to-worker blocking either way.
    returns = np.random.default_rng(31).standard_normal((300, 7)) * 0.01
    serial = DCC(n_jobs=1).fit(returns)
    par = DCC(n_jobs=n_jobs).fit(returns)
    for g_s, g_p in zip(serial.garch_models, par.garch_models, strict=True):
        assert g_p.omega == g_s.omega
        np.testing.assert_array_equal(g_p.sigma2, g_s.sigma2)
    assert (par.a, par.b) == (serial.a, serial.b)


def test_dcc_passes_gjr_t_spec_to_univariate_models():
    returns = simulate_dcc_returns(800, 4, a=0.05, b=0.90, seed=60)
    m = DCC(n_jobs=1, o=1, dist="t").fit(returns)
    assert all(g.o == 1 and g.dist == "t" and g.nu is not None for g in m.garch_models)
    assert np.isfinite(m.loglik_)


def simulate_adcc(T, n, a, b, g, rho=0.4, seed=0):
    """ADCC returns (unit-variance margins scaled by 0.01) with N_bar from a pilot draw."""
    rng = np.random.default_rng(seed)
    Q_bar = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    pilot = np.minimum(rng.multivariate_normal(np.zeros(n), Q_bar, 100_000), 0.0)
    N_bar = pilot.T @ pilot / len(pilot)
    Q = Q_bar.copy()
    e = np.empty((T, n))
    for t in range(T):
        d = 1 / np.sqrt(np.diag(Q))
        e[t] = np.linalg.cholesky(Q * np.outer(d, d)) @ rng.standard_normal(n)
        ne = np.minimum(e[t], 0.0)
        Q = (
            (1 - a - b) * Q_bar
            - g * N_bar
            + a * np.outer(e[t], e[t])
            + g * np.outer(ne, ne)
            + b * Q
        )
    return 0.01 * e


@pytest.fixture(scope="module")
def adcc_returns():
    return simulate_adcc(3000, 4, a=0.03, b=0.90, g=0.06, seed=90)


@pytest.mark.parametrize("method", ["full", "cl"])
def test_adcc_recovers_asymmetry(adcc_returns, method):
    m = DCC(n_jobs=1, method=method, asymmetric=True).fit(adcc_returns)
    assert m.converged
    assert abs(m.g - 0.06) < 0.03
    assert abs(m.b - 0.90) < 0.05
    assert m.a + m.b + m._delta * m.g < 0.999
    assert len(m.std_err) == 3


def test_adcc_preferred_by_aic_and_nests_dcc(adcc_returns):
    sym = DCC(n_jobs=1, method="full").fit(adcc_returns)
    asym = DCC(n_jobs=1, method="full", asymmetric=True).fit(adcc_returns)
    assert asym.aic < sym.aic
    assert asym.loglik_ >= sym.loglik_ - 1e-6
    assert sym.g == 0.0
    assert "ADCC" in asym.summary() and "g:" in asym.summary()


def test_adcc_n_bar_is_the_negative_second_moment(adcc_returns):
    m = DCC(n_jobs=1, asymmetric=True).fit(adcc_returns)
    neg = np.minimum(m._std_resid, 0.0)
    np.testing.assert_allclose(m.N_bar, neg.T @ neg / len(neg), rtol=1e-12)
    Qh = np.linalg.cholesky(m.Q_bar)
    M = np.linalg.solve(Qh, np.linalg.solve(Qh, m.N_bar).T)
    assert m._delta == pytest.approx(np.linalg.eigvalsh(M).max(), rel=1e-9)


def test_adcc_first_forecast_step_uses_last_negative_shock(adcc_returns):
    m = DCC(n_jobs=1, method="full", asymmetric=True).fit(adcc_returns)
    e = m._std_resid[-1]
    ne = np.minimum(e, 0.0)
    Q = (
        (1 - m.a - m.b) * m.Q_bar
        - m.g * m.N_bar
        + m.a * np.outer(e, e)
        + m.g * np.outer(ne, ne)
        + m.b * m.Q_last
    )
    d = 1 / np.sqrt(np.diag(Q))
    np.testing.assert_allclose(m._forecast_parts(1)[1][0], Q * np.outer(d, d), rtol=1e-12)


def test_adcc_low_memory_matches_path_and_refilter(adcc_returns):
    full = DCC(n_jobs=1, method="full", asymmetric=True).fit(adcc_returns)
    lm = DCC(n_jobs=1, method="full", asymmetric=True, low_memory=True).fit(adcc_returns)
    np.testing.assert_allclose(lm.H, full.H[-1], rtol=1e-10)
    np.testing.assert_allclose(
        full._refilter(adcc_returns).forecast(3), full.forecast(3), rtol=1e-10
    )
    var = lm.portfolio_value_at_risk(np.full(4, 0.25), 0.99)
    assert var.shape == (1,) and var[0] > 0


def test_adcc_warns_when_stationarity_bound_binds():
    # weakly correlated assets: delta is large, so g hits the bound
    r = simulate_adcc(1500, 40, a=0.03, b=0.90, g=0.06, rho=0.05, seed=91)
    with pytest.warns(RuntimeWarning, match="ADCC stationarity bound is binding"):
        m = DCC(n_jobs=1, method="cl", asymmetric=True, low_memory=True).fit(r)
    assert m.a + m.b + m._delta * m.g == pytest.approx(0.999, abs=1e-6)
