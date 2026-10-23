import numpy as np
import pytest

from multigarch import GARCH


def simulate_garch(T, omega, alpha, beta, seed):
    """Simulate a GARCH(1,1) series with Gaussian innovations."""
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(T)
    r = np.empty(T)
    sigma2[0] = omega / (1.0 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, T):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return r


def test_recovers_parameters():
    r = simulate_garch(5000, omega=0.05, alpha=0.08, beta=0.88, seed=42)
    m = GARCH().fit(r)
    assert abs(m.alpha[0] - 0.08) < 0.04
    assert abs(m.beta[0] - 0.88) < 0.06
    assert m.converged


def test_persistence_constraint_respected():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(500)  # white noise: persistence ill-identified
    m = GARCH().fit(r)
    assert m.alpha.sum() + m.beta.sum() <= 0.999 + 1e-8
    assert m.omega > 0
    assert np.all(m.alpha >= 0) and np.all(m.beta >= 0)


def test_forecast_converges_to_unconditional_variance():
    r = simulate_garch(3000, omega=0.05, alpha=0.05, beta=0.90, seed=7)
    m = GARCH().fit(r)
    f = m.forecast(500)
    uncond = m.omega / (1.0 - m.alpha.sum() - m.beta.sum())
    assert abs(f[-1] - uncond) / uncond < 0.05


def test_sigma2_positive():
    r = simulate_garch(1000, omega=0.05, alpha=0.05, beta=0.90, seed=9)
    m = GARCH(p=2, q=2).fit(r)
    assert np.all(m.sigma2 > 0)


def test_mean_constant_recovers_mu():
    r = simulate_garch(4000, omega=0.05, alpha=0.05, beta=0.90, seed=3) + 1.5
    m = GARCH(mean="constant").fit(r)
    assert abs(m.mu - 1.5) < 0.1
    # residuals are demeaned
    assert abs(m.resid.mean()) < 1e-10


def test_mean_zero_default():
    r = simulate_garch(1000, omega=0.05, alpha=0.05, beta=0.90, seed=4)
    m = GARCH().fit(r)
    assert m.mu == 0.0


def test_invalid_mean_raises():
    with pytest.raises(ValueError, match="mean"):
        GARCH(mean="arma")


def test_diagnostics():
    r = simulate_garch(1000, omega=0.05, alpha=0.05, beta=0.90, seed=11)
    m = GARCH().fit(r)
    assert np.isfinite(m.loglik_)
    k = 3  # omega, alpha, beta
    assert m.aic == pytest.approx(2 * k - 2 * m.loglik_)
    assert m.bic == pytest.approx(k * np.log(1000) - 2 * m.loglik_)


def test_std_err():
    r = simulate_garch(2000, omega=0.05, alpha=0.08, beta=0.88, seed=12)
    m = GARCH().fit(r)
    se = m.std_err
    assert se.shape == (3,)
    assert np.all((se > 0) | np.isnan(se))


def test_summary():
    r = simulate_garch(1000, omega=0.05, alpha=0.05, beta=0.90, seed=13)
    m = GARCH().fit(r)
    s = m.summary()
    assert "GARCH(1, 1)" in s
    assert "loglik" in s


def test_fit_is_scale_invariant():
    # Daily-return scale (omega ~ 1e-6) used to stall SLSQP far from the
    # optimum; fitting at unit scale and at 1e-2 scale must agree.
    r = simulate_garch(2000, omega=0.05, alpha=0.08, beta=0.90, seed=21)
    big = GARCH().fit(r)
    small = GARCH().fit(r * 0.01)
    np.testing.assert_allclose(small.alpha, big.alpha, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(small.beta, big.beta, rtol=1e-4, atol=1e-6)
    assert small.omega == pytest.approx(big.omega * 1e-4, rel=1e-3)
    assert small.loglik_ == pytest.approx(big.loglik_ - 2000 * np.log(0.01), abs=1e-6)
    np.testing.assert_allclose(small.sigma2, big.sigma2 * 1e-4, rtol=1e-3)
    assert small.std_err[0] == pytest.approx(big.std_err[0] * 1e-4, rel=1e-2)


def test_zero_variance_raises():
    with pytest.raises(ValueError, match="zero variance"):
        GARCH().fit(np.zeros(100))


@pytest.mark.parametrize(
    "spec",
    [
        (1, 0, 1, "normal"),
        (2, 0, 1, "normal"),
        (1, 0, 2, "normal"),
        (2, 0, 2, "normal"),
        (1, 1, 1, "normal"),
        (1, 0, 1, "t"),
        (1, 1, 1, "t"),
        (2, 1, 2, "t"),
    ],
)
def test_numba_fit_not_worse_than_slsqp(spec):
    from multigarch.models import _garch_slsqp, _garch_starts

    p, o, q, dist = spec
    r = simulate_garch(1500, omega=1e-6, alpha=0.07, beta=0.90, seed=40 + p + 2 * q + 3 * o)
    m = GARCH(p, q, o=o, dist=dist).fit(r)
    assert m.converged
    x0 = _garch_starts(p, o, q, dist)[0][0]
    ref = _garch_slsqp(m._z, m._var0, p, o, q, dist, x0)
    assert m._nll <= ref.fun * len(r) + len(r) * np.log(m._scale) + 1e-3
    assert m.persistence < 0.999
    assert np.all(m.sigma2 > 0)


def test_slsqp_fallback_when_numba_fit_does_not_converge(monkeypatch):
    import multigarch.models as models

    monkeypatch.setattr(models, "_FIT_MAXITER", 1)
    r = simulate_garch(1000, omega=0.05, alpha=0.08, beta=0.88, seed=41)
    m = GARCH().fit(r)
    monkeypatch.undo()
    ref = GARCH().fit(r)
    assert m.converged
    assert m._nll == pytest.approx(ref._nll, abs=1e-3)


@pytest.mark.parametrize(("n_jobs", "expected"), [(1, 1), (2, 2), (-1, None), (10_000, None)])
def test_resolve_n_jobs(n_jobs, expected):
    import numba

    from multigarch.models import _resolve_n_jobs

    max_threads = numba.config.NUMBA_NUM_THREADS
    assert _resolve_n_jobs(n_jobs) == (expected or max_threads) if max_threads >= 2 else 1
    with pytest.raises(ValueError):
        _resolve_n_jobs(0)


def simulate_gjr_t(T, omega, alpha, gamma, beta, nu, seed):
    """Simulate GJR-GARCH(1,1,1) with unit-variance Student-t innovations."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, T) * np.sqrt((nu - 2.0) / nu)
    r = np.empty(T)
    s2 = omega / (1.0 - alpha - beta - 0.5 * gamma)
    for t in range(T):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + gamma * r[t] ** 2 * (r[t] < 0) + beta * s2
    return r


def test_gjr_t_recovers_parameters():
    r = simulate_gjr_t(6000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, nu=6.0, seed=50)
    m = GARCH(o=1, dist="t").fit(r)
    assert m.converged
    assert abs(m.gamma[0] - 0.10) < 0.04
    assert abs(m.beta[0] - 0.88) < 0.04
    assert 4.5 < m.nu < 8.0
    assert len(m.std_err) == 5
    assert np.all(np.isfinite(m.std_err))


def test_gjr_t_preferred_by_aic_on_gjr_t_data():
    r = simulate_gjr_t(4000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, nu=6.0, seed=51)
    aic = {kw: GARCH(**dict(kw)).fit(r).aic for kw in [(), (("o", 1),), (("o", 1), ("dist", "t"))]}
    assert aic[(("o", 1), ("dist", "t"))] < aic[(("o", 1),)] < aic[()]


def test_gjr_forecast_converges_to_unconditional_variance():
    r = simulate_gjr_t(3000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, nu=6.0, seed=52)
    m = GARCH(o=1, dist="t").fit(r)
    f = m.forecast(3000)
    assert f[-1] == pytest.approx(m.omega / (1.0 - m.persistence), rel=1e-6)


def test_plain_garch_has_empty_gamma_and_no_nu():
    m = GARCH().fit(simulate_garch(500, omega=0.05, alpha=0.05, beta=0.90, seed=53))
    assert m.gamma.shape == (0,)
    assert m.nu is None


@pytest.mark.parametrize(("kw", "match"), [({"o": -1}, "o must"), ({"dist": "skewt"}, "dist must")])
def test_invalid_spec_raises(kw, match):
    with pytest.raises(ValueError, match=match):
        GARCH(**kw)
