"""Value at risk and expected shortfall."""

import numpy as np
import pytest
from scipy import integrate, stats

from multigarch import CCC, DCC, GARCH


def simulate_gjr_t(T, n, seed, nu=6.0):
    """n independent-ish GJR-t series with a common factor for correlation."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(T)
    out = np.empty((T, n))
    for i in range(n):
        z = 0.6 * f + 0.8 * rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
        s2 = 1e-4
        for t in range(T):
            out[t, i] = np.sqrt(s2) * z[t]
            s2 = 1e-6 + 0.03 * out[t, i] ** 2 + 0.1 * out[t, i] ** 2 * (out[t, i] < 0) + 0.87 * s2
    return out


@pytest.fixture(scope="module")
def returns():
    return simulate_gjr_t(1500, 3, seed=70)


def test_normal_var_and_es_closed_form(returns):
    m = GARCH(mean="constant").fit(returns[:, 0])
    sigma = np.sqrt(m.forecast(3))
    z = stats.norm.ppf(0.01)
    np.testing.assert_allclose(m.value_at_risk(0.99, 3), -(m.mu + sigma * z), rtol=1e-12)
    es = sigma * stats.norm.pdf(stats.norm.ppf(0.025)) / 0.025 - m.mu
    np.testing.assert_allclose(m.expected_shortfall(0.975, 3), es, rtol=1e-12)


def test_t_var_and_es_match_scipy_distribution(returns):
    m = GARCH(o=1, dist="t").fit(returns[:, 0])
    sigma = float(np.sqrt(m.forecast(1))[0])
    scale = sigma * np.sqrt((m.nu - 2) / m.nu)  # unit-variance t scaled to sigma
    dist = stats.t(df=m.nu, scale=scale)
    assert m.value_at_risk(0.99)[0] == pytest.approx(-dist.ppf(0.01), rel=1e-10)
    q = dist.ppf(0.025)
    es = -integrate.quad(lambda x: x * dist.pdf(x), -np.inf, q)[0] / 0.025
    assert m.expected_shortfall(0.975)[0] == pytest.approx(es, rel=1e-7)


def test_var_increases_with_level(returns):
    m = GARCH(dist="t").fit(returns[:, 0])
    assert m.value_at_risk(0.95)[0] < m.value_at_risk(0.99)[0] < m.value_at_risk(0.999)[0]
    assert m.value_at_risk(0.99)[0] < m.expected_shortfall(0.99)[0]


@pytest.mark.parametrize("level", [0.0, 1.0, 1.5])
def test_invalid_level_raises(returns, level):
    m = GARCH().fit(returns[:, 0])
    with pytest.raises(ValueError, match="level"):
        m.value_at_risk(level)


def test_dcc_first_forecast_step_uses_last_shock(returns):
    m = DCC(n_jobs=1, method="full").fit(returns)
    e = m._std_resid[-1]
    Q = (1 - m.a - m.b) * m.Q_bar + m.a * np.outer(e, e) + m.b * m.Q_last
    d = 1 / np.sqrt(np.diag(Q))
    _, corr_f = m._forecast_parts(1)
    np.testing.assert_allclose(corr_f[0], Q * np.outer(d, d), rtol=1e-12)


def test_dcc_long_horizon_correlation_reverts_to_target(returns):
    m = DCC(n_jobs=1, method="full").fit(returns)
    _, corr_f = m._forecast_parts(20_000)
    d = 1 / np.sqrt(np.diag(m.Q_bar))
    np.testing.assert_allclose(corr_f[-1], m.Q_bar * np.outer(d, d), atol=1e-8)


@pytest.mark.parametrize("model_cls", [CCC, DCC])
def test_single_asset_portfolio_matches_univariate(returns, model_cls):
    for dist in ("normal", "t"):
        m = model_cls(n_jobs=1, dist=dist).fit(returns[:, :1])
        g = m.garch_models[0]
        port = m.portfolio_value_at_risk([1.0], 0.99, horizon=2, n_sims=400_000, seed=1)
        rtol = 1e-10 if dist == "normal" else 0.02
        np.testing.assert_allclose(port, g.value_at_risk(0.99, 2), rtol=rtol)


def test_normal_portfolio_var_matches_simulation(returns):
    m = DCC(n_jobs=1).fit(returns)
    w = np.array([0.5, 0.3, 0.2])
    H = m.forecast(1)[0]
    loss = -(np.random.default_rng(2).multivariate_normal(np.zeros(3), H, 400_000) @ w)
    assert m.portfolio_value_at_risk(w, 0.99)[0] == pytest.approx(np.quantile(loss, 0.99), rel=0.02)
    es = loss[loss >= np.quantile(loss, 0.975)].mean()
    assert m.portfolio_expected_shortfall(w, 0.975)[0] == pytest.approx(es, rel=0.02)


def test_t_margins_fatten_the_portfolio_tail(returns):
    w = np.array([0.5, 0.3, 0.2])
    normal = DCC(n_jobs=1).fit(returns).portfolio_value_at_risk(w, 0.995)[0]
    heavy = DCC(n_jobs=1, dist="t").fit(returns).portfolio_value_at_risk(w, 0.995, seed=3)[0]
    assert heavy > normal


def test_portfolio_weights_shape_checked(returns):
    m = CCC(n_jobs=1).fit(returns)
    with pytest.raises(ValueError, match="weights"):
        m.portfolio_value_at_risk([0.5, 0.5])
