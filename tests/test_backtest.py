"""Rolling VaR backtests and coverage tests."""

import numpy as np
import pytest

from multigarch import DCC, GARCH
from multigarch.backtest import christoffersen_test, kupiec_test, rolling_var


def simulate_gjr_t(T, seed, nu=5.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
    r = np.empty(T)
    s2 = 1e-4
    for t in range(T):
        r[t] = np.sqrt(s2) * z[t]
        s2 = 1e-6 + 0.03 * r[t] ** 2 + 0.1 * r[t] ** 2 * (r[t] < 0) + 0.87 * s2
    return r


def test_kupiec_matches_explicit_likelihood_ratio():
    hits = np.zeros(250, dtype=bool)
    hits[[10, 50, 90, 130, 170, 210, 240]] = True  # 7 hits, 2.8% vs 1% expected
    n, x, p = 250, 7, 0.01
    lr = -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n)
    )
    res = kupiec_test(hits, 0.99)
    assert res.statistic == pytest.approx(lr, rel=1e-12)
    assert res.pvalue < 0.05


def test_kupiec_zero_hits_is_finite():
    res = kupiec_test(np.zeros(500, dtype=bool), 0.99)
    assert res.statistic == pytest.approx(-2 * 500 * np.log(0.99))
    assert 0.0 < res.pvalue < 1.0


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=bool)
    clustered[[100, 101, 102, 500, 501, 502, 900, 901, 902, 903]] = True
    spread = np.zeros(1000, dtype=bool)
    spread[np.arange(50, 1000, 100)] = True  # same 10 hits, evenly spaced
    ind_c, _ = christoffersen_test(clustered, 0.99)
    ind_s, cc_s = christoffersen_test(spread, 0.99)
    assert ind_c.pvalue < 0.001
    assert ind_s.pvalue > 0.3
    assert cc_s.statistic == pytest.approx(kupiec_test(spread, 0.99).statistic + ind_s.statistic)


def test_correct_model_passes_and_normal_model_fails_on_fat_tails():
    r = simulate_gjr_t(4000, seed=80)
    good = rolling_var(r, GARCH(o=1, dist="t"), window=1000, level=0.99, refit_every=50)
    bad = rolling_var(r, GARCH(), window=1000, level=0.99, refit_every=50)
    assert good.var.shape == (3000,)
    assert good.kupiec().pvalue > 0.05
    assert bad.hit_rate > good.hit_rate
    assert np.all(good.var > 0)


def test_refit_every_matches_full_refit_at_refit_steps():
    r = simulate_gjr_t(900, seed=81)
    every = rolling_var(r, GARCH(o=1), window=600, refit_every=1)
    sparse = rolling_var(r, GARCH(o=1), window=600, refit_every=25)
    steps = np.arange(0, 300, 25)
    np.testing.assert_allclose(sparse.var[steps], every.var[steps], rtol=1e-12)
    # between refits the forecasts differ but stay close (same data, stale parameters)
    np.testing.assert_allclose(sparse.var, every.var, rtol=0.25)


def test_refilter_with_same_data_reproduces_forecast():
    r = simulate_gjr_t(800, seed=82)
    m = GARCH(o=1, dist="t").fit(r)
    np.testing.assert_allclose(
        m._refilter(r).value_at_risk(0.99, 5), m.value_at_risk(0.99, 5), rtol=1e-12
    )


def test_portfolio_backtest_runs_and_template_is_untouched():
    R = np.column_stack([simulate_gjr_t(700, seed=s) for s in (84, 85, 86)])
    R[:, 1] += 0.5 * R[:, 0]
    template = DCC(n_jobs=1, o=1)
    bt = rolling_var(
        R, template, window=500, weights=np.full(3, 1 / 3), refit_every=40, es_level=0.975
    )
    assert bt.var.shape == bt.es.shape == (200,)
    assert np.all(bt.es > bt.var * 0.8)
    assert template.H is None  # template never fitted in place
    assert "Kupiec" in bt.summary()


def test_rolling_var_argument_checks():
    r = simulate_gjr_t(300, seed=87)
    with pytest.raises(ValueError, match="1-D"):
        rolling_var(np.column_stack([r, r]), GARCH(), window=100)
    with pytest.raises(ValueError, match="weights"):
        rolling_var(np.column_stack([r, r]), DCC(), window=100)
    with pytest.raises(ValueError, match="window"):
        rolling_var(r, GARCH(), window=300)
