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
