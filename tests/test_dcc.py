import numpy as np

from multigarch import DCC


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
