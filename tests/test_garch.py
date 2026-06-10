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
