"""GARCH and DCC-GARCH model implementations."""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import minimize

from multigarch._jit import (
    dcc_covariance_loop,
    dcc_final_covariance,
    dcc_loglik_loop,
    garch_loglik,
    garch_variance_loop,
)


def _validate_returns(returns: NDArray[np.float64], min_obs: int) -> NDArray[np.float64]:
    """Validate a returns array: finite values and enough observations."""
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must contain only finite values (no NaN/inf)")
    if returns.shape[0] < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {returns.shape[0]}")
    return returns


def _check_fitted(is_fitted: bool) -> None:
    if not is_fitted:
        raise ValueError("Model not fitted; call fit() first")


def _fit_single_garch(returns: NDArray, p: int, q: int) -> GARCH:
    """Fit a single GARCH model (helper for parallel fitting)."""
    return GARCH(p=p, q=q).fit(returns)


class GARCH:
    """Univariate GARCH(p,q) model with JIT-accelerated fitting.

    Model: σ²_t = ω + Σᵢ αᵢ * ε²_{t-i} + Σⱼ βⱼ * σ²_{t-j}

    Parameters after fitting:
        omega: long-run variance weight
        alpha: ARCH coefficients (array of length p)
        beta: GARCH coefficients (array of length q)
    """

    def __init__(self, p: int = 1, q: int = 1) -> None:
        """Initialize GARCH(p,q) model.

        Args:
            p: ARCH order (number of lagged squared residuals)
            q: GARCH order (number of lagged variances)
        """
        if p < 1 or q < 1:
            raise ValueError("p and q must be >= 1")
        self.p = p
        self.q = q
        self.omega: float | None = None
        self.alpha: NDArray[np.float64] | None = None
        self.beta: NDArray[np.float64] | None = None
        self.resid: NDArray[np.float64] | None = None
        self.sigma2: NDArray[np.float64] | None = None
        self.converged: bool = False
        self._nll: float | None = None
        self._var0: float | None = None
        self._x_opt: NDArray[np.float64] | None = None
        self._T: int = 0

    def fit(self, returns: NDArray[np.float64]) -> GARCH:
        """Fit GARCH(p,q) to return series.

        Args:
            returns: 1D array of returns

        Returns:
            self for chaining
        """
        returns = np.asarray(returns, dtype=np.float64).flatten()
        _validate_returns(returns, min_obs=max(10, self.p + self.q + 2))
        T = len(returns)
        var0 = float(np.var(returns))
        p, q = self.p, self.q

        def objective(params: NDArray[np.float64]) -> float:
            return garch_loglik(returns, params[0], params[1 : 1 + p], params[1 + p :], var0)

        # Initial values
        alpha0 = np.full(p, 0.05 / p)
        beta0 = np.full(q, 0.90 / q)
        x0 = np.concatenate([[var0 * 0.05], alpha0, beta0])

        bounds = [(1e-12, None)] + [(0.0, 0.999)] * (p + q)
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - np.sum(x[1:])}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500},
        )
        if not result.success:
            warnings.warn(
                f"GARCH optimizer did not converge: {result.message}", RuntimeWarning, stacklevel=2
            )
        self.converged = bool(result.success)

        self.omega = float(result.x[0])
        self.alpha = result.x[1 : 1 + p].copy()
        self.beta = result.x[1 + p :].copy()
        self._nll = float(result.fun)
        self._var0 = var0
        self._x_opt = result.x.copy()
        self._T = T

        self.resid = returns.copy()
        self.sigma2 = garch_variance_loop(returns, self.omega, self.alpha, self.beta, var0)

        return self

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast conditional variance.

        Args:
            horizon: number of periods ahead

        Returns:
            Array of forecasted variances
        """
        _check_fitted(self.sigma2 is not None)

        p, q = self.p, self.q

        # Build history arrays for forecasting
        resid2_hist = list(self.resid[-p:] ** 2)
        sigma2_hist = list(self.sigma2[-q:])

        forecasts = np.zeros(horizon)

        for h in range(horizon):
            arch_term = sum(self.alpha[i] * resid2_hist[-(i + 1)] for i in range(p))
            garch_term = sum(self.beta[j] * sigma2_hist[-(j + 1)] for j in range(q))
            forecast_h = self.omega + arch_term + garch_term
            forecasts[h] = forecast_h

            # For future periods, E[ε²] = σ²
            resid2_hist.append(forecast_h)
            sigma2_hist.append(forecast_h)

        return forecasts


class CCC:
    """Constant Conditional Correlation GARCH model.

    Assumes correlation is constant over time, only variances are dynamic.
    Much faster than DCC for many assets.

    H_t = D_t * R * D_t

    where D_t = diag(σ_{1,t}, ..., σ_{n,t}) and R is constant.
    """

    def __init__(
        self, p: int = 1, q: int = 1, n_jobs: int = -1, low_memory: bool = False
    ) -> None:
        """Initialize CCC-GARCH model.

        Args:
            p: ARCH order for univariate GARCH models
            q: GARCH order for univariate GARCH models
            n_jobs: Number of parallel jobs for GARCH fitting (-1 = all cores)
            low_memory: If True, only store final covariance matrix (not full path)
        """
        self.p = p
        self.q = q
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.garch_models: list[GARCH] = []
        self.R: NDArray[np.float64] | None = None  # Constant correlation
        self.H: NDArray[np.float64] | None = None  # Covariances (final or path)
        self._n_assets: int = 0
        self._T: int = 0

    def fit(self, returns: NDArray[np.float64]) -> CCC:
        """Fit CCC-GARCH model.

        Args:
            returns: 2D array of shape (T, n_assets)

        Returns:
            self for chaining
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        _validate_returns(returns, min_obs=max(10, self.p + self.q + 2))

        T, n = returns.shape
        self._T = T
        self._n_assets = n

        # Fit univariate GARCH models in parallel
        self.garch_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_garch)(returns[:, i], self.p, self.q) for i in range(n)
        )

        # Compute standardized residuals
        std_resid = np.zeros((T, n))
        sigmas = np.zeros((T, n))
        for i, garch in enumerate(self.garch_models):
            sigmas[:, i] = np.sqrt(garch.sigma2)
            std_resid[:, i] = garch.resid / sigmas[:, i]

        # Constant correlation from standardized residuals
        self.R = np.corrcoef(std_resid.T)
        if n == 1:
            self.R = np.array([[1.0]])

        if self.low_memory:
            # Only store final covariance matrix
            D = np.diag(sigmas[-1])
            self.H = D @ self.R @ D  # shape: (n, n)
        else:
            # Store full covariance path
            self.H = np.zeros((T, n, n))
            for t in range(T):
                D = np.diag(sigmas[t])
                self.H[t] = D @ self.R @ D

        return self

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast covariance matrices.

        Args:
            horizon: number of periods ahead

        Returns:
            Array of shape (horizon, n_assets, n_assets) of forecasted covariances
        """
        _check_fitted(self.H is not None)

        n = self._n_assets
        forecasts = np.zeros((horizon, n, n))

        # Forecast individual variances
        var_forecasts = np.zeros((horizon, n))
        for i, garch in enumerate(self.garch_models):
            var_forecasts[:, i] = garch.forecast(horizon)

        # Correlation is constant
        for h in range(horizon):
            D = np.diag(np.sqrt(var_forecasts[h]))
            forecasts[h] = D @ self.R @ D

        return forecasts


class DCC:
    """Dynamic Conditional Correlation GARCH model with JIT acceleration.

    Two-step estimation:
        1. Fit univariate GARCH(p,q) to each series (parallel)
        2. Estimate DCC parameters for correlation dynamics (JIT-accelerated)

    DCC model: Q_t = (1-a-b)*Q̄ + a*ε_{t-1}*ε'_{t-1} + b*Q_{t-1}
               R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
    """

    def __init__(
        self, p: int = 1, q: int = 1, n_jobs: int = -1, low_memory: bool = False
    ) -> None:
        """Initialize DCC-GARCH model.

        Args:
            p: ARCH order for univariate GARCH models
            q: GARCH order for univariate GARCH models
            n_jobs: Number of parallel jobs for GARCH fitting (-1 = all cores)
            low_memory: If True, only store final covariance/correlation matrices
        """
        self.p = p
        self.q = q
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.garch_models: list[GARCH] = []
        self.a: float | None = None
        self.b: float | None = None
        self.Q_bar: NDArray[np.float64] | None = None
        self.R: NDArray[np.float64] | None = None  # (n,n) if low_memory else (T,n,n)
        self.H: NDArray[np.float64] | None = None  # (n,n) if low_memory else (T,n,n)
        self.Q_last: NDArray[np.float64] | None = None  # Final Q_t for forecasting
        self._n_assets: int = 0
        self._T: int = 0

    def fit(self, returns: NDArray[np.float64]) -> DCC:
        """Fit DCC-GARCH model.

        Args:
            returns: 2D array of shape (T, n_assets)

        Returns:
            self for chaining
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        _validate_returns(returns, min_obs=max(10, self.p + self.q + 2))

        T, n = returns.shape
        self._T = T
        self._n_assets = n

        # Step 1: Fit univariate GARCH models in parallel
        self.garch_models = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_garch)(returns[:, i], self.p, self.q) for i in range(n)
        )

        # Compute standardized residuals
        std_resid = np.zeros((T, n), dtype=np.float64)
        sigmas = np.zeros((T, n), dtype=np.float64)
        for i, garch in enumerate(self.garch_models):
            sigmas[:, i] = np.sqrt(garch.sigma2)
            std_resid[:, i] = garch.resid / sigmas[:, i]

        # Step 2: Estimate DCC parameters
        self.Q_bar = np.corrcoef(std_resid.T).astype(np.float64)
        if n == 1:
            self.Q_bar = np.array([[1.0]], dtype=np.float64)

        def neg_log_likelihood(params: NDArray[np.float64]) -> float:
            return dcc_loglik_loop(std_resid, self.Q_bar, params[0], params[1])

        x0 = np.array([0.05, 0.90])
        bounds = [(1e-6, 0.499), (1e-6, 0.998)]

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
        self.a, self.b = result.x

        # Compute covariance matrices
        if self.low_memory:
            self.R, self.H, self.Q_last = dcc_final_covariance(
                std_resid, sigmas, self.Q_bar, self.a, self.b
            )
        else:
            self.R, self.H, self.Q_last = dcc_covariance_loop(
                std_resid, sigmas, self.Q_bar, self.a, self.b
            )

        return self

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast covariance matrices.

        Args:
            horizon: number of periods ahead

        Returns:
            Array of shape (horizon, n_assets, n_assets) of forecasted covariances
        """
        _check_fitted(self.H is not None)

        if self.Q_bar is None or self.Q_last is None:
            raise ValueError("Missing DCC state for forecasting")

        n = self._n_assets
        forecasts = np.zeros((horizon, n, n))

        var_forecasts = np.zeros((horizon, n))
        for i, garch in enumerate(self.garch_models):
            var_forecasts[:, i] = garch.forecast(horizon)

        # Forecast Q_t using expected shock term E[εε'] = Q̄
        Q_forecast = self.Q_last.copy()
        one_minus_b = 1.0 - self.b

        for h in range(horizon):
            Q_forecast = one_minus_b * self.Q_bar + self.b * Q_forecast

            # Normalize to correlation
            Q_diag = np.diag(Q_forecast)
            inv_sqrt = np.diag(1.0 / np.sqrt(Q_diag))
            R_forecast = inv_sqrt @ Q_forecast @ inv_sqrt

            D = np.diag(np.sqrt(var_forecasts[h]))
            forecasts[h] = D @ R_forecast @ D

        return forecasts
