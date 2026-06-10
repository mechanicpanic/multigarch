"""GARCH and DCC-GARCH model implementations."""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import minimize

from multigarch._jit import (
    dcc_cl_loglik,
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


def _numerical_hessian(f, x: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
    """Central-difference Hessian of scalar function f at x."""
    k = len(x)
    H = np.zeros((k, k))
    h = eps * np.maximum(np.abs(x), 1.0)
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy()
            xpp[i] += h[i]
            xpp[j] += h[j]
            xpm = x.copy()
            xpm[i] += h[i]
            xpm[j] -= h[j]
            xmp = x.copy()
            xmp[i] -= h[i]
            xmp[j] += h[j]
            xmm = x.copy()
            xmm[i] -= h[i]
            xmm[j] -= h[j]
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h[i] * h[j])
            H[j, i] = H[i, j]
    return H


def _se_from_hessian(H: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard errors from a Hessian of the negative log-likelihood.

    Returns NaN where the inverse Hessian has non-positive diagonal.
    """
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    diag = np.diag(cov).copy()
    diag[diag <= 0] = np.nan
    return np.sqrt(diag)


def _fit_single_garch(returns: NDArray, p: int, q: int, mean: str) -> GARCH:
    """Fit a single GARCH model (helper for parallel fitting)."""
    return GARCH(p=p, q=q, mean=mean).fit(returns)


class GARCH:
    """Univariate GARCH(p,q) model with JIT-accelerated fitting.

    Model: σ²_t = ω + Σᵢ αᵢ * ε²_{t-i} + Σⱼ βⱼ * σ²_{t-j}

    Parameters after fitting:
        omega: long-run variance weight
        alpha: ARCH coefficients (array of length p)
        beta: GARCH coefficients (array of length q)
    """

    def __init__(self, p: int = 1, q: int = 1, mean: str = "zero") -> None:
        """Initialize GARCH(p,q) model.

        Args:
            p: ARCH order (number of lagged squared residuals)
            q: GARCH order (number of lagged variances)
            mean: "zero" (default) fits raw returns; "constant" demeans
                first and stores the sample mean as ``mu``
        """
        if p < 1 or q < 1:
            raise ValueError("p and q must be >= 1")
        if mean not in ("zero", "constant"):
            raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
        self.p = p
        self.q = q
        self.mean = mean
        self.mu: float = 0.0
        self._std_err: NDArray[np.float64] | None = None
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
        self._std_err = None
        self.mu = float(returns.mean()) if self.mean == "constant" else 0.0
        returns = returns - self.mu
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

    def _objective(self, x: NDArray[np.float64]) -> float:
        return garch_loglik(self.resid, x[0], x[1 : 1 + self.p], x[1 + self.p :], self._var0)

    @property
    def _n_params(self) -> int:
        return 1 + self.p + self.q + (1 if self.mean == "constant" else 0)

    @property
    def loglik_(self) -> float:
        """Gaussian log-likelihood at the optimum (includes the 2*pi constant)."""
        _check_fitted(self.sigma2 is not None)
        return -self._nll - 0.5 * self._T * np.log(2.0 * np.pi)

    @property
    def aic(self) -> float:
        return 2.0 * self._n_params - 2.0 * self.loglik_

    @property
    def bic(self) -> float:
        return self._n_params * np.log(self._T) - 2.0 * self.loglik_

    @property
    def std_err(self) -> NDArray[np.float64]:
        """Approximate standard errors for (omega, alpha..., beta...).

        Inverse numerical Hessian of the negative log-likelihood at the
        optimum. NaN where the Hessian is not positive definite. The mean
        parameter (if any) is excluded.
        """
        _check_fitted(self.sigma2 is not None)
        if self._std_err is None:
            H = _numerical_hessian(self._objective, self._x_opt)
            self._std_err = _se_from_hessian(H)
        return self._std_err

    def summary(self) -> str:
        """Formatted model summary."""
        _check_fitted(self.sigma2 is not None)
        lines = [f"GARCH({self.p}, {self.q})  T={self._T}  mean={self.mean}"]
        if self.mean == "constant":
            lines.append(f"  mu:    {self.mu:.6g}")
        lines.append(f"  omega: {self.omega:.6g}")
        for i, a in enumerate(self.alpha, 1):
            lines.append(f"  alpha[{i}]: {a:.4f}")
        for j, b in enumerate(self.beta, 1):
            lines.append(f"  beta[{j}]:  {b:.4f}")
        lines += [
            f"  persistence: {self.alpha.sum() + self.beta.sum():.4f}",
            f"  loglik: {self.loglik_:.2f}  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}",
            f"  converged: {self.converged}",
        ]
        return "\n".join(lines)

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
        self,
        p: int = 1,
        q: int = 1,
        n_jobs: int = -1,
        low_memory: bool = False,
        mean: str = "zero",
        shrinkage: float = 0.0,
    ) -> None:
        """Initialize CCC-GARCH model.

        Args:
            p: ARCH order for univariate GARCH models
            q: GARCH order for univariate GARCH models
            n_jobs: Number of parallel jobs for GARCH fitting (-1 = all cores)
            low_memory: If True, only store final covariance matrix (not full path)
            mean: mean model for the univariate fits ("zero" or "constant")
            shrinkage: shrink the correlation matrix toward identity,
                R = (1-shrinkage)*S + shrinkage*I. Use > 0 when T is not
                much larger than n; it guarantees a well-conditioned target
        """
        if mean not in ("zero", "constant"):
            raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.shrinkage = shrinkage
        self.p = p
        self.q = q
        self.mean = mean
        self.n_jobs = n_jobs
        self.low_memory = low_memory
        self.garch_models: list[GARCH] = []
        self.R: NDArray[np.float64] | None = None  # Constant correlation
        self.H: NDArray[np.float64] | None = None  # Covariances (final or path)
        self._n_assets: int = 0
        self._T: int = 0
        self._std_resid: NDArray[np.float64] | None = None
        self._loglik: float | None = None

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
        self.garch_models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_single_garch)(returns[:, i], self.p, self.q, self.mean)
            for i in range(n)
        )

        # Compute standardized residuals
        std_resid = np.zeros((T, n))
        sigmas = np.zeros((T, n))
        for i, garch in enumerate(self.garch_models):
            sigmas[:, i] = np.sqrt(garch.sigma2)
            std_resid[:, i] = garch.resid / sigmas[:, i]
        self._std_resid = std_resid
        self._loglik = None

        # Constant correlation from standardized residuals
        S = np.corrcoef(std_resid.T) if n > 1 else np.array([[1.0]])
        self.R = (1.0 - self.shrinkage) * S + self.shrinkage * np.eye(n)

        if self.low_memory:
            # Only store final covariance matrix
            s_last = sigmas[-1]
            self.H = self.R * np.outer(s_last, s_last)  # shape: (n, n)
        else:
            # Store full covariance path: H[t] = diag(s_t) @ R @ diag(s_t)
            self.H = sigmas[:, :, None] * self.R[None, :, :] * sigmas[:, None, :]

        return self

    @property
    def _n_params(self) -> int:
        """Optimizer-estimated parameters only; R is moment-estimated."""
        return sum(g._n_params for g in self.garch_models)

    @property
    def loglik_(self) -> float:
        """Two-step Gaussian quasi-log-likelihood (volatility + correlation parts)."""
        _check_fitted(self.H is not None)
        if self._loglik is None:
            try:
                L = np.linalg.cholesky(self.R)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "R is not positive definite; loglik_ is NaN (try shrinkage > 0)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._loglik = float("nan")
                return self._loglik
            sol = np.linalg.solve(L, self._std_resid.T)
            quad = float((sol**2).sum())
            logdet = 2.0 * float(np.log(np.diag(L)).sum())
            sum_ee = float((self._std_resid**2).sum())
            corr_ll = -0.5 * (self._T * logdet + quad - sum_ee)
            self._loglik = sum(g.loglik_ for g in self.garch_models) + corr_ll
        return self._loglik

    @property
    def aic(self) -> float:
        return 2.0 * self._n_params - 2.0 * self.loglik_

    @property
    def bic(self) -> float:
        return self._n_params * np.log(self._T) - 2.0 * self.loglik_

    def summary(self) -> str:
        """Formatted model summary."""
        _check_fitted(self.H is not None)
        return "\n".join(
            [
                f"CCC({self.p}, {self.q})  T={self._T}  n={self._n_assets}",
                f"  shrinkage: {self.shrinkage}",
                f"  loglik: {self.loglik_:.2f}  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}",
            ]
        )

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
            s = np.sqrt(var_forecasts[h])
            forecasts[h] = self.R * np.outer(s, s)

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
        self,
        p: int = 1,
        q: int = 1,
        n_jobs: int = -1,
        low_memory: bool = False,
        mean: str = "zero",
        method: str = "auto",
        shrinkage: float = 0.0,
    ) -> None:
        """Initialize DCC-GARCH model.

        Args:
            p: ARCH order for univariate GARCH models
            q: GARCH order for univariate GARCH models
            n_jobs: Number of parallel jobs for GARCH fitting (-1 = all cores)
            low_memory: If True, only store final covariance/correlation matrices
            mean: mean model for the univariate fits ("zero" or "constant")
            method: second-stage estimator — "full" exact likelihood
                (O(T·n³) per evaluation), "cl" pairwise composite likelihood
                (O(T·n), recommended for large n: dramatically faster and
                less biased), or "auto" (default) which picks "cl" when
                n > 25
            shrinkage: shrink the correlation target toward identity,
                Q_bar = (1-shrinkage)*S + shrinkage*I. Use > 0 when T is not
                much larger than n; it guarantees a well-conditioned target
        """
        if mean not in ("zero", "constant"):
            raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
        if method not in ("auto", "full", "cl"):
            raise ValueError(f"method must be 'auto', 'full' or 'cl', got {method!r}")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.shrinkage = shrinkage
        self.method = method
        self.method_: str | None = None
        self.converged: bool = False
        self.p = p
        self.q = q
        self.mean = mean
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
        self._std_resid: NDArray[np.float64] | None = None
        self._loglik: float | None = None
        self._std_err: NDArray[np.float64] | None = None

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
        self.garch_models = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_single_garch)(returns[:, i], self.p, self.q, self.mean)
            for i in range(n)
        )

        # Compute standardized residuals
        std_resid = np.zeros((T, n), dtype=np.float64)
        sigmas = np.zeros((T, n), dtype=np.float64)
        for i, garch in enumerate(self.garch_models):
            sigmas[:, i] = np.sqrt(garch.sigma2)
            std_resid[:, i] = garch.resid / sigmas[:, i]
        self._std_resid = std_resid
        self._loglik = None
        self._std_err = None

        # Step 2: Estimate DCC parameters
        S = np.corrcoef(std_resid.T) if n > 1 else np.array([[1.0]])
        self.Q_bar = (1.0 - self.shrinkage) * S + self.shrinkage * np.eye(n)

        method = self.method
        if method == "auto":
            method = "cl" if n > 25 else "full"
        self.method_ = method
        loglik_fn = dcc_cl_loglik if method == "cl" else dcc_loglik_loop

        def objective(params: NDArray[np.float64]) -> float:
            return loglik_fn(std_resid, self.Q_bar, params[0], params[1])

        x0 = np.array([0.05, 0.90])
        bounds = [(1e-8, 0.999), (1e-8, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[0] - x[1]}]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200},
        )
        if not result.success:
            warnings.warn(
                f"DCC optimizer did not converge: {result.message}", RuntimeWarning, stacklevel=2
            )
        self.converged = bool(result.success)
        self.a, self.b = float(result.x[0]), float(result.x[1])

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

    @property
    def _n_params(self) -> int:
        """Optimizer-estimated parameters: univariate + (a, b). Q_bar is targeted."""
        return sum(g._n_params for g in self.garch_models) + 2

    @property
    def loglik_(self) -> float:
        """Two-step Gaussian quasi-log-likelihood (volatility + correlation parts).

        Always evaluates the FULL correlation likelihood once at the fitted
        (a, b), regardless of estimation method; the correlation part
        conditions on the first observation. One O(T*n^3) pass — a few
        seconds at n=500.
        """
        _check_fitted(self.H is not None)
        if self._loglik is None:
            corr_nll = dcc_loglik_loop(self._std_resid, self.Q_bar, self.a, self.b)
            sum_ee = float((self._std_resid[1:] ** 2).sum())
            corr_ll = -corr_nll + 0.5 * sum_ee
            self._loglik = sum(g.loglik_ for g in self.garch_models) + corr_ll
        return self._loglik

    @property
    def aic(self) -> float:
        return 2.0 * self._n_params - 2.0 * self.loglik_

    @property
    def bic(self) -> float:
        return self._n_params * np.log(self._T) - 2.0 * self.loglik_

    @property
    def std_err(self) -> NDArray[np.float64]:
        """Approximate standard errors for (a, b).

        Inverse numerical Hessian of the second-stage objective actually
        used for estimation (full or composite). Both ignore first-stage
        estimation error (no sandwich correction) — treat as indicative.
        """
        _check_fitted(self.H is not None)
        if self._std_err is None:
            loglik_fn = dcc_cl_loglik if self.method_ == "cl" else dcc_loglik_loop

            def f(x: NDArray[np.float64]) -> float:
                return loglik_fn(self._std_resid, self.Q_bar, x[0], x[1])

            H = _numerical_hessian(f, np.array([self.a, self.b]))
            self._std_err = _se_from_hessian(H)
        return self._std_err

    def summary(self) -> str:
        """Formatted model summary."""
        _check_fitted(self.H is not None)
        return "\n".join(
            [
                f"DCC({self.p}, {self.q})  T={self._T}  n={self._n_assets}"
                f"  method={self.method_}",
                f"  a: {self.a:.4f}  b: {self.b:.4f}  a+b: {self.a + self.b:.4f}",
                f"  shrinkage: {self.shrinkage}",
                f"  loglik: {self.loglik_:.2f}  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}",
                f"  converged: {self.converged}",
            ]
        )

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
            d = 1.0 / np.sqrt(np.diag(Q_forecast))
            R_forecast = Q_forecast * np.outer(d, d)

            s = np.sqrt(var_forecasts[h])
            forecasts[h] = R_forecast * np.outer(s, s)

        return forecasts
