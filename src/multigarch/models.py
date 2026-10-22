"""GARCH and DCC-GARCH model implementations."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from multigarch._jit import (
    dcc_cl_loglik,
    dcc_covariance_loop,
    dcc_final_covariance,
    dcc_loglik_loop,
    garch_fit_batch,
    garch_loglik,
    garch_loglik_grad,
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


def _resolve_n_jobs(n_jobs: int) -> int:
    """Map an ``n_jobs`` value (-1 = all cores) to a numba thread count."""
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero")
    max_threads = numba.config.NUMBA_NUM_THREADS
    k = n_jobs if n_jobs > 0 else max_threads + 1 + n_jobs
    return max(1, min(k, max_threads))


@contextmanager
def _numba_threads(n_jobs: int) -> Iterator[None]:
    """Run numba parallel kernels on ``n_jobs`` threads, then restore."""
    prev = numba.get_num_threads()
    numba.set_num_threads(_resolve_n_jobs(n_jobs))
    try:
        yield
    finally:
        numba.set_num_threads(prev)


# Projected-BFGS settings for the numba GARCH fit (objective is the
# per-observation NLL; gtol bounds the projected-gradient norm, ftol the
# relative per-step improvement)
_FIT_MAXITER = 500
_FIT_GTOL = 1e-7
_FIT_FTOL = 1e-10


def _garch_starts(p: int, q: int) -> tuple[NDArray[np.float64], int]:
    """Starting points (omega, alpha, beta) in unit-variance space, one per row.

    Returns (starts, n_always). Higher-order likelihoods are often
    multimodal, so all their starts always run. GARCH(1,1) runs its
    second, low-persistence start only if the first fit ends on the
    boundary: on near-white-noise series the default start leads onto
    the flat omega-beta ridge.
    """

    def x(alpha, beta, omega=0.05):
        return np.concatenate([[omega], alpha, beta])

    starts = [x(np.full(p, 0.05 / p), np.full(q, 0.90 / q))]
    if p + q == 2:
        starts.append(x(np.array([0.10]), np.array([0.70]), omega=0.20))
        return np.array(starts), 1
    e = 1e-3
    starts += [
        x(np.r_[0.05, np.full(p - 1, e)], np.r_[0.90, np.full(q - 1, e)]),
        x(np.r_[np.full(p - 1, e), 0.05], np.r_[np.full(q - 1, e), 0.90]),
        x(np.full(p, 0.15 / p), np.full(q, 0.60 / q), omega=0.25),
        x(np.full(p, 0.30 / p), np.full(q, 0.40 / q), omega=0.30),
    ]
    return np.array(starts), len(starts)


def _garch_slsqp(z: NDArray[np.float64], var0: float, p: int, q: int, x0: NDArray[np.float64]):
    """Constrained SLSQP fit in the unit-variance parametrization (fallback)."""
    T = len(z)

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nll, grad = garch_loglik_grad(z, params, p, q, var0)
        return nll / T, grad / T

    jac_persistence = np.concatenate([[0.0], -np.ones(p + q)])
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: 0.999 - np.sum(x[1:]),
            "jac": lambda x: jac_persistence,
        }
    ]
    return minimize(
        objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=[(1e-12, None)] + [(0.0, 0.999)] * (p + q),
        constraints=constraints,
        options={"maxiter": 500},
    )


def _fit_garch_models(
    models: list[GARCH], series: list[NDArray[np.float64]], n_jobs: int
) -> list[GARCH]:
    """Fit each model to its series with one parallel numba batch.

    Series where projected BFGS does not converge are refit with SLSQP
    from its point; a RuntimeWarning is emitted only if that fails too.
    """
    m0 = models[0]
    p, q = m0.p, m0.q
    prepared = [m._prepare(r) for m, r in zip(models, series, strict=True)]
    Z = np.stack([z for _, z, _ in prepared])
    var0 = Z.var(axis=1)
    starts, n_always = _garch_starts(p, q)
    with _numba_threads(n_jobs):
        X, F, conv = garch_fit_batch(
            Z, starts, p, q, var0, _FIT_MAXITER, _FIT_GTOL, _FIT_FTOL, n_always
        )
    for i, (m, (resid, z, scale)) in enumerate(zip(models, prepared, strict=True)):
        x, f, ok = X[i], float(F[i]), bool(conv[i])
        if not ok:
            result = _garch_slsqp(z, float(var0[i]), p, q, x)
            if result.success or result.fun < f:
                x, f, ok = result.x, float(result.fun), bool(result.success)
            if not ok:
                warnings.warn(
                    f"GARCH optimizer did not converge: {result.message}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        m._finish(resid, z, scale, float(var0[i]), x, f, ok)
    return models


def _fit_garch_columns(
    returns: NDArray[np.float64], p: int, q: int, mean: str, n_jobs: int
) -> list[GARCH]:
    """Fit a univariate GARCH to every column of a (T, n) array."""
    n = returns.shape[1]
    models = [GARCH(p=p, q=q, mean=mean) for _ in range(n)]
    return _fit_garch_models(models, [returns[:, i] for i in range(n)], n_jobs)


class GARCH:
    r"""Univariate GARCH(p,q) model with JIT-accelerated fitting.

    Model:

    $$\sigma^2_t = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon^2_{t-i}
                 + \sum_{j=1}^{q} \beta_j \sigma^2_{t-j}$$

    Parameters after fitting:
        omega: long-run variance weight $\omega$
        alpha: ARCH coefficients $\alpha_1, \dots, \alpha_p$
        beta: GARCH coefficients $\beta_1, \dots, \beta_q$
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
        self._z: NDArray[np.float64] | None = None
        self._scale: float = 1.0
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
        return _fit_garch_models([self], [returns], n_jobs=1)[0]

    def _prepare(self, returns: NDArray[np.float64]):
        """Demean and rescale; returns (resid, z, scale) with z = resid / scale.

        The optimizer works on unit-variance z: with raw daily returns
        omega is ~1e-6 next to alpha, beta ~0.1, which stalls optimizers
        far from the optimum. omega scales by scale**2, alpha and beta are
        scale-free.
        """
        returns = np.asarray(returns, dtype=np.float64).flatten()
        self.mu = float(returns.mean()) if self.mean == "constant" else 0.0
        resid = returns - self.mu
        scale = float(np.std(resid))
        if scale == 0.0:
            raise ValueError("returns have zero variance")
        return resid, resid / scale, scale

    def _finish(
        self,
        resid: NDArray[np.float64],
        z: NDArray[np.float64],
        scale: float,
        var0: float,
        x: NDArray[np.float64],
        nll_per_obs: float,
        converged: bool,
    ) -> None:
        """Store a fit given optimal unit-variance params x = (omega_z, alpha, beta)."""
        p = self.p
        T = len(z)
        scale2 = scale * scale
        self._std_err = None
        self.converged = converged
        self.omega = float(x[0]) * scale2
        self.alpha = np.array(x[1 : 1 + p])
        self.beta = np.array(x[1 + p :])
        # nll(resid) = nll(z) + T * log(scale)
        self._nll = nll_per_obs * T + T * np.log(scale)
        self._z = z
        self._scale = scale
        self._var0 = var0
        self._x_opt = np.array(x)
        self._T = T
        self.resid = resid
        self.sigma2 = scale2 * garch_variance_loop(z, float(x[0]), self.alpha, self.beta, var0)

    def _objective(self, x: NDArray[np.float64]) -> float:
        """Negative log-likelihood in the unit-variance parametrization."""
        return garch_loglik(self._z, x[0], x[1 : 1 + self.p], x[1 + self.p :], self._var0)

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
            # Hessian in the unit-variance parametrization, where all
            # parameters are O(1); omega = scale**2 * omega_z
            H = _numerical_hessian(self._objective, self._x_opt)
            se = _se_from_hessian(H)
            se[0] *= self._scale**2
            self._std_err = se
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
    r"""Constant Conditional Correlation GARCH model.

    Assumes correlation is constant over time, only variances are dynamic.
    Much faster than DCC for many assets.

    $$H_t = D_t \, R \, D_t$$

    where $D_t = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{n,t})$ and the
    correlation matrix $R$ is constant.
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
            n_jobs: Number of threads for the numba fitting kernels
                (-1 = all cores); works with or without the GIL
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
        self.garch_models = _fit_garch_columns(returns, self.p, self.q, self.mean, self.n_jobs)

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
    r"""Dynamic Conditional Correlation GARCH model with JIT acceleration.

    Two-step estimation:
        1. Fit univariate GARCH(p,q) to each series (parallel)
        2. Estimate DCC parameters for correlation dynamics (JIT-accelerated)

    DCC model:

    $$Q_t = (1 - a - b)\,\bar{Q} + a\,\varepsilon_{t-1}\varepsilon_{t-1}'
          + b\,Q_{t-1}$$
    $$R_t = \mathrm{diag}(Q_t)^{-1/2} \, Q_t \, \mathrm{diag}(Q_t)^{-1/2}$$

    For large cross-sections the second stage can use a pairwise composite
    likelihood over contiguous pairs $(i, i+1)$ — set `method="cl"` (chosen
    automatically for $n > 25$): $O(Tn)$ per optimizer evaluation instead of
    $O(Tn^3)$, and less biased at large $n$.
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
            n_jobs: Number of threads for the numba fitting kernels
                (-1 = all cores); works with or without the GIL
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
        self.garch_models = _fit_garch_columns(returns, self.p, self.q, self.mean, self.n_jobs)

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

        # SLSQP degenerates silently (returns x0, reports success) when the
        # objective magnitude is large — observed at n=500, T=1258 with the
        # raw summed CL objective. Optimize the per-pair, per-step average;
        # a constant rescale leaves the optimum unchanged.
        denom = max(float((T - 1) * (n - 1 if method == "cl" else 1)), 1.0)

        def objective(params: NDArray[np.float64]) -> float:
            return loglik_fn(std_resid, self.Q_bar, params[0], params[1]) / denom

        x0 = np.array([0.05, 0.90])
        bounds = [(1e-8, 0.999), (1e-8, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[0] - x[1]}]

        with _numba_threads(self.n_jobs):
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
                    f"DCC optimizer did not converge: {result.message}",
                    RuntimeWarning,
                    stacklevel=2,
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
                f"DCC({self.p}, {self.q})  T={self._T}  n={self._n_assets}  method={self.method_}",
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
