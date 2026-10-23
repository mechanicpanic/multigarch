"""GARCH and DCC-GARCH model implementations."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numba
import numpy as np
from numpy.typing import NDArray
from scipy import special, stats
from scipy.optimize import minimize

from multigarch._jit import (
    _ETA_MAX,
    _ETA_MIN,
    _MAX_PERSISTENCE,
    _OMEGA_MIN,
    DIST_NORMAL,
    DIST_T,
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

_DISTS = {"normal": DIST_NORMAL, "t": DIST_T}
_NU_START = 8.0


def _garch_starts(p: int, o: int, q: int, dist: str) -> tuple[NDArray[np.float64], int]:
    """Starting points (omega, alpha, g, beta[, eta]) in unit-variance space, one per row.

    Returns (starts, n_always). All starts always run. GARCH(1,1) and
    GJR-GARCH(1,1,1) use a high- and a low-persistence start: even
    order-one likelihoods can be bimodal (one CRSP stock had a second
    mode 8.6 log-likelihood units better), and on near-white-noise
    series the default start leads onto the flat omega-beta ridge.
    Higher orders (p > 1 or q > 1) add starts concentrated on the first
    and last lags.
    """
    eta = [1.0 / _NU_START] if dist == "t" else []

    def x(a_tot, g_tot, b_tot, omega, a_w=None, b_w=None):
        a = np.full(p, a_tot / p) if a_w is None else a_tot * np.asarray(a_w)
        g = np.full(o, g_tot / o) if o else np.empty(0)
        b = np.full(q, b_tot / q) if b_w is None else b_tot * np.asarray(b_w)
        return np.concatenate([[omega], a, g, b, eta])

    # gamma = 2 g: the default start has gamma ~ 0.04, alpha + gamma/2 ~ 0.05
    g0 = 0.02 if o else 0.0
    a0 = 0.05 - g0
    starts = [x(a0, g0, 0.90, 0.05)]
    if p == 1 and q == 1:
        starts.append(x(0.10 - g0, g0, 0.70, 0.20))
        return np.array(starts), len(starts)
    e = 1e-3

    def first(k):
        return np.r_[1.0, np.full(k - 1, e)]

    def last(k):
        return np.r_[np.full(k - 1, e), 1.0]

    starts += [
        x(a0, g0, 0.90, 0.05, a_w=first(p), b_w=first(q)),
        x(a0, g0, 0.90, 0.05, a_w=last(p), b_w=last(q)),
        x(0.15 - g0, g0, 0.60, 0.25),
        x(0.30 - g0, g0, 0.40, 0.30),
    ]
    return np.array(starts), len(starts)


def _garch_slsqp(
    z: NDArray[np.float64],
    var0: float,
    p: int,
    o: int,
    q: int,
    dist: str,
    x0: NDArray[np.float64],
):
    """Constrained SLSQP fit in the unit-variance parametrization (fallback)."""
    T = len(z)
    code = _DISTS[dist]
    n_coef = p + o + q

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nll, grad = garch_loglik_grad(z, params, p, o, q, code, var0)
        return nll / T, grad / T

    jac_persistence = np.zeros(len(x0))
    jac_persistence[1 : 1 + n_coef] = -1.0
    constraints = [
        {
            "type": "ineq",
            "fun": lambda x: _MAX_PERSISTENCE - np.sum(x[1 : 1 + n_coef]),
            "jac": lambda x: jac_persistence,
        }
    ]
    bounds = [(_OMEGA_MIN, None)] + [(0.0, _MAX_PERSISTENCE)] * n_coef
    if dist == "t":
        bounds.append((_ETA_MIN, _ETA_MAX))
    return minimize(
        objective,
        x0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500},
    )


def _fit_garch_models(
    models: list[GARCH], series: list[NDArray[np.float64]], n_jobs: int
) -> list[GARCH]:
    """Fit each model to its series with one parallel numba batch.

    All models must share (p, o, q, dist). Series where projected BFGS
    does not converge are refit with SLSQP from its point; a
    RuntimeWarning is emitted only if that fails too.
    """
    m0 = models[0]
    p, o, q, dist = m0.p, m0.o, m0.q, m0.dist
    prepared = [m._prepare(r) for m, r in zip(models, series, strict=True)]
    Z = np.stack([z for _, z, _ in prepared])
    var0 = Z.var(axis=1)
    starts, n_always = _garch_starts(p, o, q, dist)
    with _numba_threads(n_jobs):
        X, F, conv = garch_fit_batch(
            Z,
            starts,
            p,
            o,
            q,
            _DISTS[dist],
            var0,
            _FIT_MAXITER,
            _FIT_GTOL,
            _FIT_FTOL,
            n_always,
        )
    for i, (m, (resid, z, scale)) in enumerate(zip(models, prepared, strict=True)):
        x, f, ok = X[i], float(F[i]), bool(conv[i])
        if not ok:
            result = _garch_slsqp(z, float(var0[i]), p, o, q, dist, x)
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
    returns: NDArray[np.float64],
    p: int,
    q: int,
    mean: str,
    n_jobs: int,
    o: int = 0,
    dist: str = "normal",
) -> list[GARCH]:
    """Fit a univariate GARCH to every column of a (T, n) array."""
    n = returns.shape[1]
    models = [GARCH(p=p, q=q, mean=mean, o=o, dist=dist) for _ in range(n)]
    return _fit_garch_models(models, [returns[:, i] for i in range(n)], n_jobs)


def _check_garch_spec(p: int, q: int, o: int, mean: str, dist: str) -> None:
    if p < 1 or q < 1:
        raise ValueError("p and q must be >= 1")
    if o < 0:
        raise ValueError("o must be >= 0")
    if mean not in ("zero", "constant"):
        raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
    if dist not in _DISTS:
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def _std_quantile(dist: str, nu: float | None, prob: float) -> float:
    """Quantile of the unit-variance innovation distribution."""
    if dist == "t":
        return float(special.stdtrit(nu, prob) * np.sqrt((nu - 2.0) / nu))
    return float(special.ndtri(prob))


def _std_tail_mean(dist: str, nu: float | None, level: float) -> float:
    """E[-z | z <= q_{1-level}] for the unit-variance innovation z (positive)."""
    a = 1.0 - level
    if dist == "t":
        t_a = special.stdtrit(nu, a)
        tail = stats.t.pdf(t_a, nu) * (nu + t_a * t_a) / ((nu - 1.0) * a)
        return float(tail * np.sqrt((nu - 2.0) / nu))
    return float(stats.norm.pdf(special.ndtri(a)) / a)


class GARCH:
    r"""Univariate GARCH(p,q) or GJR-GARCH(p,o,q) model with JIT-accelerated fitting.

    Model:

    $$\sigma^2_t = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon^2_{t-i}
                 + \sum_{k=1}^{o} \gamma_k \varepsilon^2_{t-k}
                   \mathbb{1}[\varepsilon_{t-k} < 0]
                 + \sum_{j=1}^{q} \beta_j \sigma^2_{t-j}$$

    with $\varepsilon_t = \sigma_t z_t$ and $z_t$ standard normal or
    Student-t standardized to unit variance. The asymmetric (GJR) terms
    let negative shocks raise volatility more than positive ones — the
    leverage effect typical of equity returns. All coefficients are
    non-negative and the persistence
    $\sum\alpha + \sum\beta + \tfrac12\sum\gamma$ is kept below 0.999.

    Parameters after fitting:
        omega: long-run variance weight $\omega$
        alpha: ARCH coefficients $\alpha_1, \dots, \alpha_p$
        gamma: asymmetry coefficients $\gamma_1, \dots, \gamma_o$ (empty if o = 0)
        beta: GARCH coefficients $\beta_1, \dots, \beta_q$
        nu: Student-t degrees of freedom $\nu > 2$ (None for normal)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "zero",
        o: int = 0,
        dist: str = "normal",
    ) -> None:
        """Initialize a GARCH(p,q) / GJR-GARCH(p,o,q) model.

        Args:
            p: ARCH order (number of lagged squared residuals)
            q: GARCH order (number of lagged variances)
            mean: "zero" (default) fits raw returns; "constant" demeans
                first and stores the sample mean as ``mu``
            o: asymmetric (GJR) order; 0 (default) is plain GARCH
            dist: innovation distribution, "normal" (default) or "t"
                (Student-t with estimated degrees of freedom ``nu``)
        """
        _check_garch_spec(p, q, o, mean, dist)
        self.p = p
        self.q = q
        self.o = o
        self.mean = mean
        self.dist = dist
        self.mu: float = 0.0
        self._std_err: NDArray[np.float64] | None = None
        self.omega: float | None = None
        self.alpha: NDArray[np.float64] | None = None
        self.gamma: NDArray[np.float64] | None = None
        self.beta: NDArray[np.float64] | None = None
        self.nu: float | None = None
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
        """Fit the model to a return series.

        Args:
            returns: 1D array of returns

        Returns:
            self for chaining
        """
        returns = np.asarray(returns, dtype=np.float64).flatten()
        _validate_returns(returns, min_obs=max(10, self.p + self.o + self.q + 2))
        return _fit_garch_models([self], [returns], n_jobs=1)[0]

    def _prepare(self, returns: NDArray[np.float64]):
        """Demean and rescale; returns (resid, z, scale) with z = resid / scale.

        The optimizer works on unit-variance z: with raw daily returns
        omega is ~1e-6 next to alpha, beta ~0.1, which stalls optimizers
        far from the optimum. omega scales by scale**2; alpha, gamma,
        beta and nu are scale-free.
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
        """Store a fit given optimal unit-variance params x = (omega_z, alpha, g, beta[, eta])."""
        p, o, q = self.p, self.o, self.q
        T = len(z)
        scale2 = scale * scale
        x = np.array(x, dtype=np.float64)
        self._std_err = None
        self.converged = converged
        self.omega = float(x[0]) * scale2
        self.alpha = x[1 : 1 + p].copy()
        self.gamma = 2.0 * x[1 + p : 1 + p + o]
        self.beta = x[1 + p + o : 1 + p + o + q].copy()
        self.nu = 1.0 / float(x[1 + p + o + q]) if self.dist == "t" else None
        # nll(resid) = nll(z) + T * log(scale) (Jacobian of the rescale)
        self._nll = nll_per_obs * T + T * np.log(scale)
        self._z = z
        self._scale = scale
        self._var0 = var0
        self._x_opt = x
        self._T = T
        self.resid = resid
        self.sigma2 = scale2 * garch_variance_loop(z, x, p, o, q, var0)

    def _objective(self, x: NDArray[np.float64]) -> float:
        """Negative log-likelihood in the unit-variance parametrization."""
        return garch_loglik(self._z, x, self.p, self.o, self.q, _DISTS[self.dist], self._var0)

    @property
    def _n_params(self) -> int:
        return (
            1
            + self.p
            + self.o
            + self.q
            + (1 if self.dist == "t" else 0)
            + (1 if self.mean == "constant" else 0)
        )

    @property
    def persistence(self) -> float:
        r"""$\sum\alpha + \sum\beta + \tfrac12\sum\gamma$ (variance persistence)."""
        _check_fitted(self.sigma2 is not None)
        return float(self.alpha.sum() + self.beta.sum() + 0.5 * self.gamma.sum())

    @property
    def loglik_(self) -> float:
        """Log-likelihood at the optimum, constants included."""
        _check_fitted(self.sigma2 is not None)
        if self.dist == "t":
            return -self._nll
        return -self._nll - 0.5 * self._T * np.log(2.0 * np.pi)

    @property
    def aic(self) -> float:
        return 2.0 * self._n_params - 2.0 * self.loglik_

    @property
    def bic(self) -> float:
        return self._n_params * np.log(self._T) - 2.0 * self.loglik_

    @property
    def std_err(self) -> NDArray[np.float64]:
        """Approximate standard errors for (omega, alpha..., gamma..., beta...[, nu]).

        Inverse numerical Hessian of the negative log-likelihood at the
        optimum. NaN where the Hessian is not positive definite. The mean
        parameter (if any) is excluded.
        """
        _check_fitted(self.sigma2 is not None)
        if self._std_err is None:
            # Hessian in the internal parametrization, where all parameters
            # are O(1); map back: omega = scale**2 omega_z, gamma = 2 g,
            # nu = 1 / eta (delta method)
            H = _numerical_hessian(self._objective, self._x_opt)
            se = _se_from_hessian(H)
            se[0] *= self._scale**2
            se[1 + self.p : 1 + self.p + self.o] *= 2.0
            if self.dist == "t":
                eta = self._x_opt[-1]
                se[-1] /= eta * eta
            self._std_err = se
        return self._std_err

    def summary(self) -> str:
        """Formatted model summary."""
        _check_fitted(self.sigma2 is not None)
        name = (
            f"GJR-GARCH({self.p}, {self.o}, {self.q})" if self.o else (f"GARCH({self.p}, {self.q})")
        )
        lines = [f"{name}  T={self._T}  mean={self.mean}  dist={self.dist}"]
        if self.mean == "constant":
            lines.append(f"  mu:    {self.mu:.6g}")
        lines.append(f"  omega: {self.omega:.6g}")
        for i, a in enumerate(self.alpha, 1):
            lines.append(f"  alpha[{i}]: {a:.4f}")
        for k, g in enumerate(self.gamma, 1):
            lines.append(f"  gamma[{k}]: {g:.4f}")
        for j, b in enumerate(self.beta, 1):
            lines.append(f"  beta[{j}]:  {b:.4f}")
        if self.dist == "t":
            lines.append(f"  nu:    {self.nu:.2f}")
        lines += [
            f"  persistence: {self.persistence:.4f}",
            f"  loglik: {self.loglik_:.2f}  AIC: {self.aic:.2f}  BIC: {self.bic:.2f}",
            f"  converged: {self.converged}",
        ]
        return "\n".join(lines)

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast conditional variance.

        Future squared shocks are replaced by their expectations:
        ``E[eps^2] = sigma^2`` and, for symmetric innovations,
        ``E[eps^2 1(eps < 0)] = sigma^2 / 2``.

        Args:
            horizon: number of periods ahead

        Returns:
            Array of forecasted variances
        """
        _check_fitted(self.sigma2 is not None)

        p, o, q = self.p, self.o, self.q
        lag = max(p, o, 1)
        e = self.resid[-lag:]
        e2_hist = list(e**2)
        neg_hist = list(np.where(e < 0.0, e**2, 0.0))
        sigma2_hist = list(self.sigma2[-q:])

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            arch_term = sum(self.alpha[i] * e2_hist[-(i + 1)] for i in range(p))
            asym_term = sum(self.gamma[k] * neg_hist[-(k + 1)] for k in range(o))
            garch_term = sum(self.beta[j] * sigma2_hist[-(j + 1)] for j in range(q))
            forecast_h = self.omega + arch_term + asym_term + garch_term
            forecasts[h] = forecast_h
            e2_hist.append(forecast_h)
            neg_hist.append(0.5 * forecast_h)
            sigma2_hist.append(forecast_h)

        return forecasts

    def value_at_risk(self, level: float = 0.99, horizon: int = 1) -> NDArray[np.float64]:
        r"""One-period value at risk for each of the next ``horizon`` periods.

        $\mathrm{VaR}_h = -(\mu + \sigma_{T+h}\, q_{1-\text{level}})$, with
        $q$ the quantile of the fitted innovation distribution (normal, or
        Student-t standardized to unit variance). Returned as positive
        losses in return units.

        Args:
            level: confidence level, e.g. 0.99 for the 1% left tail
            horizon: number of periods ahead

        Returns:
            Array of shape (horizon,)
        """
        _check_level(level)
        sigma = np.sqrt(self.forecast(horizon))
        return -(self.mu + sigma * _std_quantile(self.dist, self.nu, 1.0 - level))

    def expected_shortfall(self, level: float = 0.975, horizon: int = 1) -> NDArray[np.float64]:
        r"""One-period expected shortfall (CVaR) for each of the next ``horizon`` periods.

        Mean loss beyond the VaR at ``level``:
        $\mathrm{ES}_h = \sigma_{T+h}\, \mathbb{E}[-z \mid z \le q_{1-\text{level}}] - \mu$,
        in closed form for both innovation distributions.

        Args:
            level: confidence level (Basel FRTB uses 0.975)
            horizon: number of periods ahead

        Returns:
            Array of shape (horizon,)
        """
        _check_level(level)
        sigma = np.sqrt(self.forecast(horizon))
        return sigma * _std_tail_mean(self.dist, self.nu, level) - self.mu

    def _refilter(self, returns: NDArray[np.float64]) -> GARCH:
        """Copy of this model run over new returns with the fitted parameters.

        Recomputes residuals and the variance path only (no re-estimation);
        enough for forecasting. Likelihood diagnostics are not available on
        the copy.
        """
        _check_fitted(self.sigma2 is not None)
        new = copy.copy(self)
        resid = np.asarray(returns, dtype=np.float64).flatten() - self.mu
        params = np.concatenate([[self.omega], self.alpha, 0.5 * self.gamma, self.beta])
        new.resid = resid
        new.sigma2 = garch_variance_loop(
            resid, params, self.p, self.o, self.q, float(np.var(resid))
        )
        new._T = len(resid)
        new._nll = None
        new._z = None
        new._std_err = None
        return new


class _PortfolioRisk:
    """Covariance forecasts and portfolio risk shared by CCC and DCC.

    Subclasses provide ``_forecast_parts(horizon)`` returning
    (variance forecasts (horizon, n), correlation forecasts (horizon, n, n)).
    """

    garch_models: list[GARCH]

    def forecast(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast covariance matrices.

        Args:
            horizon: number of periods ahead

        Returns:
            Array of shape (horizon, n_assets, n_assets) of forecasted covariances
        """
        var_f, corr_f = self._forecast_parts(horizon)
        s = np.sqrt(var_f)
        return corr_f * s[:, :, None] * s[:, None, :]

    def portfolio_value_at_risk(
        self,
        weights: NDArray[np.float64],
        level: float = 0.99,
        horizon: int = 1,
        n_sims: int = 20_000,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        r"""One-period VaR of a portfolio for each of the next ``horizon`` periods.

        Normal margins: closed form,
        $-(w^\top\mu + \sqrt{w^\top H_{T+h} w}\; z_{1-\text{level}})$.
        Student-t margins: Monte Carlo with a Gaussian copula — correlated
        normals with the forecast correlation $R_{T+h}$, mapped to each
        asset's fitted unit-variance t and scaled by its volatility
        forecast. Returned as positive losses in return units.

        Args:
            weights: portfolio weights, shape (n_assets,)
            level: confidence level, e.g. 0.99
            horizon: number of periods ahead
            n_sims: simulations per period (t margins only)
            seed: random seed (t margins only)

        Returns:
            Array of shape (horizon,)
        """
        var, _ = self._portfolio_risk(weights, (level,), horizon, n_sims, seed)
        return var[0]

    def portfolio_expected_shortfall(
        self,
        weights: NDArray[np.float64],
        level: float = 0.975,
        horizon: int = 1,
        n_sims: int = 20_000,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """One-period expected shortfall of a portfolio; see ``portfolio_value_at_risk``.

        Returns:
            Array of shape (horizon,)
        """
        _, es = self._portfolio_risk(weights, (level,), horizon, n_sims, seed)
        return es[0]

    def _portfolio_risk(
        self,
        weights: NDArray[np.float64],
        levels: tuple[float, ...],
        horizon: int,
        n_sims: int,
        seed: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(VaR, ES), each (len(levels), horizon); one simulation serves all levels."""
        for level in levels:
            _check_level(level)
        var_f, corr_f = self._forecast_parts(horizon)
        n = var_f.shape[1]
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
        mu_p = float(w @ np.array([g.mu for g in self.garch_models]))
        sigma = np.sqrt(var_f)
        var_out = np.empty((len(levels), horizon))
        es_out = np.empty((len(levels), horizon))

        if self.dist == "normal":
            for h in range(horizon):
                ws = w * sigma[h]
                sd = float(np.sqrt(ws @ corr_f[h] @ ws))
                for k, level in enumerate(levels):
                    var_out[k, h] = sd * _std_quantile("normal", None, level) - mu_p
                    es_out[k, h] = sd * _std_tail_mean("normal", None, level) - mu_p
            return var_out, es_out

        nu = np.array([g.nu for g in self.garch_models])
        t_scale = np.sqrt((nu - 2.0) / nu)
        rng = np.random.default_rng(seed)
        for h in range(horizon):
            L = _safe_cholesky(corr_f[h])
            u = special.ndtr(rng.standard_normal((n_sims, n)) @ L.T)
            eps = special.stdtrit(nu, u) * t_scale
            loss = -((sigma[h] * eps) @ w) - mu_p
            for k, level in enumerate(levels):
                var_out[k, h] = np.quantile(loss, level)
                es_out[k, h] = loss[loss >= var_out[k, h]].mean()
        return var_out, es_out

    def _refilter(self, returns: NDArray[np.float64]):
        """Copy of this model run over new (T, n) returns with the fitted parameters.

        Univariate variance paths are recomputed with fixed GARCH parameters;
        the correlation model keeps its fitted parameters and target and
        only its state is advanced. Final-step matrices only (as with
        ``low_memory=True``); likelihood diagnostics are not available.
        """
        _check_fitted(self.H is not None)
        returns = np.asarray(returns, dtype=np.float64)
        new = copy.copy(self)
        new.garch_models = [g._refilter(returns[:, i]) for i, g in enumerate(self.garch_models)]
        sigmas = np.sqrt(np.column_stack([g.sigma2 for g in new.garch_models]))
        new._std_resid = np.column_stack([g.resid for g in new.garch_models]) / sigmas
        new._T = returns.shape[0]
        new._loglik = None
        new.low_memory = True
        new._advance_state(sigmas)
        return new


def _safe_cholesky(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cholesky of a correlation matrix, clipping negative eigenvalues if needed."""
    try:
        return np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(R)
        R = (vecs * np.maximum(vals, 1e-10)) @ vecs.T
        d = 1.0 / np.sqrt(np.diag(R))
        return np.linalg.cholesky(R * np.outer(d, d))


class CCC(_PortfolioRisk):
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
        o: int = 0,
        dist: str = "normal",
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
            o: asymmetric (GJR) order for the univariate models (0 = plain GARCH)
            dist: innovation distribution for the univariate models,
                "normal" or "t". The correlation stage is a Gaussian
                quasi-likelihood on the standardized residuals either way
        """
        _check_garch_spec(p, q, o, mean, dist)
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.shrinkage = shrinkage
        self.p = p
        self.q = q
        self.o = o
        self.dist = dist
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
        _validate_returns(returns, min_obs=max(10, self.p + self.o + self.q + 2))

        T, n = returns.shape
        self._T = T
        self._n_assets = n

        # Fit univariate GARCH models in parallel
        self.garch_models = _fit_garch_columns(
            returns, self.p, self.q, self.mean, self.n_jobs, o=self.o, dist=self.dist
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

    def _advance_state(self, sigmas: NDArray[np.float64]) -> None:
        self.H = self.R * np.outer(sigmas[-1], sigmas[-1])

    def _forecast_parts(self, horizon: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Variance forecasts (horizon, n) and the constant correlation (horizon, n, n)."""
        _check_fitted(self.H is not None)
        var_f = np.column_stack([g.forecast(horizon) for g in self.garch_models])
        return var_f, np.broadcast_to(self.R, (horizon, *self.R.shape))


class DCC(_PortfolioRisk):
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
        o: int = 0,
        dist: str = "normal",
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
            o: asymmetric (GJR) order for the univariate models (0 = plain GARCH)
            dist: innovation distribution for the univariate models,
                "normal" or "t". The correlation stage is a Gaussian
                quasi-likelihood on the standardized residuals either way
        """
        _check_garch_spec(p, q, o, mean, dist)
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
        self.o = o
        self.dist = dist
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
        _validate_returns(returns, min_obs=max(10, self.p + self.o + self.q + 2))

        T, n = returns.shape
        self._T = T
        self._n_assets = n

        # Step 1: Fit univariate GARCH models in parallel
        self.garch_models = _fit_garch_columns(
            returns, self.p, self.q, self.mean, self.n_jobs, o=self.o, dist=self.dist
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
        conditions on the first observation. One O(T*n^3) pass, parallel
        over time blocks — under a second at n=500 on 14 cores.
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

    def _advance_state(self, sigmas: NDArray[np.float64]) -> None:
        with _numba_threads(self.n_jobs):
            self.R, self.H, self.Q_last = dcc_final_covariance(
                self._std_resid, sigmas, self.Q_bar, self.a, self.b
            )

    def _forecast_parts(self, horizon: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Variance forecasts (horizon, n) and correlation forecasts (horizon, n, n).

        The first step uses the last standardized residual,
        $Q_{T+1} = (1-a-b)\bar{Q} + a\,\varepsilon_T\varepsilon_T' + b\,Q_T$;
        later steps replace $\varepsilon\varepsilon'$ by its approximate
        expectation $Q$ (Engle & Sheppard, 2001), so
        $Q_{T+h} = (1-a-b)\bar{Q} + (a+b)\,Q_{T+h-1}$ reverts to
        $\bar{Q}$ at rate $a+b$.
        """
        _check_fitted(self.H is not None)
        if self.Q_bar is None or self.Q_last is None:
            raise ValueError("Missing DCC state for forecasting")
        var_f = np.column_stack([g.forecast(horizon) for g in self.garch_models])
        a, b = self.a, self.b
        e = self._std_resid[-1]
        Q = (1.0 - a - b) * self.Q_bar + a * np.outer(e, e) + b * self.Q_last
        corr_f = np.empty((horizon, *Q.shape))
        for h in range(horizon):
            if h > 0:
                Q = (1.0 - a - b) * self.Q_bar + (a + b) * Q
            d = 1.0 / np.sqrt(np.diag(Q))
            corr_f[h] = Q * np.outer(d, d)
        return var_f, corr_f
