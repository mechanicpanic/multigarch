# multigarch 0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DCC-GARCH practical and statistically sound at n ≈ 30–500 assets (composite likelihood), fix optimizer numerics, vectorize covariance paths, and add diagnostics, tests, CI, and benchmarks.

**Architecture:** Two modules stay as-is: `src/multigarch/models.py` (orchestration: `GARCH`, `CCC`, `DCC` + helpers) and `src/multigarch/_jit.py` (numba kernels). New pairwise composite-likelihood kernel joins the optimized full-likelihood kernel; `DCC(method="auto")` dispatches between them. SLSQP with explicit constraints replaces L-BFGS-B + 1e10 penalty cliffs.

**Tech Stack:** Python 3.10+, numpy, scipy (SLSQP), numba (`njit`, `prange`, `nogil`), joblib (threads), pytest, ruff, uv, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-10-multigarch-0.2-design.md`

**Conventions for every task:**
- Run commands from the repo root `/Users/aleph/Projects/research/mvgarch-pkg`.
- Use `uv run pytest …` (the venv is created in Task 1).
- Commit messages: imperative, ≤72-char subject, end body with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Housekeeping — gitignore, stray dir, tooling config, version bump

**Files:**
- Create: `.gitignore`
- Delete: `src/mvgarch/` (empty stray directory), `./.DS_Store`, `src/.DS_Store`
- Modify: `pyproject.toml`
- Modify: `src/multigarch/__init__.py:6`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
.DS_Store
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 2: Remove stray files/dirs**

```bash
rmdir src/mvgarch
rm -f .DS_Store src/.DS_Store
```

- [ ] **Step 3: Update `pyproject.toml`**

Change `version = "0.1.0"` to `version = "0.2.0"`. Change the dev extra and append tool config so the file ends like this:

```toml
[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.4"]

[tool.hatch.build.targets.wheel]
packages = ["src/multigarch"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

(Only the `dev = [...]` line changes inside existing sections; `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.pytest.ini_options]` are new sections at the end.)

- [ ] **Step 4: Bump version in `src/multigarch/__init__.py`**

```python
__version__ = "0.2.0"
```

- [ ] **Step 5: Sync environment and run baseline test**

Run: `uv sync --extra dev && uv run pytest -q`
Expected: `1 passed` (the existing `tests/test_dcc.py` test). First run includes numba JIT compilation; allow ~1 min.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Add gitignore and dev tooling config, bump to 0.2.0"
```

---

### Task 2: Characterization tests for the DCC full-likelihood kernel

Pin down the current kernel's numerics with a pure-numpy reference BEFORE rewriting it in Task 6. These tests must pass against the CURRENT kernel.

**Files:**
- Create: `tests/test_jit.py`

- [ ] **Step 1: Write reference tests**

```python
"""Reference tests for numba kernels against pure-numpy implementations."""

import numpy as np
import pytest

from multigarch._jit import dcc_loglik_loop


def _numpy_dcc_nll(std_resid, Q_bar, a, b):
    """Straightforward numpy DCC negative log-likelihood (0.5*(logdet+quad), t>=1)."""
    T, n = std_resid.shape
    Q = Q_bar.copy()
    nll = 0.0
    for t in range(1, T):
        e = std_resid[t - 1]
        Q = (1.0 - a - b) * Q_bar + a * np.outer(e, e) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        _, logdet = np.linalg.slogdet(R)
        x = std_resid[t]
        nll += logdet + x @ np.linalg.solve(R, x)
    return 0.5 * nll


@pytest.mark.parametrize("n", [2, 4, 7])
@pytest.mark.parametrize("ab", [(0.05, 0.90), (0.02, 0.97), (0.20, 0.50)])
def test_full_loglik_matches_numpy_reference(n, ab):
    a, b = ab
    rng = np.random.default_rng(2)
    e = rng.standard_normal((60, n))
    Q_bar = np.corrcoef(e.T)
    val = dcc_loglik_loop(e, Q_bar, a, b)
    ref = _numpy_dcc_nll(e, Q_bar, a, b)
    assert val == pytest.approx(ref, rel=1e-9)
```

- [ ] **Step 2: Run tests — they must pass against the current kernel**

Run: `uv run pytest tests/test_jit.py -v`
Expected: 9 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jit.py
git commit -m "Add reference tests pinning DCC full-likelihood kernel"
```

---

### Task 3: Input validation and fitted-state helpers

**Files:**
- Modify: `src/multigarch/models.py` (add helpers near top; wire into all `fit`/`forecast`)
- Create: `tests/test_validation.py`

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
import pytest

from multigarch import CCC, DCC, GARCH


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_garch_rejects_nonfinite(bad):
    r = np.zeros(100)
    r[5] = bad
    with pytest.raises(ValueError, match="finite"):
        GARCH().fit(r)


@pytest.mark.parametrize("cls", [CCC, DCC])
def test_multivariate_rejects_nonfinite(cls):
    r = np.zeros((100, 3))
    r[5, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cls(n_jobs=1).fit(r)


def test_garch_rejects_too_short():
    with pytest.raises(ValueError, match="observations"):
        GARCH().fit(np.array([0.1, -0.2, 0.05]))


def test_not_fitted_raises():
    with pytest.raises(ValueError, match="not fitted"):
        GARCH().forecast(5)
    with pytest.raises(ValueError, match="not fitted"):
        CCC().forecast(5)
    with pytest.raises(ValueError, match="not fitted"):
        DCC().forecast(5)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_validation.py -v`
Expected: FAIL — current errors say "Model not fitted" (matches "not fitted") but the NaN cases don't raise ValueError, and the match strings differ. At least the nonfinite and too-short tests must fail.

- [ ] **Step 3: Implement helpers in `models.py`**

Add after the imports:

```python
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
```

Wire them in:
- `GARCH.fit`: after `returns = np.asarray(...).flatten()`, add `_validate_returns(returns, min_obs=max(10, self.p + self.q + 2))`.
- `CCC.fit` and `DCC.fit`: after the 1-D promotion, add `_validate_returns(returns, min_obs=max(10, self.p + self.q + 2))`.
- `GARCH.forecast`: replace `if self.sigma2 is None: raise ValueError("Model not fitted")` with `_check_fitted(self.sigma2 is not None)`.
- `CCC.forecast` / `DCC.forecast`: replace the `if self.H is None: raise ...` blocks with `_check_fitted(self.H is not None)`. In `DCC.forecast` keep the subsequent `Q_bar`/`Q_last` check as-is.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_validation.py tests/test_dcc.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_validation.py
git commit -m "Add input validation and unified fitted-state checks"
```

---

### Task 4: GARCH optimizer numerics — SLSQP, no penalty cliffs

**Files:**
- Modify: `src/multigarch/_jit.py:7-94` (`garch_variance_loop`, `garch_loglik`)
- Modify: `src/multigarch/models.py` (`GARCH.fit`, imports)
- Create: `tests/test_garch.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify state**

Run: `uv run pytest tests/test_garch.py -v`
Expected: FAIL — `m.converged` does not exist yet (AttributeError). Recovery tests may otherwise pass; the new attribute is the contract change.

- [ ] **Step 3: Update `_jit.py` kernels**

Replace `garch_variance_loop` and `garch_loglik` entirely with (note: clamping INSIDE the recursion keeps the objective finite even if the optimizer briefly probes persistence > 1; the 1e10 cliff returns are gone — feasibility is the optimizer's job now):

```python
@njit(cache=True)
def garch_variance_loop(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> np.ndarray:
    """Compute GARCH(p,q) conditional variance series.

    Each step is clamped to [1e-10, 1e12] so the recursion (and any
    likelihood built on it) stays finite even at infeasible parameters
    probed by the optimizer during line search.
    """
    T = len(returns)
    p = len(alpha)
    q = len(beta)
    max_lag = max(p, q)

    sigma2 = np.full(T, var0)

    for t in range(max_lag, T):
        s = omega
        for i in range(p):
            s += alpha[i] * returns[t - 1 - i] ** 2
        for j in range(q):
            s += beta[j] * sigma2[t - 1 - j]
        if s < 1e-10:
            s = 1e-10
        elif s > 1e12:
            s = 1e12
        sigma2[t] = s

    return sigma2


@njit(cache=True)
def garch_loglik(
    returns: np.ndarray,
    omega: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    var0: float,
) -> float:
    """GARCH(p,q) negative log-likelihood: 0.5 * sum(log s2 + r^2/s2).

    Finite and smooth everywhere; positivity/stationarity are enforced by
    the optimizer's bounds and constraints, not by penalty cliffs.
    """
    if omega < 1e-12:
        omega = 1e-12

    sigma2 = garch_variance_loop(returns, omega, alpha, beta, var0)

    ll = 0.0
    for t in range(len(returns)):
        ll += np.log(sigma2[t]) + returns[t] ** 2 / sigma2[t]

    return 0.5 * ll
```

(`nogil=True` is added across all kernels in Task 6; don't add it here to keep this diff focused.)

- [ ] **Step 4: Update `GARCH.fit` in `models.py`**

Add `import warnings` at the top of the file (stdlib imports first). Replace the body of `fit` from the `def neg_log_likelihood` line through the `self.beta = ...` assignment with:

```python
        def objective(params: NDArray[np.float64]) -> float:
            return garch_loglik(returns, params[0], params[1 : 1 + p], params[1 + p :], var0)

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
```

Also add to `GARCH.__init__`:

```python
        self.converged: bool = False
        self._nll: float | None = None
        self._var0: float | None = None
        self._x_opt: NDArray[np.float64] | None = None
        self._T: int = 0
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_garch.py tests/test_validation.py tests/test_dcc.py -v`
Expected: all pass (numba recompiles the changed kernels on first run).

- [ ] **Step 6: Commit**

```bash
git add src/multigarch/_jit.py src/multigarch/models.py tests/test_garch.py
git commit -m "Replace GARCH penalty cliffs with SLSQP constraints"
```

---

### Task 5: GARCH mean option and diagnostics

**Files:**
- Modify: `src/multigarch/models.py` (`GARCH.__init__`, `GARCH.fit`, new properties, `_fit_single_garch`, CCC/DCC call sites; new module helpers `_numerical_hessian`, `_se_from_hessian`)
- Modify: `tests/test_garch.py` (append tests)

- [ ] **Step 1: Append failing tests to `tests/test_garch.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_garch.py -v`
Expected: new tests FAIL (no `mean` kwarg, no `mu`/`loglik_`/`aic`/`bic`/`std_err`/`summary`).

- [ ] **Step 3: Implement in `models.py`**

Module helpers (after `_check_fitted`):

```python
def _numerical_hessian(f, x: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
    """Central-difference Hessian of scalar function f at x."""
    k = len(x)
    H = np.zeros((k, k))
    h = eps * np.maximum(np.abs(x), 1.0)
    for i in range(k):
        for j in range(i, k):
            xpp = x.copy(); xpp[i] += h[i]; xpp[j] += h[j]
            xpm = x.copy(); xpm[i] += h[i]; xpm[j] -= h[j]
            xmp = x.copy(); xmp[i] -= h[i]; xmp[j] += h[j]
            xmm = x.copy(); xmm[i] -= h[i]; xmm[j] -= h[j]
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
```

`GARCH.__init__` gains `mean: str = "zero"`:

```python
        if mean not in ("zero", "constant"):
            raise ValueError(f"mean must be 'zero' or 'constant', got {mean!r}")
        self.mean = mean
        self.mu: float = 0.0
        self._std_err: NDArray[np.float64] | None = None
```

`GARCH.fit`: right after validation, demean:

```python
        self.mu = float(returns.mean()) if self.mean == "constant" else 0.0
        returns = returns - self.mu
```

(Everything downstream — `var0`, the objective, `self.resid`, `self.sigma2` — now operates on the demeaned series; `self.resid = returns.copy()` stays as-is and is therefore demeaned. Reset `self._std_err = None` at the top of `fit` so refitting clears the cache.)

New methods/properties on `GARCH`:

```python
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
```

Mean pass-through for the multivariate models:

```python
def _fit_single_garch(returns: NDArray, p: int, q: int, mean: str) -> GARCH:
    """Fit a single GARCH model (helper for parallel fitting)."""
    return GARCH(p=p, q=q, mean=mean).fit(returns)
```

`CCC.__init__` and `DCC.__init__` gain `mean: str = "zero"` (validate exactly as in `GARCH.__init__`, store `self.mean = mean`), and both `fit` methods pass it: `delayed(_fit_single_garch)(returns[:, i], self.p, self.q, self.mean)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_garch.py
git commit -m "Add mean option and diagnostics to GARCH"
```

---

### Task 6: JIT kernels — nogil, allocation-free full likelihood, thread parallelism

The Task 2 reference tests guard this rewrite: same math, no per-step allocations, no `try/except`.

**Files:**
- Modify: `src/multigarch/_jit.py` (all kernels)
- Modify: `src/multigarch/models.py` (joblib `prefer="threads"`)

- [ ] **Step 1: Add `nogil=True` to the GARCH kernels**

Change the decorators on `garch_variance_loop` and `garch_loglik` to `@njit(cache=True, nogil=True)`.

- [ ] **Step 2: Rewrite `dcc_loglik_loop`**

Replace entirely. In-place Q update (no `eps @ eps.T` temporary), preallocated buffers, manual modified Cholesky with floored pivots instead of `try/except` (a near-singular R yields a large finite objective via the exploding quadratic form — smooth penalty, no cliff):

```python
@njit(cache=True, nogil=True)
def dcc_loglik_loop(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """DCC negative log-likelihood: 0.5 * sum_{t>=1}(log|R_t| + e' R_t^-1 e).

    Allocation-free inner loop. The Cholesky pivot floor keeps the value
    finite (and large) for near-singular R instead of raising.
    """
    T, n = std_resid.shape
    one_minus_ab = 1.0 - a - b

    Q = Q_bar.copy()
    L = np.zeros((n, n))
    y = np.zeros(n)
    d = np.zeros(n)

    ll = 0.0
    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

        for i in range(n):
            qi = Q[i, i]
            if qi < 1e-12:
                qi = 1e-12
            d[i] = 1.0 / np.sqrt(qi)

        # Cholesky of R = D Q D, built element-wise from Q without forming R
        for i in range(n):
            for j in range(i + 1):
                s = Q[i, j] * d[i] * d[j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s < 1e-10:
                        s = 1e-10
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]

        logdet = 0.0
        for i in range(n):
            logdet += 2.0 * np.log(L[i, i])

        quad = 0.0
        for i in range(n):
            s = std_resid[t, i]
            for k in range(i):
                s -= L[i, k] * y[k]
            y[i] = s / L[i, i]
            quad += y[i] * y[i]

        ll += logdet + quad

    return 0.5 * ll
```

- [ ] **Step 3: Rewrite `dcc_covariance_loop` and `dcc_final_covariance`**

Same in-place Q update, no per-step allocations:

```python
@njit(cache=True, nogil=True)
def dcc_covariance_loop(
    std_resid: np.ndarray,
    sigmas: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DCC correlation and covariance paths.

    Returns:
        (R, H, Q_last): R and H of shape (T, n, n), Q_last of shape (n, n).
    """
    T, n = std_resid.shape
    R = np.zeros((T, n, n))
    H = np.zeros((T, n, n))
    d = np.zeros(n)

    Q = Q_bar.copy()
    one_minus_ab = 1.0 - a - b

    for i in range(n):
        for j in range(n):
            R[0, i, j] = Q_bar[i, j]
            H[0, i, j] = sigmas[0, i] * Q_bar[i, j] * sigmas[0, j]

    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

        for i in range(n):
            qi = Q[i, i]
            if qi < 1e-12:
                qi = 1e-12
            d[i] = 1.0 / np.sqrt(qi)

        for i in range(n):
            for j in range(n):
                r = Q[i, j] * d[i] * d[j]
                R[t, i, j] = r
                H[t, i, j] = sigmas[t, i] * r * sigmas[t, j]

    return R, H, Q


@njit(cache=True, nogil=True)
def dcc_final_covariance(
    std_resid: np.ndarray,
    sigmas: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final-step DCC correlation/covariance only (low-memory mode).

    Returns:
        (R_final, H_final, Q_last), each of shape (n, n).
    """
    T, n = std_resid.shape
    Q = Q_bar.copy()
    one_minus_ab = 1.0 - a - b

    for t in range(1, T):
        for i in range(n):
            ei = std_resid[t - 1, i]
            for j in range(n):
                Q[i, j] = one_minus_ab * Q_bar[i, j] + a * ei * std_resid[t - 1, j] + b * Q[i, j]

    R_final = np.zeros((n, n))
    H_final = np.zeros((n, n))
    d = np.zeros(n)
    for i in range(n):
        qi = Q[i, i]
        if qi < 1e-12:
            qi = 1e-12
        d[i] = 1.0 / np.sqrt(qi)
    for i in range(n):
        for j in range(n):
            r = Q[i, j] * d[i] * d[j]
            R_final[i, j] = r
            H_final[i, j] = sigmas[T - 1, i] * r * sigmas[T - 1, j]

    return R_final, H_final, Q
```

- [ ] **Step 4: Switch joblib to threads**

In `CCC.fit` and `DCC.fit`, change `Parallel(n_jobs=self.n_jobs)` to `Parallel(n_jobs=self.n_jobs, prefer="threads")`. The kernels now release the GIL, so threads avoid pickling each column and per-process JIT cache loading.

- [ ] **Step 5: Run full suite — characterization tests prove equivalence**

Run: `uv run pytest tests/ -v`
Expected: all pass, in particular `tests/test_jit.py` (rewritten kernel matches the numpy reference) and `tests/test_dcc.py`.

- [ ] **Step 6: Commit**

```bash
git add src/multigarch/_jit.py src/multigarch/models.py
git commit -m "Make JIT kernels allocation-free and GIL-releasing"
```

---

### Task 7: Composite-likelihood kernel

**Files:**
- Modify: `src/multigarch/_jit.py` (new kernel `dcc_cl_loglik`; import `prange`)
- Modify: `tests/test_jit.py` (append reference test)

- [ ] **Step 1: Append failing reference test to `tests/test_jit.py`**

Add the import `from multigarch._jit import dcc_cl_loglik, dcc_loglik_loop` (replacing the existing `_jit` import line) and:

```python
def _numpy_cl_nll(e, Q_bar, a, b):
    """Pairwise composite NLL over contiguous pairs, pure numpy."""
    T, n = e.shape
    total = 0.0
    for i in range(n - 1):
        j = i + 1
        qii = qjj = 1.0
        qij = Q_bar[i, j]
        for t in range(1, T):
            qii = (1 - a - b) + a * e[t - 1, i] ** 2 + b * qii
            qjj = (1 - a - b) + a * e[t - 1, j] ** 2 + b * qjj
            qij = (1 - a - b) * Q_bar[i, j] + a * e[t - 1, i] * e[t - 1, j] + b * qij
            rho = qij / np.sqrt(qii * qjj)
            om = 1.0 - rho * rho
            total += np.log(om) + (e[t, i] ** 2 + e[t, j] ** 2 - 2 * rho * e[t, i] * e[t, j]) / om
    return 0.5 * total


@pytest.mark.parametrize("n", [2, 5, 12])
def test_cl_loglik_matches_numpy_reference(n):
    rng = np.random.default_rng(3)
    e = rng.standard_normal((80, n))
    Q_bar = np.corrcoef(e.T)
    val = dcc_cl_loglik(e, Q_bar, 0.05, 0.90)
    ref = _numpy_cl_nll(e, Q_bar, 0.05, 0.90)
    assert val == pytest.approx(ref, rel=1e-9)


def test_cl_loglik_single_asset_is_zero():
    rng = np.random.default_rng(4)
    e = rng.standard_normal((50, 1))
    assert dcc_cl_loglik(e, np.array([[1.0]]), 0.05, 0.90) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_jit.py -v`
Expected: FAIL with ImportError (`dcc_cl_loglik` not defined).

- [ ] **Step 3: Implement the kernel in `_jit.py`**

Change the numba import to `from numba import njit, prange` and append:

```python
@njit(cache=True, nogil=True, parallel=True)
def dcc_cl_loglik(
    std_resid: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Pairwise composite negative log-likelihood over contiguous pairs.

    Sums closed-form bivariate Gaussian NLLs (0.5*(log(1-rho^2)+quad), t>=1)
    over pairs (i, i+1). O(T*n) per evaluation, no linear algebra. The
    correlation is clamped to +/-(1 - 1e-8) so the value is finite
    everywhere.
    """
    T, n = std_resid.shape
    one_minus_ab = 1.0 - a - b
    rho_max = 1.0 - 1e-8

    total = 0.0
    for k in prange(n - 1):
        i = k
        j = k + 1
        qbar_ij = Q_bar[i, j]
        qii = 1.0
        qjj = 1.0
        qij = qbar_ij
        ll = 0.0
        for t in range(1, T):
            ei = std_resid[t - 1, i]
            ej = std_resid[t - 1, j]
            qii = one_minus_ab + a * ei * ei + b * qii
            qjj = one_minus_ab + a * ej * ej + b * qjj
            qij = one_minus_ab * qbar_ij + a * ei * ej + b * qij

            denom = qii * qjj
            if denom < 1e-24:
                denom = 1e-24
            rho = qij / np.sqrt(denom)
            if rho > rho_max:
                rho = rho_max
            elif rho < -rho_max:
                rho = -rho_max

            one_m_r2 = 1.0 - rho * rho
            xi = std_resid[t, i]
            xj = std_resid[t, j]
            ll += np.log(one_m_r2) + (xi * xi + xj * xj - 2.0 * rho * xi * xj) / one_m_r2
        total += ll

    return 0.5 * total
```

(The diagonal recursions use `one_minus_ab` directly because `Q_bar` has a unit diagonal. `total += ll` inside `prange` is a supported numba scalar reduction.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_jit.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/_jit.py tests/test_jit.py
git commit -m "Add pairwise composite-likelihood kernel for DCC"
```

---

### Task 8: DCC method dispatch (full / cl / auto) and SLSQP

**Files:**
- Modify: `src/multigarch/models.py` (`DCC.__init__`, `DCC.fit`, `_jit` import line)
- Modify: `tests/test_dcc.py` (append tests + simulation helper)

- [ ] **Step 1: Append failing tests to `tests/test_dcc.py`**

Add `import pytest` to the imports at the top of the file (`DCC` and `np` are already imported there), then append:

```python
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


def test_dcc_constraint_respected():
    rng = np.random.default_rng(8)
    returns = rng.standard_normal((400, 4)) * 0.01
    m = DCC(n_jobs=1).fit(returns)
    assert m.a >= 0 and m.b >= 0
    assert m.a + m.b <= 0.999 + 1e-8
```

(`numpy` is already imported at the top of the file as `np`.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dcc.py -v`
Expected: FAIL — `DCC.__init__` has no `method` kwarg.

- [ ] **Step 3: Implement in `models.py`**

Add `dcc_cl_loglik` to the `from multigarch._jit import (...)` block.

`DCC.__init__` signature becomes:

```python
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        n_jobs: int = -1,
        low_memory: bool = False,
        method: str = "auto",
        mean: str = "zero",
    ) -> None:
```

with validation and new attributes (alongside the existing ones):

```python
        if method not in ("auto", "full", "cl"):
            raise ValueError(f"method must be 'auto', 'full' or 'cl', got {method!r}")
        self.method = method
        self.method_: str | None = None
        self.converged: bool = False
```

In `DCC.fit`, replace the step-2 block (from `def neg_log_likelihood` through `self.a, self.b = result.x`) with:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_dcc.py tests/test_jit.py -v`
Expected: all pass. (`test_cl_agrees_with_full_on_small_n` is the slowest — two fits on T=1500.)

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_dcc.py
git commit -m "Add composite-likelihood method dispatch to DCC"
```

---

### Task 9: Correlation-target shrinkage (CCC and DCC)

**Files:**
- Modify: `src/multigarch/models.py` (`CCC.__init__`/`fit`, `DCC.__init__`/`fit`)
- Create: `tests/test_ccc.py` (shrinkage tests; more CCC tests arrive in Task 10)
- Modify: `tests/test_dcc.py` (append shrinkage test)

- [ ] **Step 1: Write failing tests**

`tests/test_ccc.py`:

```python
import numpy as np
import pytest

from multigarch import CCC


def test_shrinkage_one_gives_identity():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((300, 4)) * 0.01
    m = CCC(n_jobs=1, shrinkage=1.0).fit(returns)
    np.testing.assert_allclose(m.R, np.eye(4), atol=1e-12)


def test_shrinkage_interpolates():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((300, 4)) * 0.01
    m0 = CCC(n_jobs=1, shrinkage=0.0).fit(returns)
    m5 = CCC(n_jobs=1, shrinkage=0.5).fit(returns)
    np.testing.assert_allclose(m5.R, 0.5 * m0.R + 0.5 * np.eye(4), atol=1e-10)


def test_invalid_shrinkage_raises():
    with pytest.raises(ValueError, match="shrinkage"):
        CCC(shrinkage=1.5)
```

Append to `tests/test_dcc.py`:

```python
def test_dcc_shrinkage_applies_to_q_bar():
    returns = simulate_dcc_returns(300, 3, a=0.05, b=0.90, seed=9)
    m = DCC(n_jobs=1, shrinkage=1.0).fit(returns)
    np.testing.assert_allclose(m.Q_bar, np.eye(3), atol=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ccc.py tests/test_dcc.py -v`
Expected: FAIL — no `shrinkage` kwarg.

- [ ] **Step 3: Implement**

Both `CCC.__init__` and `DCC.__init__` gain `shrinkage: float = 0.0` with validation:

```python
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
        self.shrinkage = shrinkage
```

In `CCC.fit`, replace the `self.R = np.corrcoef(...)` block with:

```python
        S = np.corrcoef(std_resid.T) if n > 1 else np.array([[1.0]])
        self.R = (1.0 - self.shrinkage) * S + self.shrinkage * np.eye(n)
```

In `DCC.fit`, replace the `self.Q_bar = ...` block with:

```python
        S = np.corrcoef(std_resid.T) if n > 1 else np.array([[1.0]])
        self.Q_bar = (1.0 - self.shrinkage) * S + self.shrinkage * np.eye(n)
```

(Document in both class docstrings: use `shrinkage > 0` when T is not much larger than n; it guarantees a well-conditioned correlation target.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ccc.py tests/test_dcc.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_ccc.py tests/test_dcc.py
git commit -m "Add correlation-target shrinkage to CCC and DCC"
```

---

### Task 10: Vectorized covariance paths and forecasts; invariants and edge cases

**Files:**
- Modify: `src/multigarch/models.py` (`CCC.fit`, `CCC.forecast`, `DCC.forecast`)
- Modify: `tests/test_ccc.py`, `tests/test_dcc.py` (append tests)

- [ ] **Step 1: Append failing/characterization tests**

To `tests/test_ccc.py`:

```python
def _fit_ccc(seed=0, T=300, n=4):
    rng = np.random.default_rng(seed)
    return CCC(n_jobs=1).fit(rng.standard_normal((T, n)) * 0.01)


def test_h_path_matches_reference():
    m = _fit_ccc()
    sigmas = np.sqrt(np.column_stack([g.sigma2 for g in m.garch_models]))
    for t in [0, 150, 299]:
        D = np.diag(sigmas[t])
        np.testing.assert_allclose(m.H[t], D @ m.R @ D, rtol=1e-10)


def test_r_properties():
    m = _fit_ccc()
    np.testing.assert_allclose(np.diag(m.R), 1.0, atol=1e-10)
    np.testing.assert_allclose(m.R, m.R.T, atol=1e-12)
    assert np.linalg.eigvalsh(m.R).min() > -1e-10


def test_low_memory_matches_final_step():
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((300, 4)) * 0.01
    full = CCC(n_jobs=1).fit(returns)
    lm = CCC(n_jobs=1, low_memory=True).fit(returns)
    np.testing.assert_allclose(lm.H, full.H[-1], rtol=1e-10)


def test_forecast_shapes_and_symmetry():
    m = _fit_ccc()
    f = m.forecast(5)
    assert f.shape == (5, 4, 4)
    for h in range(5):
        np.testing.assert_allclose(f[h], f[h].T, atol=1e-12)
        assert np.linalg.eigvalsh(f[h]).min() > 0


def test_single_asset():
    rng = np.random.default_rng(2)
    m = CCC(n_jobs=1).fit(rng.standard_normal(200) * 0.01)
    assert m.H.shape == (200, 1, 1)
    assert np.all(np.isfinite(m.H))
```

To `tests/test_dcc.py`:

```python
def test_dcc_low_memory_matches_final_step():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=10)
    full = DCC(n_jobs=1, low_memory=False).fit(returns)
    lm = DCC(n_jobs=1, low_memory=True).fit(returns)
    np.testing.assert_allclose(lm.H, full.H[-1], rtol=1e-8)
    np.testing.assert_allclose(lm.R, full.R[-1], rtol=1e-8)
    np.testing.assert_allclose(lm.Q_last, full.Q_last, rtol=1e-8)


def test_dcc_invariants():
    returns = simulate_dcc_returns(400, 4, a=0.05, b=0.90, seed=11)
    m = DCC(n_jobs=1).fit(returns)
    for t in [0, 200, 399]:
        np.testing.assert_allclose(np.diag(m.R[t]), 1.0, atol=1e-10)
        np.testing.assert_allclose(m.H[t], m.H[t].T, atol=1e-14)
        assert np.abs(m.R[t]).max() <= 1.0 + 1e-10
        assert np.linalg.eigvalsh(m.H[t]).min() > -1e-12


def test_dcc_single_asset():
    rng = np.random.default_rng(12)
    m = DCC(n_jobs=1).fit(rng.standard_normal(200) * 0.01)
    assert m.H.shape == (200, 1, 1)
    assert np.all(np.isfinite(m.H))


def test_dcc_forecast_psd():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=13)
    m = DCC(n_jobs=1).fit(returns)
    f = m.forecast(10)
    assert f.shape == (10, 3, 3)
    for h in range(10):
        assert np.linalg.eigvalsh(f[h]).min() > 0
```

- [ ] **Step 2: Run — most should already pass (characterization); single-asset may fail**

Run: `uv run pytest tests/test_ccc.py tests/test_dcc.py -v`
Expected: mostly pass against current code. If `np.corrcoef` warnings or shape issues appear in single-asset tests, the implementations in Step 3 fix them. Note which fail.

- [ ] **Step 3: Vectorize in `models.py`**

`CCC.fit` — replace the `if self.low_memory: ... else: ...` block with:

```python
        if self.low_memory:
            s_last = sigmas[-1]
            self.H = self.R * np.outer(s_last, s_last)
        else:
            self.H = sigmas[:, :, None] * self.R[None, :, :] * sigmas[:, None, :]
```

`CCC.forecast` — replace the per-horizon `D = np.diag(...); forecasts[h] = D @ self.R @ D` loop with:

```python
        for h in range(horizon):
            s = np.sqrt(var_forecasts[h])
            forecasts[h] = self.R * np.outer(s, s)
```

`DCC.forecast` — replace the body of the per-horizon loop with:

```python
        for h in range(horizon):
            Q_forecast = one_minus_b * self.Q_bar + self.b * Q_forecast

            d = 1.0 / np.sqrt(np.diag(Q_forecast))
            R_forecast = Q_forecast * np.outer(d, d)

            s = np.sqrt(var_forecasts[h])
            forecasts[h] = R_forecast * np.outer(s, s)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass, including the original `test_dcc_forecast_uses_q_expectation`.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_ccc.py tests/test_dcc.py
git commit -m "Vectorize covariance paths and forecasts"
```

---

### Task 11: CCC/DCC diagnostics — loglik_, aic, bic, std_err, summary

**Files:**
- Modify: `src/multigarch/models.py` (`CCC`, `DCC`)
- Modify: `tests/test_ccc.py`, `tests/test_dcc.py` (append tests)

- [ ] **Step 1: Append failing tests**

To `tests/test_ccc.py`:

```python
def test_ccc_diagnostics():
    m = _fit_ccc(T=400)
    assert np.isfinite(m.loglik_)
    k = 4 * 3  # n assets x (omega, alpha, beta)
    assert m.aic == pytest.approx(2 * k - 2 * m.loglik_)
    assert m.bic == pytest.approx(k * np.log(400) - 2 * m.loglik_)
    s = m.summary()
    assert "CCC" in s and "loglik" in s
```

To `tests/test_dcc.py`:

```python
def test_dcc_diagnostics():
    returns = simulate_dcc_returns(400, 3, a=0.05, b=0.90, seed=14)
    m = DCC(n_jobs=1).fit(returns)
    assert np.isfinite(m.loglik_)
    k = 3 * 3 + 2  # univariate params + (a, b)
    assert m.aic == pytest.approx(2 * k - 2 * m.loglik_)
    assert m.bic == pytest.approx(k * np.log(400) - 2 * m.loglik_)
    se = m.std_err
    assert se.shape == (2,)
    assert np.all((se > 0) | np.isnan(se))
    s = m.summary()
    assert "DCC" in s and "method" in s


def test_dcc_loglik_consistent_across_methods():
    # loglik_ always reports the full-likelihood value, so CL and full fits
    # on the same data should report similar (not identical) values
    returns = simulate_dcc_returns(800, 3, a=0.06, b=0.90, seed=15)
    full = DCC(n_jobs=1, method="full").fit(returns)
    cl = DCC(n_jobs=1, method="cl").fit(returns)
    assert abs(full.loglik_ - cl.loglik_) / abs(full.loglik_) < 0.01
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ccc.py::test_ccc_diagnostics tests/test_dcc.py::test_dcc_diagnostics -v`
Expected: FAIL — no `loglik_` attribute.

- [ ] **Step 3: Implement in `models.py`**

Both `CCC.fit` and `DCC.fit` must store `self._std_resid = std_resid` (right after computing it) and reset caches at the top of `fit`: `self._loglik = None`, and for DCC `self._std_err = None`. Initialize all three to `None` in `__init__`.

`CCC` additions:

```python
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
```

`DCC` additions:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multigarch/models.py tests/test_ccc.py tests/test_dcc.py
git commit -m "Add quasi-likelihood diagnostics to CCC and DCC"
```

---

### Task 12: Ruff clean-up pass and CI workflow

**Files:**
- Modify: anything ruff flags (`src/`, `tests/`, `benchmarks/` once it exists)
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Run ruff and fix findings**

Run: `uv run ruff check src tests --fix` then `uv run ruff check src tests`
Expected after fixes: `All checks passed!`. Likely findings: import sorting (I), `np.ndarray` type modernization hints (UP), line length (E501). Fix manually what `--fix` can't. Do not change numerical logic.

- [ ] **Step 2: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --extra dev
      - run: uv run ruff check src tests
      - run: uv run pytest -q
```

- [ ] **Step 3: Run the full suite once more**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Add ruff fixes and GitHub Actions CI"
```

---

### Task 13: Benchmarks and README

**Files:**
- Create: `benchmarks/bench.py`
- Modify: `README.md`

- [ ] **Step 1: Create `benchmarks/bench.py`**

```python
"""Fit-time benchmarks for multigarch.

Run:  uv run python benchmarks/bench.py [--quick]
"""

import sys
import time

import numpy as np

from multigarch import CCC, DCC


def simulate(T, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.01


def timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main():
    quick = "--quick" in sys.argv
    T = 500 if quick else 2000
    sizes = [5, 10] if quick else [10, 50, 100, 300]
    full_max_n = 50  # full likelihood is O(T*n^3); skip above this

    # Warm up JIT compilation so timings measure the algorithms
    warm = simulate(200, 3, seed=1)
    DCC(n_jobs=1, method="full", low_memory=True).fit(warm)
    DCC(n_jobs=1, method="cl", low_memory=True).fit(warm)

    print(f"\nT={T}, low_memory=True, times in seconds\n")
    print("| n | CCC | DCC (cl) | DCC (full) |")
    print("|---|-----|----------|------------|")
    for n in sizes:
        r = simulate(T, n)
        t_ccc = timeit(lambda: CCC(low_memory=True).fit(r))
        t_cl = timeit(lambda: DCC(method="cl", low_memory=True).fit(r))
        if n <= full_max_n:
            t_full = f"{timeit(lambda: DCC(method='full', low_memory=True).fit(r)):.2f}"
        else:
            t_full = "—"
        print(f"| {n} | {t_ccc:.2f} | {t_cl:.2f} | {t_full} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `uv run python benchmarks/bench.py`
Expected: a markdown table; CL column grows roughly linearly in n, full column steeply. Copy the printed table for Step 3. (If the n=300 row takes more than ~10 minutes, rerun with `--quick` and note the limitation in the README instead — but CL at n=300 should be seconds.)

- [ ] **Step 3: Update `README.md`**

Replace the `## Models` and `## Parameters` sections with:

```markdown
## Models

| Model | Description | Cost per likelihood eval |
|-------|-------------|--------------------------|
| `GARCH(p,q)` | Univariate GARCH | O(T) |
| `CCC(p,q)` | Constant Conditional Correlation | no correlation optimization |
| `DCC(p,q)` | Dynamic Conditional Correlation | O(T·n³) full / O(T·n) composite |

## Options

- `p`, `q`: ARCH / GARCH orders (default 1)
- `mean`: `"zero"` (default) or `"constant"` (demean, stores `mu`)
- `n_jobs`: parallel jobs for univariate fits, -1 = all cores (CCC/DCC)
- `low_memory`: store only final covariance/correlation instead of the full
  (T, n, n) paths. Storing both R and H paths costs `2·T·n²·8` bytes —
  about 8 GB at T=2000, n=500 — so use `low_memory=True` at large n.
- `method` (DCC): `"full"` exact likelihood, `"cl"` pairwise composite
  likelihood (recommended for large n: less biased and dramatically faster),
  `"auto"` (default) picks `"cl"` when n > 25.
- `shrinkage` (CCC/DCC): shrink the correlation target toward identity,
  `Q̄ ← (1−δ)·S + δ·I`. Use δ > 0 when T is not much larger than n.

## Diagnostics

After fitting: `loglik_`, `aic`, `bic`, `summary()` on all models;
`std_err` on `GARCH` (per-parameter) and `DCC` (for `a`, `b`; approximate —
no sandwich correction). Non-convergence raises a `RuntimeWarning`.

## Benchmarks

<!-- paste the table printed by `uv run python benchmarks/bench.py` here -->
```

Then paste the actual benchmark table from Step 2 in place of the comment, with a one-line note of the machine it was run on (e.g. "Apple Silicon, single run"). Also update the usage example to mention `DCC(method="auto")` behavior in a comment.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench.py README.md
git commit -m "Add benchmarks and document 0.2.0 options"
```

---

### Task 14: Final verification

- [ ] **Step 1: Full suite, clean tree check**

Run: `uv run ruff check src tests && uv run pytest -q && git status --short`
Expected: ruff clean, all tests pass, no unexpected untracked files.

- [ ] **Step 2: Smoke-test the headline feature end to end**

Run:
```bash
uv run python -c "
import numpy as np
from multigarch import DCC
rng = np.random.default_rng(0)
r = rng.standard_normal((1000, 100)) * 0.01
import time; t0 = time.perf_counter()
m = DCC(low_memory=True).fit(r)
print(f'n=100 fit in {time.perf_counter()-t0:.1f}s, method={m.method_}, a={m.a:.4f}, b={m.b:.4f}')
print(m.summary())
"
```
Expected: completes in seconds (not minutes), `method=cl`, finite parameters, summary prints.

- [ ] **Step 3: Verify all spec requirements are met**

Walk through `docs/superpowers/specs/2026-06-10-multigarch-0.2-design.md` section by section and confirm each maps to landed code. Use superpowers:verification-before-completion before claiming done.

---

## Spec coverage map

| Spec section | Tasks |
|---|---|
| §1 Composite likelihood + method dispatch | 7, 8 |
| §1 Full likelihood optimized (alloc-free, flagged Cholesky) | 2 (pin), 6 |
| §1 Shrinkage | 9 |
| §2 SLSQP, no cliffs, convergence warnings | 4 (GARCH), 8 (DCC) |
| §3 nogil + threads, mean option | 6, 5 |
| §4 Vectorized paths/forecasts | 10 |
| §5 Diagnostics, validation, fitted checks | 3, 5, 11 |
| §6 Tests/CI/benchmarks | 2–11 (tests), 12 (CI), 13 (bench) |
| §7 Cleanups (gitignore, stray dir, README) | 1, 13 |




