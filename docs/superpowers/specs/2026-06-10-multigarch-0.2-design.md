# multigarch 0.2.0 — large-n DCC: design

**Date:** 2026-06-10
**Status:** Approved
**Goal:** Make the package statistically sound and fast for cross-sections of
n ≈ 30–500 assets, while tightening numerics, API, and test coverage.

## Motivation

The current DCC estimator maximizes the full second-stage likelihood, which
costs O(T·n³) per optimizer evaluation (T Cholesky factorizations of an n×n
matrix) and is evaluated dozens of times by the optimizer. At n ≥ ~100 this is
impractical, and the full-likelihood estimator of (a, b) is known to be biased
toward zero as n grows. Several smaller defects compound this: 1e10 penalty
cliffs corrupt finite-difference gradients, `CCC.fit` builds its covariance
path with a Python loop of dense matmuls, and the JIT loops allocate
temporaries every step.

## Scope and compatibility

- Same public classes: `GARCH`, `CCC`, `DCC`. Same module layout:
  `models.py` (orchestration) + `_jit.py` (numba kernels).
- No new dependencies (numpy, scipy, numba, joblib).
- Existing attributes keep their names and shapes: `omega`, `alpha`, `beta`,
  `resid`, `sigma2`, `a`, `b`, `Q_bar`, `Q_last`, `R`, `H`, `low_memory`,
  `n_jobs`.
- New behavior arrives via new keyword arguments whose defaults preserve
  current semantics, except where the current behavior is a defect (penalty
  cliffs, optimizer choice). Fitted parameter values may shift within
  optimizer tolerance; at large n with `method="auto"`, DCC (a, b) estimates
  change by design (composite likelihood).
- Version bumps to 0.2.0.

## 1. DCC estimation (core change)

### Composite likelihood

`DCC(method="auto")` with three values:

- `"full"` — current estimator, optimized (below).
- `"cl"` — pairwise composite likelihood over contiguous pairs
  (Engle–Shephard–Sheppard style).
- `"auto"` — `"cl"` when n > 25, else `"full"`. Default.

CL implementation: for each contiguous pair (i, i+1), run three scalar
recursions

```
q_ii[t] = (1-a-b)·q̄_ii + a·e_i[t-1]² + b·q_ii[t-1]      (same for q_jj)
q_ij[t] = (1-a-b)·q̄_ij + a·e_i[t-1]·e_j[t-1] + b·q_ij[t-1]
ρ[t]    = q_ij[t] / sqrt(q_ii[t]·q_jj[t])
```

and accumulate the closed-form bivariate normal log-likelihood

```
ll_t = log(1-ρ²) + (e_i² + e_j² - 2ρ·e_i·e_j) / (1-ρ²)
```

Total objective = 0.5·Σ_pairs Σ_t ll_t. Cost is O(T·n) per evaluation with no
linear algebra. Kernel is `@njit(cache=True, nogil=True, parallel=True)` with
`prange` over pairs. ρ is clamped to [-1+1e-8, 1-1e-8] so the objective stays
finite everywhere.

After (a, b) are estimated (by either method), the R/H path computation is
identical to today (`dcc_covariance_loop` / `dcc_final_covariance`).

### Full likelihood, optimized

- Preallocate Q, R, and solve buffers once; update Q in place
  (rank-1 update written as explicit loops, no `eps @ eps.T` temporary).
- Replace `try: np.linalg.cholesky` with a manual Cholesky that returns a
  failure flag; non-PD R contributes a large *finite, smooth* penalty rather
  than raising.

### Correlation-target shrinkage

`DCC(shrinkage=0.0)` and `CCC(shrinkage=0.0)`: with float δ ∈ [0, 1],
Q̄ ← (1−δ)·S + δ·I where S is the sample correlation of standardized
residuals. Default 0.0 (current behavior). Documented guidance: use δ > 0
when T is not ≫ n.

## 2. Optimizer numerics (GARCH and DCC stages)

- Switch both stages from L-BFGS-B with 1e10 penalty cliffs to SLSQP with
  bounds plus explicit inequality constraints:
  - GARCH: Σα + Σβ ≤ 0.999, ω ≥ 1e-12, α_i, β_j ≥ 0.
  - DCC: a + b ≤ 0.999, a, b ≥ 1e-8.
- Likelihoods are finite and smooth everywhere they can be evaluated:
  variance recursions keep a positive floor (`max(σ², 1e-10)`, as today), CL
  clamps ρ, full
  likelihood uses the Cholesky failure flag → finite penalty. No more 1e10
  returns.
- If `result.success` is False, emit `RuntimeWarning` with the optimizer
  message and keep the best-found parameters.

## 3. Univariate stage

- All `_jit.py` kernels gain `nogil=True`; joblib calls switch to
  `Parallel(n_jobs=..., prefer="threads")`. No pickling of return columns, no
  per-process JIT cache loading.
- `GARCH(mean="zero")` new option: `"zero"` (default, current behavior) or
  `"constant"` (demean before fitting; store `mu`; `resid = returns - mu`).
  `CCC`/`DCC` forward a `mean` kwarg to their univariate fits.

## 4. Covariance-path vectorization

- `CCC.fit` full path: replace the T-loop of `D @ R @ D` with
  `H = sigmas[:, :, None] * R[None, :, :] * sigmas[:, None, :]`
  — O(T·n²) instead of O(T·n³).
- `dcc_covariance_loop`: keep the sequential Q recursion in numba but write
  R and H with the same O(n²) elementwise scheme, no temporaries (already
  elementwise; ensure no per-t allocations remain).
- Forecast methods replace `np.diag(...) @ X @ np.diag(...)` with outer-product
  broadcasting.

## 5. Diagnostics and API

- `loglik_` (float), `aic`, `bic` properties on all three models. For
  CCC/DCC this is the two-step Gaussian quasi-log-likelihood (univariate sum
  + correlation part), documented as such.
- `summary()` → formatted string: model spec, T, n, parameters, loglik,
  AIC/BIC, convergence status.
- `std_err` for univariate GARCH parameters from a finite-difference Hessian
  at the optimum; same for DCC (a, b) with a documented caveat that two-step
  and CL standard errors are approximate (no sandwich correction).
- Input validation in all `fit` methods: reject NaN/inf, require T ≥ 10 and
  T > p+q+1; DCC/CCC require 2-D input or promote 1-D as today.
- Unified `NotFittedError`-style behavior: a `_check_fitted()` helper raising
  `ValueError` with a consistent message (keeps current exception type).

## 6. Tests, CI, benchmarks

- `tests/test_garch.py`: parameter recovery on simulated GARCH(1,1) (seeded,
  loose tolerances), persistence constraint respected, forecast converges to
  unconditional variance, mean="constant" recovers mu, validation errors.
- `tests/test_ccc.py`: R constant and symmetric with unit diagonal, H path
  matches per-t D@R@D reference on small n, PSD, forecast shapes.
- `tests/test_dcc.py`: existing forecast test kept; CL vs full (a, b)
  agreement on small n (n=3, generous tolerance); low_memory final H/R equal
  the last step of the full path; invariants (diag(R)=1, |ρ|≤1, H symmetric
  PSD); n=1 edge case; method="auto" dispatch.
- For numba paths, small sanity checks rather than golden outputs (per
  AGENTS.md).
- Ruff configured in pyproject (default ruleset + isort); pytest config.
- GitHub Actions: uv-based, ruff check + pytest on Python 3.10–3.13.
- `benchmarks/bench.py`: fit wall-times for (T=2000, n ∈ {10, 50, 100, 300}),
  full vs CL vs CCC, printed as a markdown table; README gets the results.

## 7. Cleanups

- Delete the stray empty `src/mvgarch/` directory.
- Add `.gitignore` (`.DS_Store`, `__pycache__/`, `*.egg-info`, `.venv/`,
  `dist/`).
- README: document `method`, `shrinkage`, `mean`, diagnostics, and memory
  expectations (storing both R and H paths costs 2·T·n²·8 bytes ≈ 8 GB at
  T=2000, n=500; `low_memory=True` is the escape hatch).

## Error handling summary

| Condition | Behavior |
|---|---|
| NaN/inf or too-short input | `ValueError` at fit time |
| Optimizer did not converge | `RuntimeWarning`, keep best params |
| Non-PD R inside full likelihood | finite smooth penalty (flagged Cholesky) |
| ρ → ±1 in CL | clamp, finite objective |
| Methods called before fit | `ValueError` via `_check_fitted()` |

## Out of scope

- Non-Gaussian innovations (t-distribution), asymmetric/ADCC dynamics,
  sandwich standard errors, lazy/streaming H storage beyond `low_memory`,
  pandas-aware API. All possible later; none block the 0.2.0 goal.
