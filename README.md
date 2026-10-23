# multigarch

Simple Multivariate GARCH models for Python 3.10+.

Features:
- JIT-compiled inner loops (numba)
- Composite-likelihood DCC estimation — practical at hundreds of assets
- Parallel univariate GARCH fitting in numba (projected BFGS, all cores,
  with or without the GIL)
- GARCH(p,q) and GJR-GARCH(p,o,q) (leverage effect), normal or Student-t
  innovations; diagnostics (loglik/AIC/BIC/std errors)

## Installation

```bash
pip install multigarch
```

or the development version straight from GitHub:

```bash
pip install git+https://github.com/mechanicpanic/multigarch
```

Works on free-threaded Python (`3.13t`/`3.14t`) as well; parallelism
comes from numba threads either way.

## Usage

```python
import numpy as np
from multigarch import GARCH, CCC, DCC

# Generate sample returns (T periods, n assets)
returns = np.random.randn(500, 30) * 0.02

# Univariate GARCH(1,1)
garch = GARCH().fit(returns[:, 0])
garch.forecast(horizon=5)

# GJR-GARCH(1,1,1) with Student-t innovations — the usual choice for equity VaR
gjr = GARCH(o=1, dist="t").fit(returns[:, 0])
gjr.gamma, gjr.nu  # leverage coefficient, degrees of freedom

# CCC: Constant Conditional Correlation (fastest)
ccc = CCC().fit(returns)

# DCC: Dynamic Conditional Correlation
# method="auto" (default) uses the exact likelihood for n <= 25 and the
# pairwise composite likelihood above that
dcc = DCC().fit(returns)
dcc.forecast(horizon=5)  # shape: (5, 30, 30)

# Access results
dcc.H  # time-varying covariances (T, n, n)
dcc.R  # time-varying correlations (T, n, n)
ccc.R  # constant correlation matrix (n, n)

# Configure the univariate models and parallelism
dcc = DCC(p=2, q=1, o=1, dist="t", n_jobs=4).fit(returns)
```

## Models

| Model | Description | Cost per likelihood eval |
|-------|-------------|--------------------------|
| `GARCH(p,q)` | Univariate GARCH / GJR-GARCH (`o > 0`) | O(T) |
| `CCC(p,q)` | Constant Conditional Correlation | no correlation optimization |
| `DCC(p,q)` | Dynamic Conditional Correlation | O(T·n³) full / O(T·n) composite |

## Options

- `p`, `q`: ARCH / GARCH orders (default 1)
- `o`: asymmetric (GJR) order, default 0. Adds `γ·ε²·1[ε<0]` terms, γ ≥ 0;
  stationarity `Σα + Σβ + ½Σγ < 0.999` is enforced
- `dist`: `"normal"` (default) or `"t"` (Student-t, degrees of freedom `nu`
  estimated in [2.1, 200]). In CCC/DCC this sets the margins; the
  correlation stage is a Gaussian quasi-likelihood either way
- `mean`: `"zero"` (default) or `"constant"` (demean, stores `mu`)
- `n_jobs`: threads for the numba kernels, -1 = all cores (CCC/DCC)
- `low_memory`: store only the final covariance/correlation instead of the
  full (T, n, n) paths. Storing both R and H paths costs `2·T·n²·8` bytes —
  about 8 GB at T=2000, n=500 — so use `low_memory=True` at large n.
- `method` (DCC): `"full"` exact likelihood, `"cl"` pairwise composite
  likelihood (recommended for large n: less biased and dramatically faster),
  `"auto"` (default) picks `"cl"` when n > 25.
- `shrinkage` (CCC/DCC): shrink the correlation target toward identity,
  `Q̄ ← (1−δ)·S + δ·I`. Use δ > 0 when T is not much larger than n.

## Diagnostics

After fitting: `loglik_`, `aic`, `bic`, `summary()` on all models;
`std_err` on `GARCH` (per-parameter) and `DCC` (for `a`, `b`; approximate —
no sandwich correction). Non-convergence emits a `RuntimeWarning`.

## Benchmarks

All timings on an Apple M4 Pro (10 performance + 4 efficiency cores),
best of 3.

`uv run python benchmarks/bench.py` — simulated returns, T=2000,
`low_memory=True`, times in seconds:

| n | CCC | DCC (cl) | DCC (full) |
|---|-----|----------|------------|
| 10 | 0.002 | 0.003 | 0.008 |
| 50 | 0.006 | 0.008 | 0.037 |
| 100 | 0.016 | 0.019 | 0.18 |
| 300 | 0.036 | 0.043 | — |

The full likelihood is O(T·n³) per optimizer evaluation (parallel over time
blocks) and is skipped above n=100; the composite likelihood scales linearly
in n.

On real data (CRSP daily returns, 500 most liquid common shares with
complete 2019–2023 history, T=1258 — see `benchmarks/bench_crsp.py`):
`DCC(method="cl")` fits n=500 in ~0.06 s on all cores (~0.4 s on one)
with a=0.025, b=0.930; with GJR-t margins in ~0.13 s. `loglik_` (and so
`summary()`) evaluates the full likelihood once — ~0.8 s at n=500.

## Comparison

**R `rmgarch`** (the reference DCC implementation) — `benchmarks/bench_vs_rmgarch.R`
and `.py`: two-step DCC(1,1), GARCH(1,1)-normal margins, full likelihood,
first n CRSP stocks above, seconds:

| n | rmgarch | multigarch, 1 thread | multigarch, all cores | a, b rmgarch | a, b multigarch |
|---|---|---|---|---|---|
| 5 | 0.55 | 0.024 | 0.026 | 0.0170, 0.9457 | 0.0169, 0.9460 |
| 10 | 0.75 | 0.063 | 0.036 | 0.0121, 0.9703 | 0.0128, 0.9686 |
| 30 | 2.21 | 0.36 | 0.068 | 0.0069, 0.9659 | 0.0070, 0.9646 |
| 50 | 4.99 | 0.91 | 0.13 | 0.0054, 0.9438 | 0.0060, 0.9345 |
| 100 | 32.4 | 4.0 | 0.50 | 0.0056, 0.8174 | 0.0058, 0.8120 |

Estimates agree; rmgarch's (a, b) never scores better than multigarch's
on multigarch's second-stage likelihood. rmgarch has no composite
likelihood, so n=500 is out of its reach in practice.

**Python `arch`** (univariate only; no multivariate models) —
`benchmarks/bench_vs_arch.py`, 500 CRSP stocks: multigarch fits
GARCH(1,1)-normal in 0.7 ms vs 4.3 ms and GJR(1,1,1)-t in 2.0 ms vs
9.2 ms per series (and fits all series in parallel inside CCC/DCC). The
packages initialize σ² differently, so each fit is scored on both
likelihoods: arch's estimates beat multigarch's on multigarch's own
likelihood only on series where arch leaves multigarch's parameter set
(γ < 0, persistence above 0.999).

## License

MIT
