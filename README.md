# multigarch

Simple Multivariate GARCH models for Python 3.10+.

Features:
- JIT-compiled inner loops (numba)
- Composite-likelihood DCC estimation — practical at hundreds of assets
- Parallel GARCH fitting (joblib threads, GIL-releasing kernels)
- Configurable GARCH(p,q) orders, diagnostics (loglik/AIC/BIC/std errors)

## Installation

```bash
pip install multigarch
```

## Usage

```python
import numpy as np
from multigarch import GARCH, CCC, DCC

# Generate sample returns (T periods, n assets)
returns = np.random.randn(500, 30) * 0.02

# Univariate GARCH(1,1)
garch = GARCH().fit(returns[:, 0])
garch.forecast(horizon=5)

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

# Configure GARCH orders and parallelism
dcc = DCC(p=2, q=1, n_jobs=4).fit(returns)
```

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

`uv run python benchmarks/bench.py` — Apple Silicon, single run,
T=2000, `low_memory=True`, times in seconds:

| n | CCC | DCC (cl) | DCC (full) |
|---|-----|----------|------------|
| 10 | 0.06 | 0.05 | 0.06 |
| 50 | 0.21 | 0.21 | 0.58 |
| 100 | 0.40 | 0.42 | — |
| 300 | 1.18 | 1.20 | — |

The full likelihood is O(T·n³) per optimizer evaluation and is skipped above
n=50; the composite likelihood scales linearly in n.

## License

MIT
