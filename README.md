# multigarch

Multivariate GARCH models for Python: GARCH / GJR-GARCH, CCC and DCC, with
value at risk and VaR backtesting.

- **DCC at hundreds of assets** — composite-likelihood estimation, 500
  stocks in ~0.06 s
- **GJR asymmetry and Student-t innovations** for the volatility models
- **VaR and expected shortfall** for single assets and portfolios, plus
  rolling backtests with Kupiec and Christoffersen tests
- **Fast**: numba kernels, parallel across assets, with or without the GIL;
  20–70× faster than R's `rmgarch` on DCC
- **Installs anywhere**: pure Python + numba, no compiler needed —
  including free-threaded Python (3.13t/3.14t)

Documentation: https://mechanicpanic.github.io/multigarch/

## Installation

```bash
pip install multigarch
```

or the development version from GitHub:

```bash
pip install git+https://github.com/mechanicpanic/multigarch
```

## Quickstart

```python
import numpy as np
from multigarch import GARCH, CCC, DCC
from multigarch.backtest import rolling_var

returns = np.random.default_rng(0).standard_normal((1500, 10)) * 0.01  # (T, n)

# One asset: GJR-GARCH(1,1,1) with Student-t innovations
g = GARCH(o=1, dist="t").fit(returns[:, 0])
print(g.summary())
g.forecast(5)                  # variance forecasts
g.value_at_risk(0.99)          # 1-day 99% VaR, as a positive loss
g.expected_shortfall(0.975)

# Many assets: GJR-t margins + dynamic correlations
dcc = DCC(o=1, dist="t", low_memory=True).fit(returns)
dcc.a, dcc.b                   # correlation dynamics
dcc.forecast(5)                # (5, 10, 10) covariance forecasts
w = np.full(10, 0.1)
dcc.portfolio_value_at_risk(w, 0.99)

# Constant correlations: the fastest multivariate model
ccc = CCC().fit(returns)

# Out-of-sample VaR backtest
bt = rolling_var(returns[:, 0], GARCH(o=1, dist="t"), window=1000)
print(bt.summary())            # hit rate, Kupiec, Christoffersen
```

A complete VaR workflow — model comparison, portfolio VaR, backtest — is in
[`examples/var_backtest.py`](examples/var_backtest.py).

## Models

| Model | What it models |
|---|---|
| `GARCH(p, q, o=0, dist="normal")` | one series; `o > 0` adds GJR asymmetry (leverage effect), `dist="t"` Student-t innovations |
| `CCC(...)` | univariate models per asset + constant correlation |
| `DCC(...)` | univariate models per asset + dynamic correlation (Engle, 2002) |

Main options (CCC and DCC take the univariate ones too):

| Option | Meaning |
|---|---|
| `p`, `q`, `o` | ARCH, GARCH and GJR orders (default 1, 1, 0) |
| `dist` | `"normal"` or `"t"` (degrees of freedom estimated) |
| `mean` | `"zero"` or `"constant"` |
| `method` (DCC) | `"full"` likelihood, `"cl"` composite likelihood, `"auto"`: `"cl"` for n > 25 |
| `shrinkage` | shrink the correlation target toward identity; use when T is not ≫ n |
| `low_memory` | keep only the final covariance, not the (T, n, n) path |
| `n_jobs` | numba threads, -1 = all cores |

After fitting: `loglik_`, `aic`, `bic`, `summary()`, `forecast()`;
`std_err` on `GARCH` and `DCC`; `value_at_risk` / `expected_shortfall`
(`GARCH`) and `portfolio_value_at_risk` / `portfolio_expected_shortfall`
(`CCC`, `DCC`). The [documentation](https://mechanicpanic.github.io/multigarch/)
has the model equations, estimation details and references.

## Comparison

**R `rmgarch`** — the reference DCC implementation. Two-step DCC(1,1) with
GARCH(1,1)-normal margins and the full likelihood, on the first n of the
CRSP stocks below (`benchmarks/bench_vs_rmgarch.R` and `.py`), seconds:

| n | rmgarch | multigarch, 1 thread | multigarch, all cores | a, b rmgarch | a, b multigarch |
|---|---|---|---|---|---|
| 5 | 0.55 | 0.024 | 0.026 | 0.0170, 0.9457 | 0.0169, 0.9460 |
| 10 | 0.75 | 0.063 | 0.036 | 0.0121, 0.9703 | 0.0128, 0.9686 |
| 30 | 2.21 | 0.36 | 0.068 | 0.0069, 0.9659 | 0.0070, 0.9646 |
| 50 | 4.99 | 0.91 | 0.13 | 0.0054, 0.9438 | 0.0060, 0.9345 |
| 100 | 32.4 | 4.0 | 0.50 | 0.0056, 0.8174 | 0.0058, 0.8120 |

Estimates agree, and rmgarch's (a, b) never score better than multigarch's
on multigarch's second-stage likelihood. rmgarch has no composite
likelihood, which puts n = 500 out of practical reach.

**Python `arch`** — univariate models only (no multivariate). On 500 CRSP
stocks (`benchmarks/bench_vs_arch.py`) multigarch fits GARCH(1,1)-normal in
0.7 ms vs 4.3 ms and GJR(1,1,1)-t in 2.0 ms vs 9.2 ms per series, and fits
all series in parallel inside CCC/DCC. The packages initialize σ²
differently, so each fit is scored on both likelihoods: arch's estimates
beat multigarch's on multigarch's own likelihood only where arch leaves
multigarch's parameter set (γ < 0 or persistence above 0.999). arch ships
a compiled extension; on free-threaded Python it has no wheels and builds
from source, and it re-enables the GIL on import.

## Benchmarks

Apple M4 Pro (10 performance + 4 efficiency cores), best of 3.

`uv run python benchmarks/bench.py` — simulated returns, T=2000,
`low_memory=True`, seconds:

| n | CCC | DCC (cl) | DCC (full) |
|---|-----|----------|------------|
| 10 | 0.002 | 0.003 | 0.008 |
| 50 | 0.006 | 0.008 | 0.037 |
| 100 | 0.016 | 0.019 | 0.18 |
| 300 | 0.036 | 0.043 | — |

The full likelihood costs O(T·n³) per optimizer evaluation (parallel over
time blocks) and is skipped above n=100; the composite likelihood scales
linearly in n.

Real data — CRSP daily returns, the 500 most liquid common shares with a
complete 2019–2023 history, T=1258 (`benchmarks/bench_crsp.py`):
`DCC(method="cl")` fits n=500 in ~0.06 s on all cores (~0.4 s on one) with
a=0.025, b=0.930; with GJR-t margins in ~0.13 s. `loglik_` (and so
`summary()`) evaluates the full likelihood once — ~0.8 s at n=500.

## License

MIT
