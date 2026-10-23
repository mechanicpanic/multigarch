r"""Multivariate GARCH models for Python: GARCH/GJR-GARCH, CCC, DCC and ADCC, with VaR.

multigarch fits univariate GARCH and GJR-GARCH models (normal or Student-t
innovations) and multivariate correlation models on top of them —
constant (CCC), dynamic (DCC) and asymmetric dynamic (ADCC) conditional
correlation. Estimation runs
in numba, in parallel across assets, and DCC scales to hundreds of assets
through a composite-likelihood estimator. Value at risk, expected
shortfall and VaR backtesting are built in.

It is pure Python + numba: `pip install multigarch` needs no compiler,
on any platform, including free-threaded Python (3.13t/3.14t).

## Quickstart

```python
import numpy as np
from multigarch import GARCH, DCC
from multigarch.backtest import rolling_var

returns = np.random.default_rng(0).standard_normal((1500, 10)) * 0.01  # (T, n)

# One asset: GJR-GARCH(1,1,1) with Student-t innovations
g = GARCH(o=1, dist="t").fit(returns[:, 0])
print(g.summary())
g.forecast(5)                 # variance forecasts, 5 periods
g.value_at_risk(0.99)         # 1-day 99% VaR (positive loss)
g.expected_shortfall(0.975)

# Many assets: GJR-t margins + DCC correlations
dcc = DCC(o=1, dist="t", low_memory=True).fit(returns)
dcc.a, dcc.b                  # correlation dynamics
dcc.forecast(5)               # (5, n, n) covariance forecasts
w = np.full(10, 0.1)
dcc.portfolio_value_at_risk(w, 0.99)

# Out-of-sample VaR backtest
bt = rolling_var(returns[:, 0], GARCH(o=1, dist="t"), window=1000)
print(bt.summary())           # hit rate, Kupiec, Christoffersen
```

## Models

### Univariate: `GARCH`

$$\varepsilon_t = \sigma_t z_t, \qquad
\sigma^2_t = \omega + \sum_{i=1}^{p} \alpha_i \varepsilon^2_{t-i}
           + \sum_{k=1}^{o} \gamma_k \varepsilon^2_{t-k}\,\mathbb{1}[\varepsilon_{t-k} < 0]
           + \sum_{j=1}^{q} \beta_j \sigma^2_{t-j}$$

- `p`, `q` — ARCH and GARCH orders (default 1).
- `o` — GJR asymmetry order (Glosten, Jagannathan & Runkle, 1993). With
  $\gamma > 0$ a negative shock raises next-period variance by
  $\alpha + \gamma$ times its square, a positive one by $\alpha$ — the
  leverage effect seen in equity returns. Default 0 (plain GARCH).
- `dist` — `"normal"` or `"t"`: $z_t$ standard normal or Student-t
  standardized to unit variance, with degrees of freedom $\nu$ estimated
  (Bollerslev, 1987). Fat tails matter for VaR: normal quantiles
  understate the tails of daily returns.
- `mean` — `"zero"` (default) or `"constant"` (sample mean, stored as `mu`).

Constraints: $\omega > 0$; $\alpha, \gamma, \beta \ge 0$; persistence
$\sum\alpha + \sum\beta + \tfrac12\sum\gamma \le 0.999$; $\nu \in [2.1, 200]$.

### Multivariate: `CCC` and `DCC`

Both fit a univariate model to every column (same `p`, `q`, `o`, `dist`,
`mean`), standardize the residuals $e_t = \varepsilon_t / \sigma_t$, and
model their correlation $R_t$; the covariance is
$H_t = D_t R_t D_t$ with $D_t = \mathrm{diag}(\sigma_{1,t}, \dots, \sigma_{n,t})$.

- **CCC** (Bollerslev, 1990): $R_t = R$, the sample correlation of $e_t$.
  No second-stage optimization; the fastest choice.
- **DCC** (Engle, 2002):
  $$Q_t = (1 - a - b)\,\bar{Q} + a\, e_{t-1} e_{t-1}' + b\, Q_{t-1}, \qquad
    R_t = \mathrm{diag}(Q_t)^{-1/2}\, Q_t\, \mathrm{diag}(Q_t)^{-1/2}$$
  with correlation targeting ($\bar{Q}$ = sample correlation of $e_t$).
- **ADCC** (Cappiello, Engle & Sheppard, 2006) — `DCC(asymmetric=True)`:
  $$Q_t = (1 - a - b)\,\bar{Q} - g\,\bar{N} + a\, e_{t-1} e_{t-1}'
    + g\, n_{t-1} n_{t-1}' + b\, Q_{t-1}$$
  with $n_t = e_t \odot \mathbb{1}[e_t < 0]$ and $\bar{N}$ their sample
  second moment $\mathbb{E}[n_t n_t']$. With $g > 0$, assets that fall
  together become more correlated than assets that rise together — the
  crisis effect that matters for portfolio VaR. Stationarity:
  $a + b + \delta g < 1$, $\delta$ the largest eigenvalue of
  $\bar{Q}^{-1/2}\bar{N}\bar{Q}^{-1/2}$. (R's `rmgarch` uses the
  *centered* covariance of $n_t$ for $\bar{N}$, which does not keep
  $\mathbb{E}[Q_t] = \bar{Q}$; its estimates differ accordingly.)

  A single scalar $g$ for many assets has a limit: $\delta$ grows with
  $n$ (the negative parts share a common mean), so the bound caps $g$ ever
  harder. On 500 CRSP stocks, composite-likelihood ADCC improves the
  (full-likelihood) AIC over DCC up to $n \approx 100$, but not at
  $n = 200$ or $500$, where the bound binds. A `RuntimeWarning` flags a
  binding bound; at large $n$ compare `aic` with the symmetric model.

DCC's second stage is estimated by one of:

- `method="full"` — the Gaussian likelihood of $e_t$, $O(T n^3)$ per
  evaluation (one Cholesky per period; parallel over time blocks).
- `method="cl"` — the pairwise composite likelihood over contiguous pairs
  $(i, i+1)$ (Engle, Shephard & Sheppard, 2008), $O(T n)$. The full
  likelihood is badly biased toward small $a$ and $b$ when $n$ is large
  relative to $T$; the composite likelihood is not.
- `method="auto"` (default) — `"cl"` for $n > 25$, `"full"` otherwise.

`shrinkage=δ` replaces the correlation target by $(1-δ)\,\bar{Q} + δ I$ —
use it when $T$ is not much larger than $n$ and the sample correlation is
ill-conditioned.

Estimation is two-step (volatilities, then correlations), so the
reported `loglik_` is a quasi-likelihood: univariate log-likelihoods plus
the Gaussian correlation term.

### Which model?

| Situation | Model |
|---|---|
| one asset, variance or VaR | `GARCH(o=1, dist="t")` (equities); `GARCH()` as a baseline |
| several assets, correlations roughly stable | `CCC(o=1, dist="t")` |
| correlations move (crises, regime changes) | `DCC(o=1, dist="t")` |
| correlations jump when markets fall together | `DCC(o=1, dist="t", asymmetric=True)` (ADCC) |
| hundreds of assets | `DCC(method="cl", low_memory=True)` |

Compare specifications with `aic` / `bic`, and above all with an
out-of-sample VaR backtest (`multigarch.backtest`).

## Forecasting

`forecast(h)` returns variance forecasts (`GARCH`, shape `(h,)`) or
covariance forecasts (`CCC`/`DCC`, shape `(h, n, n)`). Future squared
shocks are replaced by their expectations, with
$\mathbb{E}[\varepsilon^2 \mathbb{1}(\varepsilon<0)] = \sigma^2/2$ for the
GJR terms. DCC's first step uses the last observed residual; later steps
revert to $\bar{Q}$ at rate $a+b$ (Engle & Sheppard, 2001).

## Value at risk and expected shortfall

All risk measures are one-period, for each of the next `horizon` periods,
returned as positive losses in return units:

- `GARCH.value_at_risk(level, horizon)` —
  $-(\mu + \sigma_{T+h}\, q_{1-\text{level}})$, with $q$ the quantile of the
  fitted innovation distribution.
- `GARCH.expected_shortfall(level, horizon)` — mean loss beyond the VaR,
  closed form for both distributions.
- `CCC/DCC.portfolio_value_at_risk(weights, level, horizon)` and
  `portfolio_expected_shortfall` — for normal margins in closed form from
  $\sqrt{w' H_{T+h} w}$; for Student-t margins by Monte Carlo with a
  Gaussian copula (correlated normals with the forecast $R_{T+h}$, mapped
  to each asset's fitted t). `n_sims` (default 20 000) controls the
  Monte Carlo error, about 1% at the default.

## Backtesting

`multigarch.backtest.rolling_var` produces out-of-sample one-step-ahead
VaR (and ES) forecasts over a rolling or expanding window, re-estimating
every `refit_every` steps, and reports the Kupiec (1995) and Christoffersen
(1998) coverage tests. See `multigarch.backtest`.

## Estimation details

- Each series is rescaled to unit variance before optimization ($\omega$
  of daily returns is ~1e-6 next to $\alpha, \beta$ ~0.1, which stalls
  optimizers); estimates are mapped back.
- Univariate models are fit by projected BFGS with analytic gradients,
  from two starting points (more for higher orders, whose likelihoods are
  often multimodal); a constrained SLSQP fit is the fallback.
- The variance recursion starts from the sample variance.
- `std_err` are inverse-Hessian standard errors (DCC: second stage only,
  no correction for first-stage estimation error) — indicative.

## Performance

- `n_jobs` sets the number of numba threads (default: all cores); results
  do not depend on it. Parallelism works with or without the GIL.
- `low_memory=True` keeps only the final covariance and correlation
  instead of `(T, n, n)` paths ($2 T n^2 \cdot 8$ bytes — 8 GB at T=2000,
  n=500).
- On an Apple M4 Pro, DCC with the composite likelihood fits 500 assets ×
  1258 days of CRSP returns in ~0.06 s (GJR-t margins: ~0.13 s), and is
  20–70× faster than R's `rmgarch` with the full likelihood at n ≤ 100.
  Benchmarks and comparisons: see the project README.

## References

- Bollerslev, T. (1987). A conditionally heteroskedastic time series model
  for speculative prices and rates of return. *Review of Economics and
  Statistics*, 69(3), 542–547.
- Bollerslev, T. (1990). Modelling the coherence in short-run nominal
  exchange rates: a multivariate generalized ARCH model. *Review of
  Economics and Statistics*, 72(3), 498–505.
- Cappiello, L., Engle, R. F., & Sheppard, K. (2006). Asymmetric dynamics
  in the correlations of global equity and bond returns. *Journal of
  Financial Econometrics*, 4(4), 537–572.
- Christoffersen, P. F. (1998). Evaluating interval forecasts.
  *International Economic Review*, 39(4), 841–862.
- Engle, R. (2002). Dynamic conditional correlation: a simple class of
  multivariate GARCH models. *Journal of Business & Economic Statistics*,
  20(3), 339–350.
- Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties
  of dynamic conditional correlation multivariate GARCH. NBER Working
  Paper 8554.
- Engle, R. F., Shephard, N., & Sheppard, K. (2008). Fitting vast
  dimensional time-varying covariance models. Working paper, University
  of Oxford.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation
  between the expected value and the volatility of the nominal excess
  return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk
  measurement models. *Journal of Derivatives*, 3(2), 73–84.

Project home: https://github.com/mechanicpanic/multigarch
"""

from multigarch.models import CCC, DCC, GARCH

__all__ = ["GARCH", "CCC", "DCC"]
__version__ = "0.4.0"
