r"""Rolling VaR forecasts and coverage tests.

```python
from multigarch import GARCH
from multigarch.backtest import rolling_var

bt = rolling_var(returns, GARCH(o=1, dist="t"), window=1000, level=0.99)
print(bt.summary())
bt.kupiec()           # unconditional coverage: is the hit rate 1%?
bt.christoffersen()   # are hits independent, and is coverage right?
```

A *hit* (exceedance) at time $t$ is a realized return below the VaR
forecast made with data up to $t-1$: $r_t < -\mathrm{VaR}_t$.

- **Kupiec (1995)**, proportion of failures: likelihood-ratio test that
  the hit rate equals $1 - \text{level}$; $\chi^2_1$.
- **Christoffersen (1998)**: independence of hits against a first-order
  Markov alternative ($\chi^2_1$; detects clustered exceedances), and
  conditional coverage, the sum of both statistics ($\chi^2_2$).

`kupiec_test` and `christoffersen_test` accept any boolean hit series, so
they also work for VaR forecasts from other sources.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from multigarch.models import CCC, DCC, GARCH

__all__ = ["TestResult", "VaRBacktest", "christoffersen_test", "kupiec_test", "rolling_var"]


class TestResult(NamedTuple):
    """Likelihood-ratio statistic and its asymptotic p-value."""

    __test__ = False  # not a pytest test class despite the name

    statistic: float
    pvalue: float


def _hits_array(hits) -> NDArray[np.bool_]:
    h = np.asarray(hits, dtype=bool).reshape(-1)
    if h.size == 0:
        raise ValueError("hits is empty")
    return h


def _nonneg(lr: float) -> float:
    """Clip rounding-level negative LR statistics (and -0.0) to 0."""
    lr = float(lr)
    return lr if lr > 0.0 else 0.0


def kupiec_test(hits, level: float) -> TestResult:
    """Kupiec (1995) proportion-of-failures test of unconditional coverage.

    Args:
        hits: boolean exceedance series
        level: VaR confidence level, e.g. 0.99 (expected hit rate 0.01)

    Returns:
        TestResult(statistic, pvalue), statistic ~ chi2(1) under correct coverage
    """
    h = _hits_array(hits)
    p = 1.0 - level
    n, x = h.size, int(h.sum())
    pi = x / n
    ll_null = special.xlogy(n - x, 1.0 - p) + special.xlogy(x, p)
    ll_alt = special.xlogy(n - x, 1.0 - pi) + special.xlogy(x, pi)
    lr = _nonneg(-2.0 * (ll_null - ll_alt))
    return TestResult(lr, float(stats.chi2.sf(lr, 1)))


def christoffersen_test(hits, level: float) -> tuple[TestResult, TestResult]:
    """Christoffersen (1998) independence and conditional-coverage tests.

    Independence compares a first-order Markov chain for the hits with an
    i.i.d. one; conditional coverage adds the Kupiec statistic.

    Args:
        hits: boolean exceedance series
        level: VaR confidence level, e.g. 0.99

    Returns:
        (independence ~ chi2(1), conditional coverage ~ chi2(2))
    """
    h = _hits_array(hits).astype(np.int64)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total else 0.0
    ll_null = special.xlogy(n00 + n10, 1.0 - pi) + special.xlogy(n01 + n11, pi)
    ll_alt = (
        special.xlogy(n00, 1.0 - pi0)
        + special.xlogy(n01, pi0)
        + special.xlogy(n10, 1.0 - pi1)
        + special.xlogy(n11, pi1)
    )
    lr_ind = _nonneg(-2.0 * (ll_null - ll_alt))
    lr_cc = kupiec_test(h, level).statistic + lr_ind
    return (
        TestResult(lr_ind, float(stats.chi2.sf(lr_ind, 1))),
        TestResult(lr_cc, float(stats.chi2.sf(lr_cc, 2))),
    )


@dataclass
class VaRBacktest:
    """Result of ``rolling_var``: one-step-ahead forecasts and realized returns.

    Attributes:
        level: VaR confidence level
        var: VaR forecasts, positive losses, shape (n_test,)
        realized: realized (portfolio) returns at the same times
        index: positions of the forecast times in the input series
        es: expected-shortfall forecasts at ``es_level`` (None if not requested)
        es_level: confidence level of ``es``
    """

    level: float
    var: NDArray[np.float64]
    realized: NDArray[np.float64]
    index: NDArray[np.intp]
    es: NDArray[np.float64] | None = None
    es_level: float | None = None

    @property
    def hits(self) -> NDArray[np.bool_]:
        """Exceedances: realized return below the negative VaR."""
        return self.realized < -self.var

    @property
    def hit_rate(self) -> float:
        return float(self.hits.mean())

    @property
    def expected_rate(self) -> float:
        return 1.0 - self.level

    def kupiec(self) -> TestResult:
        """Kupiec unconditional-coverage test; see ``kupiec_test``."""
        return kupiec_test(self.hits, self.level)

    def christoffersen(self) -> tuple[TestResult, TestResult]:
        """Christoffersen independence and conditional-coverage tests."""
        return christoffersen_test(self.hits, self.level)

    def summary(self) -> str:
        """Formatted backtest report."""
        uc = self.kupiec()
        ind, cc = self.christoffersen()
        n = self.var.size
        lines = [
            f"VaR backtest  level={self.level}  n={n}",
            f"  hits: {int(self.hits.sum())} (rate {self.hit_rate:.4f}, "
            f"expected {self.expected_rate:.4f})",
            f"  Kupiec POF:              LR={uc.statistic:7.3f}  p={uc.pvalue:.4f}",
            f"  Christoffersen indep.:   LR={ind.statistic:7.3f}  p={ind.pvalue:.4f}",
            f"  Christoffersen cond.cov: LR={cc.statistic:7.3f}  p={cc.pvalue:.4f}",
        ]
        if self.es is not None:
            lines.append(f"  mean ES({self.es_level}): {self.es.mean():.6g}")
        return "\n".join(lines)


def rolling_var(
    returns: NDArray[np.float64],
    model: GARCH | CCC | DCC,
    window: int,
    level: float = 0.99,
    refit_every: int = 1,
    weights: NDArray[np.float64] | None = None,
    expanding: bool = False,
    es_level: float | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
) -> VaRBacktest:
    """Out-of-sample one-step-ahead VaR forecasts over a rolling window.

    For each time t from ``window`` to the end, the model is fitted to
    the preceding ``window`` observations (or all of them if
    ``expanding``) and forecasts the VaR for t. Between refits, every
    ``refit_every`` steps, the last estimates are kept and only run over
    the new window — much faster, and how VaR models are typically
    operated.

    Args:
        returns: (T,) returns for a univariate ``GARCH`` model, or (T, n)
            asset returns for ``CCC``/``DCC`` together with ``weights``
        model: an unfitted model used as a template (it is copied for
            each refit; its own state is not changed)
        window: estimation window length
        level: VaR confidence level
        refit_every: re-estimate every this many steps
        weights: portfolio weights, shape (n,), for multivariate models
        expanding: use all data up to t instead of a fixed-length window
        es_level: if given, also forecast expected shortfall at this level
        n_sims: Monte Carlo draws per step (portfolio with t margins only)
        seed: base random seed for the Monte Carlo draws

    Returns:
        VaRBacktest with forecasts, realized returns, and coverage tests
    """
    returns = np.asarray(returns, dtype=np.float64)
    univariate = isinstance(model, GARCH)
    if univariate:
        if returns.ndim != 1:
            raise ValueError("a GARCH model needs 1-D returns")
    else:
        if returns.ndim != 2 or weights is None:
            raise ValueError("CCC/DCC need (T, n) returns and portfolio weights")
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape != (returns.shape[1],):
            raise ValueError(f"weights must have shape ({returns.shape[1]},)")
    if refit_every < 1:
        raise ValueError("refit_every must be >= 1")
    T = returns.shape[0]
    if not 0 < window < T:
        raise ValueError(f"window must be in (0, {T}), got {window}")

    index = np.arange(window, T)
    var = np.empty(index.size)
    es = np.empty(index.size) if es_level is not None else None
    fitted = None
    for k, t in enumerate(index):
        data = returns[(0 if expanding else t - window) : t]
        if k % refit_every == 0:
            fitted = copy.deepcopy(model).fit(data)
            current = fitted
        else:
            current = fitted._refilter(data)
        if univariate:
            var[k] = current.value_at_risk(level, 1)[0]
            if es is not None:
                es[k] = current.expected_shortfall(es_level, 1)[0]
        else:
            levels = (level,) if es is None else (level, es_level)
            v, e = current._portfolio_risk(weights, levels, 1, n_sims, seed + k)
            var[k] = v[0, 0]
            if es is not None:
                es[k] = e[1, 0]
    realized = returns[index] if univariate else returns[index] @ weights
    return VaRBacktest(level, var, realized, index, es, es_level)
