"""Value-at-risk workflow: choose a volatility model, forecast VaR, backtest it.

    uv run python examples/var_backtest.py              # simulated data
    uv run python examples/var_backtest.py --npy r.npy  # your (T, n) returns

1. Compare GARCH / GJR-GARCH with normal and Student-t innovations on one
   asset by AIC.
2. Backtest their 99% one-day VaR out of sample (rolling window).
3. Fit DCC with GJR-t margins to all assets, test for asymmetric
   correlations (ADCC), and forecast portfolio VaR and expected
   shortfall for the next days.
4. Backtest the portfolio VaR.
"""

import argparse
import time

import numpy as np

from multigarch import DCC, GARCH
from multigarch.backtest import rolling_var

SPECS = {
    "GARCH-normal": {},
    "GARCH-t": {"dist": "t"},
    "GJR-normal": {"o": 1},
    "GJR-t": {"o": 1, "dist": "t"},
}


def simulate(T: int, n: int, seed: int = 0) -> np.ndarray:
    """Equity-like returns: GJR volatility, t(5) shocks, one common factor.

    The factor loading drifts slowly, so correlations change over time.
    """
    rng = np.random.default_rng(seed)
    nu = 5.0
    factor = rng.standard_normal(T)
    loading = 0.5 + 0.35 * np.sin(2 * np.pi * np.arange(T) / 400)
    out = np.empty((T, n))
    for i in range(n):
        idio = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
        z = loading * factor + np.sqrt(1 - loading**2) * idio
        s2 = 2e-4
        for t in range(T):
            out[t, i] = np.sqrt(s2) * z[t]
            s2 = 2e-6 + 0.02 * out[t, i] ** 2 + 0.12 * out[t, i] ** 2 * (out[t, i] < 0) + 0.90 * s2
    return out


def main(returns: np.ndarray, window: int = 1000, level: float = 0.99, refit_every: int = 20):
    T, n = returns.shape
    r0 = returns[:, 0]

    print(f"Data: T={T}, n={n}\n\n1. Model comparison on asset 0 (full sample)")
    print(f"   {'model':14s} {'loglik':>10s} {'AIC':>10s}  extras")
    for name, kw in SPECS.items():
        m = GARCH(**kw).fit(r0)
        extras = []
        if m.o:
            extras.append(f"gamma={m.gamma[0]:.3f}")
        if m.nu:
            extras.append(f"nu={m.nu:.1f}")
        print(f"   {name:14s} {m.loglik_:10.1f} {m.aic:10.1f}  {' '.join(extras)}")

    print(f"\n2. {level:.0%} VaR backtest on asset 0, window={window}, refit every {refit_every}")
    header = ["hits", "rate", "Kupiec p", "indep. p", "cond.cov p"]
    widths = [5, 7, 9, 9, 11]
    print(
        f"   {'model':14s} " + " ".join(f"{h:>{w}s}" for h, w in zip(header, widths, strict=True))
    )
    for name, kw in SPECS.items():
        bt = rolling_var(r0, GARCH(**kw), window=window, level=level, refit_every=refit_every)
        uc = bt.kupiec()
        ind, cc = bt.christoffersen()
        print(
            f"   {name:14s} {int(bt.hits.sum()):5d} {bt.hit_rate:7.4f} "
            f"{uc.pvalue:9.3f} {ind.pvalue:9.3f} {cc.pvalue:11.3f}"
        )
    print(f"   (expected rate {1 - level:.4f}; small p-values reject the model)")

    weights = np.full(n, 1.0 / n)
    t0 = time.perf_counter()
    dcc = DCC(o=1, dist="t", low_memory=True).fit(returns)
    print(f"\n3. DCC with GJR-t margins, equal-weight portfolio ({time.perf_counter() - t0:.2f} s)")
    print(f"   a={dcc.a:.4f}  b={dcc.b:.4f}  AIC={dcc.aic:.1f}")
    adcc = DCC(o=1, dist="t", low_memory=True, asymmetric=True).fit(returns)
    better = "ADCC" if adcc.aic < dcc.aic else "DCC"
    print(f"   ADCC: g={adcc.g:.4f}  AIC={adcc.aic:.1f}  -> {better} preferred")
    var = dcc.portfolio_value_at_risk(weights, level, horizon=5, seed=1)
    es = dcc.portfolio_expected_shortfall(weights, 0.975, horizon=5, seed=1)
    for h in range(5):
        print(f"   day +{h + 1}: VaR({level:.0%}) = {var[h]:.4%}   ES(97.5%) = {es[h]:.4%}")

    print(f"\n4. Portfolio VaR backtest (DCC GJR-t), window={window}, refit every {refit_every}")
    bt = rolling_var(
        returns,
        DCC(o=1, dist="t", low_memory=True),
        window=window,
        level=level,
        weights=weights,
        refit_every=refit_every,
        n_sims=5_000,
    )
    print("   " + bt.summary().replace("\n", "\n   "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", help="(T, n) array of returns")
    args = parser.parse_args()
    main(np.load(args.npy) if args.npy else simulate(2000, 5))
