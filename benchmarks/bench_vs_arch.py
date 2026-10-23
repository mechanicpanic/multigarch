"""Univariate comparison against the `arch` package.

Needs arch (not a dependency):

    uv run --with arch python benchmarks/bench_vs_arch.py [--npy panel.npy]

``--npy`` is a (T, n) array of returns (e.g. the CRSP panel built by
bench_crsp.py); without it a simulated GJR-t panel is used.

The two packages initialize sigma^2 differently (arch backcasts, we use
the sample variance), so their likelihoods differ slightly. Each fit is
therefore scored on both likelihoods: on multigarch's, arch's estimates
show whether multigarch missed a better point of its own objective, and
vice versa. arch also allows gamma < 0 and persistence up to 1, so it
can legitimately win on series outside multigarch's feasible set.
"""

import argparse
import time
import warnings

import numpy as np
from arch import arch_model

from multigarch import GARCH
from multigarch._jit import DIST_NORMAL, DIST_T, garch_loglik


def simulate_panel(T: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((T, n))
    for i in range(n):
        a, g, b, nu = rng.uniform(0.01, 0.06), rng.uniform(0.0, 0.15), 0.88, rng.uniform(4, 12)
        z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
        s2 = 1e-4
        for t in range(T):
            out[t, i] = np.sqrt(s2) * z[t]
            s2 = 1e-6 + a * out[t, i] ** 2 + g * out[t, i] ** 2 * (out[t, i] < 0) + b * s2
    return out


def multigarch_loglik(r: np.ndarray, params: np.ndarray, o: int, dist: str) -> float:
    """multigarch's log-likelihood at raw-scale (omega, alpha, [gamma], beta, [nu])."""
    T = len(r)
    sc = r.std()
    z = r / sc
    x = [params[0] / sc**2, params[1]]
    if o:
        x.append(params[2] / 2.0)
    x.append(params[2 + o])
    if dist == "t":
        x.append(1.0 / params[-1])
    code = DIST_T if dist == "t" else DIST_NORMAL
    ll = -garch_loglik(z, np.array(x), 1, o, 1, code, float(z.var())) - T * np.log(sc)
    return ll - (0.5 * T * np.log(2 * np.pi) if dist == "normal" else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy", help="(T, n) returns array")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    panel = np.load(args.npy) if args.npy else simulate_panel(1500, 200)
    T, n = panel.shape

    for o, dist in [(0, "normal"), (1, "t")]:
        GARCH(o=o, dist=dist).fit(panel[:, 0])  # JIT warm-up
        t_mg = t_arch = 0.0
        on_ours, on_arch, arch_fail, arch_neg_gamma = [], [], 0, 0
        for i in range(n):
            r = panel[:, i]
            t0 = time.perf_counter()
            m = GARCH(o=o, dist=dist).fit(r)
            t_mg += time.perf_counter() - t0
            am = arch_model(100 * r, mean="Zero", p=1, o=o, q=1, dist=dist)
            t0 = time.perf_counter()
            res = am.fit(disp="off", show_warning=False)
            t_arch += time.perf_counter() - t0
            arch_fail += res.convergence_flag != 0
            pa = res.params.to_numpy()
            arch_neg_gamma += bool(o and pa[2] < 0)
            pa_raw = pa.copy()
            pa_raw[0] /= 1e4
            ours = [m.omega * 1e4, m.alpha[0], *m.gamma, m.beta[0]] + ([m.nu] if m.nu else [])
            on_ours.append(m.loglik_ - multigarch_loglik(r, pa_raw, o, dist))
            on_arch.append(am.fix(np.array(ours)).loglikelihood - res.loglikelihood)
        name = "GJR-GARCH(1,1,1)-t" if o else "GARCH(1,1)-normal"
        print(f"\n{name}  n={n}  T={T}")
        print(f"  ms per fit: multigarch {t_mg / n * 1e3:.2f}, arch {t_arch / n * 1e3:.2f}")
        print(
            f"  arch non-converged: {arch_fail}" + (f", gamma < 0: {arch_neg_gamma}" if o else "")
        )
        for label, d in [("multigarch's", on_ours), ("arch's", on_arch)]:
            d = np.array(d)
            print(
                f"  on {label:13s} likelihood, multigarch - arch: median {np.median(d):+.4f}, "
                f"better by >0.1 on {(d > 0.1).sum()}, worse by >0.1 on {(d < -0.1).sum()}"
            )


if __name__ == "__main__":
    main()
