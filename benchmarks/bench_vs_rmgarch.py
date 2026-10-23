"""Compare DCC(1,1) fits against R's rmgarch.

    Rscript benchmarks/bench_vs_rmgarch.R panel.csv 5,10,30,50,100 > rmgarch.txt
    uv run python benchmarks/bench_vs_rmgarch.py panel.csv rmgarch.txt

For the asymmetric model pass ``aDCC`` to the R script; the R output then
carries ``g`` and the Python side fits ``DCC(asymmetric=True)``.

Both sides fit two-step DCC(1,1) with zero-mean GARCH(1,1)-normal margins
and the full likelihood (rmgarch has no composite-likelihood estimator).
The last column scores rmgarch's (a, b) on multigarch's second-stage
objective: >= 0 means multigarch's optimum is at least as good there.
"""

import re
import sys
import time

import numpy as np

from multigarch import DCC
from multigarch._jit import dcc_loglik_loop


def best_time(fn, repeat=3):
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> None:
    panel = np.loadtxt(sys.argv[1], delimiter=",")
    pattern = r"RESULT n=(\d+) time=([\d.]+) a=([\d.e-]+) b=([\d.e-]+)(?: g=([\d.e-]+))?"
    r_fits = {
        int(m[1]): (float(m[2]), float(m[3]), float(m[4]), float(m[5] or 0.0))
        for m in (re.match(pattern, line) for line in open(sys.argv[2]))
        if m
    }
    asym = "aDCC" in sys.argv[3:] or any(g > 0 for *_, g in r_fits.values())
    kw = {"method": "full", "low_memory": True, "asymmetric": asym}
    DCC(**kw).fit(panel[:200, :3])  # JIT warm-up
    par = "a, b, g" if asym else "a, b"
    print(
        f"| n | rmgarch (s) | multigarch, 1 thread (s) | multigarch, all cores (s) "
        f"| {par} rmgarch | {par} multigarch | NLL gap |"
    )
    print("|---|---|---|---|---|---|---|")
    for n, (t_r, a_r, b_r, g_r) in sorted(r_fits.items()):
        r = panel[:, :n]
        t1, _ = best_time(lambda r=r: DCC(**kw, n_jobs=1).fit(r))
        tn, m = best_time(lambda r=r: DCC(**kw).fit(r))
        e, Qb, Nb = m._std_resid, m.Q_bar, m.N_bar
        gap = dcc_loglik_loop(e, Qb, Nb, a_r, b_r, g_r) - dcc_loglik_loop(e, Qb, Nb, m.a, m.b, m.g)
        ours = f"{m.a:.4f}, {m.b:.4f}" + (f", {m.g:.4f}" if asym else "")
        theirs = f"{a_r:.4f}, {b_r:.4f}" + (f", {g_r:.4f}" if asym else "")
        print(f"| {n} | {t_r:.2f} | {t1:.3f} | {tn:.3f} | {theirs} | {ours} | {gap:+.3f} |")


if __name__ == "__main__":
    main()
