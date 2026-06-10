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
