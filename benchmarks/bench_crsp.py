"""DCC benchmark on real CRSP daily returns.

Needs pandas (not a package dependency):

    uv run --with pandas python benchmarks/bench_crsp.py \
        --path /path/to/crsp_daily.csv.gz [--start 2019-01-01] [--end 2023-12-31]

Expects CRSP daily stock file columns: PERMNO, date, RET, SHRCD, EXCHCD,
PRC, VOL. Keeps common shares (SHRCD 10/11) on NYSE/AMEX/NASDAQ
(EXCHCD 1/2/3) with a complete return history over the window, ranked by
median dollar volume.
"""

import argparse
import time

import numpy as np
import pandas as pd

from multigarch import CCC, DCC


def load_returns(path: str, start: str, end: str) -> pd.DataFrame:
    usecols = ["PERMNO", "date", "RET", "SHRCD", "EXCHCD", "PRC", "VOL"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=2_000_000,
        dtype={"RET": "object"},
        parse_dates=["date"],
    ):
        chunk = chunk[(chunk["date"] >= start_ts) & (chunk["date"] <= end_ts)]
        chunk = chunk[chunk["SHRCD"].isin([10, 11]) & chunk["EXCHCD"].isin([1, 2, 3])]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
    df["dollar_vol"] = df["PRC"].abs() * df["VOL"]
    return df


def build_panel(df: pd.DataFrame, n_max: int) -> np.ndarray:
    wide = df.pivot_table(index="date", columns="PERMNO", values="RET")
    full = wide.dropna(axis=1)  # complete history over the window
    liq = df.groupby("PERMNO")["dollar_vol"].median()
    cols = sorted(full.columns, key=lambda c: -liq.get(c, 0.0))
    return full[cols[:n_max]].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="CRSP daily CSV (.gz ok)")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--sizes", default="30,100,300,500")
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    full_max_n = 50  # full likelihood is O(T*n^3); skip above this

    t0 = time.perf_counter()
    df = load_returns(args.path, args.start, args.end)
    panel = build_panel(df, max(sizes))
    T, n_avail = panel.shape
    print(
        f"loaded {len(df):,} rows in {time.perf_counter() - t0:.0f}s; "
        f"panel T={T} days, {n_avail} complete-history names "
        f"({args.start}..{args.end})"
    )
    if n_avail < max(sizes):
        print(f"note: only {n_avail} names available; trimming sizes")
        sizes = [s for s in sizes if s <= n_avail]

    # Warm up JIT
    DCC(n_jobs=1, method="cl", low_memory=True).fit(panel[:200, :3])
    DCC(n_jobs=1, method="full", low_memory=True).fit(panel[:200, :3])

    print("\n| n | CCC (s) | DCC cl (s) | a | b | a+b | DCC full (s) |")
    print("|---|---------|------------|---|---|-----|--------------|")
    for n in sizes:
        r = panel[:, :n]
        t0 = time.perf_counter()
        CCC(low_memory=True).fit(r)
        t_ccc = time.perf_counter() - t0
        t0 = time.perf_counter()
        m = DCC(method="cl", low_memory=True).fit(r)
        t_cl = time.perf_counter() - t0
        if n <= full_max_n:
            t0 = time.perf_counter()
            DCC(method="full", low_memory=True).fit(r)
            t_full = f"{time.perf_counter() - t0:.2f}"
        else:
            t_full = "—"
        print(
            f"| {n} | {t_ccc:.2f} | {t_cl:.2f} | {m.a:.4f} | {m.b:.4f} "
            f"| {m.a + m.b:.4f} | {t_full} |"
        )


if __name__ == "__main__":
    main()
