"""
Does a DIFFERENT z-score fix REV's selection problem?

backtest_rev_signal_quality showed z20 < -1.0 has a big RAW edge (+0.86% vs the
eligible pool) that is entirely market timing: date-demeaned, the cross-sectional
edge is -0.03%, i.e. on the day it fires the dipped names do no better than their
undipped peers. So the question is whether that is a flaw in the PARAMETERS
(wrong lookback, wrong threshold) or in the IDEA (dip depth does not rank).

Sweeps lookback L in {5,10,20,40,60,120} and threshold in {-0.5,-1,-1.5,-2}.
  z_L = (L-day return) / (L-day vol * sqrt(L))

Reports BOTH:
  RAW edge = signal mean minus pool mean across all dates  (timing + selection)
  XS  edge = signal mean minus the pool mean ON THE SAME DATE (selection only)

XS is the column that decides this. A parameter choice that lifts RAW but not XS
has found a better market-timing trigger, not a better name-picker - and REV
already has the 200DMA gate for timing.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

HORIZON = 20
LOOKBACKS = [5, 10, 20, 40, 60, 120]
THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]


def build():
    syms = sorted(TREND_ASSETS)
    px = yf.download(syms, period="15y", auto_adjust=True, progress=False)["Close"].dropna(how="all")
    rows = []
    for s in syms:
        c = px[s].dropna()
        if len(c) < 400:
            continue
        ret = c.pct_change()
        f = pd.DataFrame({"sym": s, "date": c.index})
        f["above200"] = (c > c.rolling(200).mean()).values
        net = (c - c.shift(20)).abs()
        path = ret.abs().mul(c.shift(1)).rolling(20).sum()
        f["er"] = (net / path).replace([np.inf, -np.inf], np.nan).values
        for L in LOOKBACKS:
            v = ret.rolling(L).std() * np.sqrt(L)
            f[f"z{L}"] = ((c / c.shift(L) - 1.0) / v).values
        f["fwd"] = (c.shift(-HORIZON) / c - 1.0).values
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def main():
    t0 = time.time()
    d = build()
    pool = d[(d.er <= 0.35) & d.above200].dropna(subset=["fwd"]).copy()
    pool["fwd_xs"] = pool.fwd - pool.groupby("date").fwd.transform("mean")
    pm = pool.fwd.mean()
    print(f"\neligible pool n={len(pool):,}  unconditional fwd {pm:+.2%}  "
          f"(fwd {HORIZON}d, {d.sym.nunique()} assets)")
    print("\nRAW edge = timing + selection   |   XS edge = selection only "
          "(date-demeaned)\n")
    print(f"{'lookback':>9}" + "".join(f"{f'thr {t}':>22}" for t in THRESHOLDS))
    best = []
    for L in LOOKBACKS:
        line = f"{f'z{L}':>9}"
        for thr in THRESHOLDS:
            s = pool[pool[f"z{L}"] < thr]
            if len(s) < 150:
                line += f"{'(n<150)':>22}"
                continue
            raw = s.fwd.mean() - pm
            xs = s.fwd_xs.mean()
            best.append((abs(xs), L, thr, raw, xs, len(s)))
            line += f"  {raw:>+7.2%}/{xs:>+7.2%} n{len(s):<4}"
        print(line)

    print("\nlargest |XS edge| found anywhere in the sweep:")
    for a, L, thr, raw, xs, n in sorted(best, reverse=True)[:5]:
        print(f"   z{L:<4} thr {thr:>5}  n={n:>5}  RAW {raw:>+7.2%}  XS {xs:>+7.2%}")

    # is the best XS result significant once date-clustered?
    a, L, thr, raw, xs, n = sorted(best, reverse=True)[0]
    s = pool[pool[f"z{L}"] < thr]
    per_date = s.groupby("date").fwd_xs.mean()
    t = per_date.mean() / (per_date.std() / np.sqrt(len(per_date)))
    print(f"\nbest cell z{L} < {thr}: date-clustered XS edge {per_date.mean():+.2%} "
          f"over {len(per_date)} dates, t={t:.2f}")
    print("\nfor reference, the LIVE rule is z20 < -1.0")
    live = pool[pool.z20 < -1.0]
    pd_live = live.groupby("date").fwd_xs.mean()
    print(f"   live: RAW {live.fwd.mean()-pm:+.2%}  XS {live.fwd_xs.mean():+.2%}  "
          f"t={pd_live.mean()/(pd_live.std()/np.sqrt(len(pd_live))):.2f}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
