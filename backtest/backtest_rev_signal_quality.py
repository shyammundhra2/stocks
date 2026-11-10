"""
Is z20 actually a good mean-reversion signal, or just a filter that fires rarely?

The REV leg buys a name when it is CHOP (ER<=0.35), above its 200DMA, oversold
(z20 < -1.0) and directionless. z20 = 20d return / (20d vol * sqrt(20)).

THE CONTROL THAT MATTERS. "z20 < -1 earns a positive forward return" proves
nothing - almost everything earns a positive forward return in a 15-year sample.
The honest question is whether it beats the UNCONDITIONAL return of the same
eligible pool, i.e. whether picking the dipped names beats picking any CHOP name
above its 200DMA at random. That is the comparison this reports.

Also tested:
    - monotonicity across z20 buckets (a real signal should grade, not spike)
    - the OVERBOUGHT mirror (z20 > +1): if reversion is real it should UNDER-
      perform. If both tails beat the pool, the signal is not reversion.
    - RSI2 < 15, the rule z20 replaced, on the same pool and horizon
    - whether the 2 extra gates (200DMA, conviction curve) add anything over
      z20 alone
    - both halves separately, because one regime proves nothing

Horizon 20 sessions, matching the sleeve's ~monthly rebalance.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators.mathstats import _efficiency_ratio

HORIZON = 20
YEARS = 15


def build_panel():
    syms = sorted(TREND_ASSETS)
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    rows = []
    for s in syms:
        c = px[s].dropna()
        if len(c) < 400:
            continue
        ret = c.pct_change()
        sma200 = c.rolling(200).mean()
        vol20 = ret.rolling(20).std() * np.sqrt(20)
        ret20 = c / c.shift(20) - 1.0
        z20 = ret20 / vol20
        # Kaufman ER, vectorised
        net = (c - c.shift(20)).abs()
        path = ret.abs().mul(c.shift(1)).rolling(20).sum()
        er = (net / path).replace([np.inf, -np.inf], np.nan)
        # RSI2 (Wilder) for the head-to-head
        d = c.diff()
        g = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
        rsi2 = 100 - 100 / (1 + g / l)
        fwd = c.shift(-HORIZON) / c - 1.0
        rows.append(pd.DataFrame({
            "sym": s, "date": c.index, "z20": z20.values, "er": er.values,
            "above200": (c > sma200).values, "rsi2": rsi2.values, "fwd": fwd.values,
        }))
    d = pd.concat(rows, ignore_index=True).dropna(subset=["z20", "er", "fwd"])
    return d


def summarise(sub, label, pool_mean):
    if len(sub) < 50:
        return f"  {label:28s} n={len(sub):>5}  (too few)"
    m = sub.fwd.mean()
    edge = m - pool_mean
    t = m / (sub.fwd.std() / np.sqrt(len(sub)))
    return (f"  {label:28s} n={len(sub):>5}  fwd {m:>+7.2%}  "
            f"vs pool {edge:>+7.2%}  hit {(sub.fwd > 0).mean():>5.0%}  t={t:>5.1f}")


def main():
    t0 = time.time()
    d = build_panel()
    print(f"\npanel: {len(d):,} (asset, day) rows, {d.sym.nunique()} assets, "
          f"{d.date.min().date()} .. {d.date.max().date()}, fwd {HORIZON}d\n")

    # THE POOL: what REV is choosing FROM - chop names above their 200DMA
    pool = d[(d.er <= 0.35) & (d.above200)]
    pm = pool.fwd.mean()
    print(f"ELIGIBLE POOL (ER<=0.35 and above 200DMA): n={len(pool):,}  "
          f"unconditional fwd {pm:+.2%}")
    print(f"  (this is the bar - a signal must beat buying a pool name at random)\n")

    print("=== z20 buckets within the pool (monotonic?) ===")
    bins = [(-99, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 99)]
    for lo, hi in bins:
        sub = pool[(pool.z20 > lo) & (pool.z20 <= hi)]
        print(summarise(sub, f"z20 in ({lo:g}, {hi:g}]", pm))

    print("\n=== the live rule vs its mirror ===")
    print(summarise(pool[pool.z20 < -1.0], "REV gate: z20 < -1.0", pm))
    print(summarise(pool[pool.z20 > 1.0], "MIRROR:   z20 > +1.0", pm))
    print(summarise(pool[pool.rsi2 < 15], "replaced: RSI2 < 15", pm))

    print("\n=== do the extra gates earn their keep? ===")
    allz = d[d.z20 < -1.0]
    print(f"  {'z20<-1 alone (no gates)':28s} n={len(allz):>5}  fwd {allz.fwd.mean():>+7.2%}  "
          f"vs ALL rows {allz.fwd.mean() - d.fwd.mean():>+7.2%}")
    chop_only = d[(d.er <= 0.35) & (d.z20 < -1.0)]
    print(f"  {'+ chop gate':28s} n={len(chop_only):>5}  fwd {chop_only.fwd.mean():>+7.2%}")
    full = pool[pool.z20 < -1.0]
    print(f"  {'+ 200DMA gate (live rule)':28s} n={len(full):>5}  fwd {full.fwd.mean():>+7.2%}")
    print(f"  all rows baseline: fwd {d.fwd.mean():+.2%}")

    # ---------------------------------------------------------------- THE CONTROLS
    # Everything above pools observations across TIME, which conflates two very
    # different claims: "dipped names bounce" (selection) and "the pool bounces
    # after a broad dip" (timing). These three corrections separate them.
    print("\n=== CONTROL 1: non-overlapping samples ===")
    dd = d.sort_values(["sym", "date"]).copy()
    dd["i"] = dd.groupby("sym").cumcount()
    nd = dd[dd.i % HORIZON == 0]
    npool = nd[(nd.er <= 0.35) & nd.above200]
    nsig = npool[npool.z20 < -1.0]
    npm = npool.fwd.mean()
    nt = (nsig.fwd.mean() - npm) / (nsig.fwd.std() / np.sqrt(max(len(nsig), 1)))
    print(f"  one obs per asset per {HORIZON}d: n={len(nsig)}  pool {npm:+.2%}  "
          f"signal {nsig.fwd.mean():+.2%}  edge {nsig.fwd.mean()-npm:+.2%}  t={nt:.2f}")
    print("  (daily sampling of 20d forward returns overlaps ~20x; the headline")
    print("   t-stats above are inflated by roughly sqrt(20) and mean nothing)")

    print("\n=== CONTROL 2: cluster by date (removes cross-asset correlation) ===")
    gmean = pool.groupby("date").fwd.mean()
    smean = pool[pool.z20 < -1.0].groupby("date").fwd.mean()
    j = smean.index.intersection(gmean.index)
    diff = smean.loc[j] - gmean.loc[j]
    print(f"  {len(diff)} distinct signal dates  mean edge {diff.mean():+.2%}  "
          f"t={diff.mean()/(diff.std()/np.sqrt(len(diff))):.2f}")

    print("\n=== CONTROL 3: market timing vs asset selection ===")
    pool2 = pool.copy()
    pool2["fwd_xs"] = pool2.fwd - pool2.groupby("date").fwd.transform("mean")
    print(f"  {'bucket':>18} {'n':>7} {'raw edge':>10} {'XS edge':>9}")
    for lo, hi in [(-99, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 99)]:
        sb = pool2[(pool2.z20 > lo) & (pool2.z20 <= hi)]
        if len(sb) < 100:
            continue
        print(f"  z20 ({lo:>5g},{hi:>5g}] {len(sb):>7} {sb.fwd.mean()-pm:>+10.2%} "
              f"{sb.fwd_xs.mean():>+9.2%}")
    print("  raw edge = vs the pool across all dates (timing + selection)")
    print("  XS edge  = vs the pool ON THE SAME DATE (selection only)")

    print("\n=== both halves (one regime proves nothing) ===")
    mid = d.date.quantile(0.5)
    for lbl, m in [("first half", d.date <= mid), ("second half", d.date > mid)]:
        p = d[m & (d.er <= 0.35) & d.above200]
        sig = p[p.z20 < -1.0]
        if len(sig) < 50:
            print(f"  {lbl}: too few")
            continue
        print(f"  {lbl:12s} pool {p.fwd.mean():>+7.2%}  signal {sig.fwd.mean():>+7.2%}  "
              f"edge {sig.fwd.mean() - p.fwd.mean():>+7.2%}  n={len(sig)}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
