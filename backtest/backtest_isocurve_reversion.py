"""
User proposal: BUY beyond the outer iso-conviction curve on the NEGATIVE side
(a strong, clean downtrend: slope*r2 < -4 in map units) under some conditions,
and SELL when a name crosses beyond the outer curve on the POSITIVE side.

That is deep per-asset mean reversion: buy the far-left of the trend map, exit
the far-right. Everything validated here says the two halves should score very
differently - buying strong downtrends is knife-catching (the 200DMA gates
exist to prevent it), while "sell at strong positive conviction" bets against
momentum continuation. But the capitulation buyer proves crash-buying CAN work
with the right condition, so test rather than assume.

Design: signed conviction = slope(20d OLS, map scale) * r2, bucketed at the
map's own curve levels (+-1, +-2, +-4). Forward 20d and 60d returns, raw and
date-demeaned (XS), split by the obvious "conditions": above/below 200DMA and
VIX/VIX3M stress. Both halves. Monthly-equivalent non-overlap noted.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"^VIX", "^VIX3M"})
    px = yf.download(syms, period="15y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    vix, v3 = px["^VIX"].dropna(), px["^VIX3M"].dropna()
    ts = (vix / v3).dropna()
    assets = [s for s in TREND_ASSETS if s in px.columns]

    rows = []
    for s in assets:
        c = px[s].dropna()
        if len(c) < 400:
            continue
        lc = np.log(c)
        # rolling 20d OLS slope + r2, closed form (same math as _trend_stats)
        L = 20
        from numpy.lib.stride_tricks import sliding_window_view
        Y = sliding_window_view(lc.values, L)
        x = np.arange(L, dtype=float); xc = x - x.mean(); sxx = float(xc @ xc)
        Yc = Y - Y.mean(axis=1, keepdims=True)
        beta = (Yc @ xc) / sxx
        sst = np.einsum("ij,ij->i", Yc, Yc)
        r2 = np.divide(beta**2 * sxx, sst, out=np.zeros_like(beta), where=sst > 0)
        slope = beta * 10 * 100                    # map units (scale=10)
        conv = slope * r2                          # SIGNED conviction
        idxs = c.index[L - 1:]
        f = pd.DataFrame({"sym": s, "date": idxs, "conv": conv,
                          "above200": (c > c.rolling(200).mean()).reindex(idxs).values,
                          "fwd20": (c.shift(-20) / c - 1).reindex(idxs).values,
                          "fwd60": (c.shift(-60) / c - 1).reindex(idxs).values})
        rows.append(f)
    d = pd.concat(rows, ignore_index=True).dropna(subset=["conv", "fwd20"])
    d["stress"] = d.date.map(ts.reindex(d.date.unique()).ffill()) > 1.0
    for h in ("fwd20", "fwd60"):
        d[h + "_xs"] = d[h] - d.groupby("date")[h].transform("mean")

    half = d.date.quantile(0.5)
    BUCKETS = [(-99, -4), (-4, -2), (-2, -1), (-1, 1), (1, 2), (2, 4), (4, 99)]

    def table(sub, label):
        print(f"\n--- {label} (n={len(sub):,}) ---")
        print(f"{'conviction':>12} {'n':>7} {'fwd20':>8} {'xs20':>8} {'fwd60':>8} "
              f"{'xs60':>8} {'h1 f20':>8} {'h2 f20':>8}")
        for lo, hi in BUCKETS:
            b = sub[(sub.conv > lo) & (sub.conv <= hi)]
            if len(b) < 100:
                print(f"{f'{lo:g}..{hi:g}':>12} {len(b):>7}  (too few)")
                continue
            b1 = b[b.date <= half]; b2 = b[b.date > half]
            print(f"{f'{lo:g}..{hi:g}':>12} {len(b):>7} {b.fwd20.mean():>+8.2%} "
                  f"{b.fwd20_xs.mean():>+8.2%} {b.fwd60.mean():>+8.2%} {b.fwd60_xs.mean():>+8.2%} "
                  f"{(b1.fwd20.mean() if len(b1)>50 else np.nan):>+8.2%} "
                  f"{(b2.fwd20.mean() if len(b2)>50 else np.nan):>+8.2%}")

    table(d, "ALL observations")
    table(d[d.above200], "condition: ABOVE 200DMA")
    table(d[~d.above200], "condition: BELOW 200DMA")
    table(d[d.stress.fillna(False)], "condition: VIX/VIX3M > 1 (stress)")

    # the SELL half: does deep positive conviction mean exit?
    print("\n=== the SELL side: is conv > +4 a reason to exit? ===")
    deep = d[d.conv > 4]; mid = d[(d.conv > 1) & (d.conv <= 4)]
    print(f"  conv > +4 : n={len(deep):,}  fwd20 {deep.fwd20.mean():+.2%}  "
          f"fwd60 {deep.fwd60.mean():+.2%}  xs20 {deep.fwd20_xs.mean():+.2%}")
    print(f"  conv +1..4: n={len(mid):,}  fwd20 {mid.fwd20.mean():+.2%}  "
          f"fwd60 {mid.fwd60.mean():+.2%}  xs20 {mid.fwd20_xs.mean():+.2%}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
