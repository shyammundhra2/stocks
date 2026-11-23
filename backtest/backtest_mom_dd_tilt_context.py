"""
Does the dd-phase tilt (backtest_trend_dd_tilt.py: 12-1 momentum's forward-
return IC peaks for names 5-15% off their 252d high) add anything ON TOP of
the LIVE router's actual MOM branch (macro/indicators/portfolio.py get_trends:
ER(20)>=0.40, above 200DMA, above 50DMA, slope(20,scale=10)>0), using the
live code's own _trend_stats/_efficiency_ratio (not a reimplementation)?

Companion to backtest_rev_trend_context.py (the REV-side question, which
found 12-1 momentum>0 is a real, specific gate on dip-buys). This is the
mirror test for the MOM side: among events where the router would actually
fire MOM, does being in the dd-zone (and/or having m12 support) earn MORE
than the unconditional MOM-eligible pool, fwd 20d (matching the sleeve's
~monthly rebalance)? Also reports the control (same dd-zone split on ALL
above-200DMA names, not just MOM-eligible ones) to check this isn't just a
generic dd-zone effect bleeding in from everywhere.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators.mathstats import _trend_stats, _efficiency_ratio

YEARS = 15
HORIZON = 20


def build_panel():
    syms = sorted(TREND_ASSETS)
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    rows = []
    for s in syms:
        c = px[s].dropna()
        if len(c) < 400:
            continue
        n = len(c)
        sma200 = c.rolling(200).mean()
        sma50 = c.rolling(50).mean()
        hi252 = c.rolling(252, min_periods=60).max()
        dd = c / hi252 - 1.0
        m12 = c.shift(21) / c.shift(252) - 1.0
        fwd = c.shift(-HORIZON) / c - 1.0

        slope = np.full(n, np.nan); r2 = np.full(n, np.nan); er = np.full(n, np.nan)
        vals = c.values
        for i in range(20, n):
            sl, r = _trend_stats(pd.Series(vals[max(0, i - 19):i + 1]), 20, 10)
            slope[i] = sl; r2[i] = r
            er[i] = _efficiency_ratio(pd.Series(vals[max(0, i - 20):i + 1]), 20)

        rows.append(pd.DataFrame({
            "sym": s, "date": c.index, "slope": slope, "r2": r2, "er": er,
            "above200": (c > sma200).values, "above50": (c > sma50).values,
            "dd": dd.values, "m12": m12.values, "fwd": fwd.values,
        }))
    d = pd.concat(rows, ignore_index=True).dropna(subset=["er", "slope", "fwd"])
    return d


def rep(sub, label, pool_mean=None):
    if len(sub) < 50:
        print(f"  {label:40s} n={len(sub):>5}  (too few)"); return
    m = sub.fwd.mean()
    edge = f"  vs base {m - pool_mean:>+6.2%}" if pool_mean is not None else ""
    print(f"  {label:40s} n={len(sub):>5}  fwd {m:>+7.2%}{edge}  hit {(sub.fwd > 0).mean():>5.0%}")


def main():
    t0 = time.time()
    d = build_panel()

    # Live MOM-branch eligibility, exactly: ER>=0.40, above200, above50, slope>0.
    mom = d[(d.er >= 0.40) & d.above200 & d.above50 & (d.slope > 0)].dropna(subset=["dd", "m12"])
    base_mean = mom.fwd.mean()
    in_zone = (mom.dd > -0.15) & (mom.dd <= -0.05)

    print(f"\nMOM-eligible events (live router rule): n={len(mom)}, base fwd {base_mean:+.2%}\n")
    rep(mom, "base (all MOM-eligible)")
    rep(mom[in_zone], "+ dd in 5-15%-off-high zone", base_mean)
    rep(mom[~in_zone], "  complement (outside zone)", base_mean)
    rep(mom[in_zone & (mom.m12 > 0)], "+ dd-zone AND m12>0", base_mean)
    rep(mom[in_zone & (mom.m12 <= 0)], "  dd-zone but m12<=0", base_mean)

    print("\ncontrol - same dd-zone split on ALL above-200DMA names (not MOM-gated):")
    allpool = d[d.above200].dropna(subset=["dd"])
    all_mean = allpool.fwd.mean()
    all_zone = (allpool.dd > -0.15) & (allpool.dd <= -0.05)
    rep(allpool, "all above-200DMA", all_mean)
    rep(allpool[all_zone], "+ dd in zone (unconditional)", all_mean)

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
