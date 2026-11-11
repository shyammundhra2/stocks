"""
Should MR (the REV leg) demand LONGER-term trend context before buying dips?

Companion to backtest_trend_lookback (the MOM-side question). REV's live gate
is px > 200DMA. Variants add: 12-1 momentum > 0, 200DMA rising over the last
month, or both. Events are the usual (CHOP & >200DMA & z20<-1), fwd 20d.

RESULT (run 2026-11-11, 15y, 42 assets):
  base (>200DMA)        n=1586  raw +1.87%   XS -0.03%
  + m12>0               n=1410  raw +2.11%   XS +0.05%
    complement m12<=0   n= 128  raw +0.02%   XS -1.13%   <- the dead weight
  + 200DMA rising       n=1472  raw +2.03%
    complement falling  n= 114  raw -0.24%   XS -0.71%
  + both                n=1361  raw +2.17%

And the control that makes it meaningful: the SAME split on the whole pool
does nothing (pool m12>0 raw +1.01% vs m12<=0 +1.22%) - so this is specific to
dip-buying, not a general m12 tilt. Mechanism-sensible: a dip in a name with
12-1 support is a pullback; a dip without it is a downtrend with a 200DMA that
hasn't caught up yet.

HONEST SIZE: the filter removes only ~8% of events (the toxic tail), lifting
the leg ~+25-30bp/event raw. Cross-sectional selection stays ~0 - this does
not make z20 a picker, it just stops a known-bad entry. Cheap one-condition
change if ever adopted: add m12_1 > 0 to the router's REV branch.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
import backtest.backtest_rev_signal_quality as Q


def main():
    d = Q.build_panel()
    px = yf.download(sorted(d.sym.unique()), period="15y",
                     auto_adjust=True, progress=False)["Close"]
    d = d.set_index(["sym", "date"])
    extra = []
    for s in d.index.get_level_values(0).unique():
        c = px[s].dropna()
        sma200 = c.rolling(200).mean()
        e = pd.DataFrame({"m12": c.shift(21) / c.shift(252) - 1.0,
                          "rising": sma200 > sma200.shift(21)})
        e["sym"] = s; e["date"] = e.index
        extra.append(e)
    d = d.join(pd.concat(extra).set_index(["sym", "date"])[["m12", "rising"]]).reset_index()

    pool = d[(d.er <= 0.35) & d.above200].dropna(subset=["fwd"]).copy()
    pool["fwd_xs"] = pool.fwd - pool.groupby("date").fwd.transform("mean")
    base = pool[pool.z20 < -1.0]

    def rep(s, label):
        if len(s) < 60:
            print(f"  {label:34s} n={len(s):>5}  (too few)"); return
        print(f"  {label:34s} n={len(s):>5}  raw {s.fwd.mean():>+6.2%}  XS {s.fwd_xs.mean():>+6.2%}")

    print("\nREV events, fwd 20d - longer-trend context variants:\n")
    rep(base, "base (live gate: >200DMA)")
    rep(base[base.m12 > 0], "+ 12-1 momentum > 0")
    rep(base[base.m12 <= 0], "  complement m12<=0")
    rep(base[base.rising], "+ 200DMA rising")
    rep(base[~base.rising.fillna(False)], "  complement falling")
    rep(base[(base.m12 > 0) & base.rising], "+ both")
    print("\ncontrol - same split on the WHOLE pool (should be ~flat):")
    rep(pool[pool.m12 > 0], "pool, m12>0")
    rep(pool[pool.m12 <= 0], "pool, m12<=0")


if __name__ == "__main__":
    main()
