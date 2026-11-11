"""
Should the trend sleeve select on LONGER-term trends?

The router's MOM entry is: ER(20)>=0.40 AND px>200DMA AND px>50DMA AND
slope(20d)>0 - i.e. the direction test lives on a ~1-month window, the horizon
the momentum literature calls noisiest (and the one the stock sleeve already
abandoned for 12-1). This holds every gate fixed and varies ONLY the trend
measure that must be positive:

    s20   slope of 20d log-price OLS      (live rule)
    s60   slope of 60d
    s120  slope of 120d
    m12_1 12-1 momentum (252d return skipping last 21d) - the stock sleeve's

Each monthly rebalance: qualifying names equal-weight, held one month; empty ->
BIL. Same universe, same gates, same dates, so any difference is the lookback.

PRE-COMMITTED BAR for "yes, go longer": the variant must beat s20 on Sharpe in
BOTH halves and not give up more than 1pp CAGR full-sample. Also reported:
average breadth (names held) and how often each variant is empty, because a
longer filter that is usually empty is a cash allocation wearing a trend badge.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

YEARS = 15


def slope_pos(c, L):
    w = np.log(c.tail(L).dropna().values)
    if len(w) < L:
        return False
    x = np.arange(len(w)); x = x - x.mean()
    return float(x @ (w - w.mean())) > 0.0


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"SPY", "BIL"})
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    spy, bil = px["SPY"].dropna(), px["BIL"].dropna()
    spy200 = spy.rolling(200).mean()
    assets = [s for s in TREND_ASSETS if s in px.columns]
    idx = px.index
    ends = px.resample("ME").last().index
    dates = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in ends
                    if idx.searchsorted(d, side="right") - 1 > 300})

    V = ["s20", "s60", "s120", "m12_1"]
    rets = {v: [] for v in V}
    breadth = {v: [] for v in V}

    for k in range(len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        ro = bool(spy.asof(d0) < spy200.asof(d0))
        cash = float(bil.asof(d1) / bil.asof(d0) - 1.0)
        picks = {v: [] for v in V}
        if not ro:
            for s in assets:
                c = px[s].loc[:d0].dropna()
                if len(c) < 260:
                    continue
                last = float(c.iloc[-1])
                if not (last > float(c.rolling(200).mean().iloc[-1])
                        and last > float(c.rolling(50).mean().iloc[-1])):
                    continue
                w = c.tail(21)
                path = w.diff().abs().sum()
                er = abs(w.iloc[-1] - w.iloc[0]) / path if path > 0 else 0.0
                if er < 0.40:
                    continue
                if slope_pos(c, 20):
                    picks["s20"].append(s)
                if slope_pos(c, 60):
                    picks["s60"].append(s)
                if slope_pos(c, 120):
                    picks["s120"].append(s)
                if float(c.iloc[-22] / c.iloc[-252] - 1.0) > 0:
                    picks["m12_1"].append(s)
        for v in V:
            sel = [s for s in picks[v]
                   if np.isfinite(px[s].asof(d0)) and np.isfinite(px[s].asof(d1))]
            r = (cash if not sel
                 else float(np.mean([px[s].asof(d1) / px[s].asof(d0) - 1.0 for s in sel])))
            rets[v].append(r)
            breadth[v].append(len(sel))

    half = (len(dates) - 1) // 2
    print(f"\n{len(dates)-1} monthly periods, {dates[0].date()}..{dates[-1].date()}"
          f"  (gates fixed: ER20>=0.40, >200DMA, >50DMA, SPY gate; only the "
          f"direction measure varies)\n")
    print(f"{'variant':>7} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>8} {'avg n':>6} "
          f"{'empty':>6} | {'Sh h1':>6} {'Sh h2':>6}")
    base_sh = {}
    for v in V:
        r = pd.Series(rets[v])
        eq = (1 + r).cumprod()
        cagr = eq.iloc[-1] ** (12 / len(r)) - 1
        sh = r.mean() / r.std() * np.sqrt(12)
        dd = float((eq / eq.cummax() - 1).min())
        s1 = r.iloc[:half]; s2 = r.iloc[half:]
        sh1 = s1.mean() / s1.std() * np.sqrt(12)
        sh2 = s2.mean() / s2.std() * np.sqrt(12)
        base_sh[v] = (sh, sh1, sh2, cagr)
        print(f"{v:>7} {cagr:>7.1%} {sh:>7.2f} {dd:>8.1%} "
              f"{np.mean(breadth[v]):>6.1f} {np.mean([b==0 for b in breadth[v]]):>6.0%} | "
              f"{sh1:>6.2f} {sh2:>6.2f}")

    print("\npre-committed bar vs s20: Sharpe higher in BOTH halves, CAGR within 1pp")
    s20 = base_sh["s20"]
    for v in ("s60", "s120", "m12_1"):
        x = base_sh[v]
        ok = x[1] > s20[1] and x[2] > s20[2] and x[3] >= s20[3] - 0.01
        print(f"  {v:>6}: h1 {x[1]:.2f} vs {s20[1]:.2f} {'y' if x[1]>s20[1] else 'n'} | "
              f"h2 {x[2]:.2f} vs {s20[2]:.2f} {'y' if x[2]>s20[2] else 'n'} | "
              f"CAGR {x[3]:.1%} vs {s20[3]:.1%} {'y' if x[3]>=s20[3]-0.01 else 'n'} "
              f"{'<-- PASSES' if ok else ''}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
