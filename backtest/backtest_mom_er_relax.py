"""
With 12-1 momentum support, is it OK to relax the router's ER>=0.40 gate?

Companion to backtest_trend_lookback.py (same monthly-rebalance methodology,
same universe/dates/book gate). That test varied only the DIRECTION measure
with ER held fixed at 0.40; this holds the direction measure fixed at the
live rule (slope20>0) and instead relaxes ER itself, admitting additional
names in [ER_LOW, 0.40) ONLY if they also have m12 (12-1 momentum) > 0 -
i.e. treats sustained 12-month momentum as a substitute confirmation for a
noisier 20-day efficiency ratio, not a free relaxation.

PRE-COMMITTED BAR (same as backtest_trend_lookback.py): the relaxed variant
must beat the live ER>=0.40 rule's Sharpe in BOTH halves and not give up
more than 1pp CAGR full-sample. Also reports breadth (names held) - a wider
net that is mostly empty months is not actually a relaxation.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

YEARS = 15
ER_RELAXED_CANDIDATES = [0.30, 0.25]


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

    V = ["er040_live"] + [f"er{int(e*100)}_m12" for e in ER_RELAXED_CANDIDATES]
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
                y = np.log(c.tail(20).values); x = np.arange(len(y)) - (len(y) - 1) / 2
                slope_pos = float(x @ (y - y.mean())) > 0.0
                if not slope_pos:
                    continue
                m12 = float(c.iloc[-22] / c.iloc[-252] - 1.0) if len(c) >= 252 else np.nan
                if er >= 0.40:
                    picks["er040_live"].append(s)
                    for e in ER_RELAXED_CANDIDATES:
                        picks[f"er{int(e*100)}_m12"].append(s)   # live names always included
                else:
                    for e in ER_RELAXED_CANDIDATES:
                        if er >= e and np.isfinite(m12) and m12 > 0:
                            picks[f"er{int(e*100)}_m12"].append(s)
        for v in V:
            sel = [s for s in picks[v]
                   if np.isfinite(px[s].asof(d0)) and np.isfinite(px[s].asof(d1))]
            r = (cash if not sel
                 else float(np.mean([px[s].asof(d1) / px[s].asof(d0) - 1.0 for s in sel])))
            rets[v].append(r)
            breadth[v].append(len(sel))

    half = (len(dates) - 1) // 2
    print(f"\n{len(dates)-1} monthly periods, {dates[0].date()}..{dates[-1].date()}"
          f"  (direction fixed: slope20>0, >200DMA, >50DMA, SPY gate; only ER varies,"
          f" relaxed names require m12>0)\n")
    print(f"{'variant':>14} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>8} {'avg n':>6} "
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
        print(f"{v:>14} {cagr:>7.1%} {sh:>7.2f} {dd:>8.1%} "
              f"{np.mean(breadth[v]):>6.1f} {np.mean([b==0 for b in breadth[v]]):>6.0%} | "
              f"{sh1:>6.2f} {sh2:>6.2f}")

    print("\npre-committed bar vs live ER>=0.40: Sharpe higher in BOTH halves, CAGR within 1pp")
    live = base_sh["er040_live"]
    for v in V[1:]:
        x = base_sh[v]
        ok = x[1] > live[1] and x[2] > live[2] and x[3] >= live[3] - 0.01
        print(f"  {v:>14}: h1 {x[1]:.2f} vs {live[1]:.2f} {'y' if x[1]>live[1] else 'n'} | "
              f"h2 {x[2]:.2f} vs {live[2]:.2f} {'y' if x[2]>live[2] else 'n'} | "
              f"CAGR {x[3]:.1%} vs {live[3]:.1%} {'y' if x[3]>=live[3]-0.01 else 'n'} "
              f"{'<-- PASSES' if ok else ''}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
