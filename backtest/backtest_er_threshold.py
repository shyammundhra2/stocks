"""
Is 0.40 the right TREND threshold for the MOM/REV router?

portfolio.py's comment says 0.40 came from widening 0.50->0.40 on a 2020-26
backtest (Sharpe 1.03->1.34). That is one move in one direction on 6 years.
This sweeps the TREND threshold properly: {0.30, 0.35, 0.40, 0.45, 0.50, 0.55},
holding the CHOP threshold fixed at 0.35 (so MID narrows/widens as TREND moves,
exactly as production does when ER sits between the two), full router rules
(sector cap irrelevant here - this scores the MOM leg alone), monthly, 15y.

PRE-COMMITTED BAR: a threshold beats 0.40 only if it wins Sharpe in BOTH
halves and does not cost more than 1pp CAGR full-sample - same bar used
throughout this session.
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

CHOP = 0.35
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
YEARS = 15


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
                    if idx.searchsorted(d, side="right") - 1 > 260})

    rets = {th: [] for th in THRESHOLDS}
    breadth = {th: [] for th in THRESHOLDS}

    for k in range(len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        ro = bool(spy.asof(d0) < spy200.asof(d0))
        cash = float(bil.asof(d1) / bil.asof(d0) - 1.0)

        ers, slopes, above200 = {}, {}, {}
        for s in assets:
            c = px[s].loc[:d0].dropna()
            if len(c) < 260:
                continue
            last = float(c.iloc[-1])
            above200[s] = last > float(c.rolling(200).mean().iloc[-1])
            s50 = float(c.rolling(50).mean().iloc[-1])
            w = np.log(c.tail(20).values)
            x = np.arange(len(w)); x = x - x.mean()
            sl = float(x @ (w - w.mean())) / float(x @ x) if len(w) == 20 else np.nan
            slopes[s] = (sl, last > s50)
            wnd = c.tail(21)
            path = wnd.diff().abs().sum()
            ers[s] = abs(wnd.iloc[-1] - wnd.iloc[0]) / path if path > 0 else 0.0

        for th in THRESHOLDS:
            sel = []
            if not ro:
                for s in assets:
                    if s not in ers:
                        continue
                    if ers[s] >= th and above200.get(s) and slopes[s][1] and slopes[s][0] > 0:
                        sel.append(s)
            sel = [s for s in sel if np.isfinite(px[s].asof(d0)) and np.isfinite(px[s].asof(d1))]
            r = cash if not sel else float(np.mean([px[s].asof(d1) / px[s].asof(d0) - 1.0 for s in sel]))
            rets[th].append(r)
            breadth[th].append(len(sel))

    half = (len(dates) - 1) // 2
    print(f"\n{len(dates)-1} monthly periods, {dates[0].date()}..{dates[-1].date()}  "
          f"(CHOP fixed at {CHOP}, sweeping TREND threshold)\n")
    print(f"{'thresh':>7} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>8} {'avg n':>6} {'empty':>6} "
          f"| {'Sh h1':>6} {'Sh h2':>6}")
    base = {}
    for th in THRESHOLDS:
        r = pd.Series(rets[th])
        eq = (1 + r).cumprod()
        cagr = eq.iloc[-1] ** (12 / len(r)) - 1
        sh = r.mean() / r.std() * np.sqrt(12)
        dd = float((eq / eq.cummax() - 1).min())
        s1, s2 = r.iloc[:half], r.iloc[half:]
        sh1 = s1.mean() / s1.std() * np.sqrt(12)
        sh2 = s2.mean() / s2.std() * np.sqrt(12)
        base[th] = (sh, sh1, sh2, cagr)
        tag = " <- live" if th == 0.40 else ""
        print(f"{th:>7.2f} {cagr:>7.1%} {sh:>7.2f} {dd:>8.1%} {np.mean(breadth[th]):>6.1f} "
              f"{np.mean([b==0 for b in breadth[th]]):>6.0%} | {sh1:>6.2f} {sh2:>6.2f}{tag}")

    print(f"\npre-committed bar vs 0.40 (live): Sharpe higher BOTH halves, CAGR within 1pp")
    live = base[0.40]
    for th in THRESHOLDS:
        if th == 0.40:
            continue
        x = base[th]
        ok = x[1] > live[1] and x[2] > live[2] and x[3] >= live[3] - 0.01
        print(f"  {th:.2f}: h1 {x[1]:.2f} vs {live[1]:.2f} {'y' if x[1]>live[1] else 'n'} | "
              f"h2 {x[2]:.2f} vs {live[2]:.2f} {'y' if x[2]>live[2] else 'n'} | "
              f"CAGR {x[3]:.1%} {'y' if x[3]>=live[3]-0.01 else 'n'} {'<-- PASSES' if ok else ''}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
