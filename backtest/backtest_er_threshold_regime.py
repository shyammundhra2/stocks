"""
Should the MOM-leg ER threshold (0.40) be conditioned on macro regime?

VIX/VIX3M is the one regime variable already validated twice in this repo (the
deploy throttle; the REV-vs-trend spread). Tests whether the SAME threshold
sweep from backtest_er_threshold picks a different optimum inside vs outside
VIX term-structure stress (ts>1.0, same cut as the prior test).

Methodology to avoid in-sample cherry-picking: find the best threshold on
EACH HALF separately, then check whether that half's "best" transfers to the
OTHER half still split by the same regime. If the optimal threshold is stable
across halves within a regime, conditioning is real; if it flips, it is noise
fit to one half.
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

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
YEARS = 15


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"SPY", "BIL", "^VIX", "^VIX3M"})
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    spy, bil, vix = px["SPY"].dropna(), px["BIL"].dropna(), px["^VIX"].dropna()
    v3 = px["^VIX3M"].dropna()
    spy200 = spy.rolling(200).mean()
    assets = [s for s in TREND_ASSETS if s in px.columns]
    idx = px.index
    ends = px.resample("ME").last().index
    dates = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in ends
                    if idx.searchsorted(d, side="right") - 1 > 260})

    rows = []
    for k in range(len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        ro = bool(spy.asof(d0) < spy200.asof(d0))
        cash = float(bil.asof(d1) / bil.asof(d0) - 1.0)
        ts = float(vix.asof(d0) / v3.asof(d0)) if np.isfinite(v3.asof(d0)) else np.nan

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

        row = {"date": d1, "ts": ts}
        for th in THRESHOLDS:
            sel = []
            if not ro:
                for s in assets:
                    if s not in ers:
                        continue
                    if ers[s] >= th and above200.get(s) and slopes[s][1] and slopes[s][0] > 0:
                        sel.append(s)
            sel = [s for s in sel if np.isfinite(px[s].asof(d0)) and np.isfinite(px[s].asof(d1))]
            row[th] = cash if not sel else float(np.mean(
                [px[s].asof(d1) / px[s].asof(d0) - 1.0 for s in sel]))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("date").dropna(subset=["ts"])
    half = df.index[len(df) // 2]
    q = df.ts.median()

    def sharpe(s):
        return s.mean() / s.std() * np.sqrt(12) if s.std() > 0 else np.nan

    print(f"\n{len(df)} months with VIX3M data, {df.index[0].date()}..{df.index[-1].date()}  "
          f"(median ts={q:.3f})\n")

    for lbl, mask_fn in [("CALM (ts <= median)", lambda d: d.ts <= q),
                         ("STRESS (ts > median)", lambda d: d.ts > q)]:
        sub = df[mask_fn(df)]
        print(f"=== {lbl}, n={len(sub)} ===")
        print(f"  {'thresh':>7} {'Sharpe all':>11} {'Sharpe h1':>10} {'Sharpe h2':>10}")
        best_all, best_h1, best_h2 = None, None, None
        for th in THRESHOLDS:
            s = sub[th]
            sh = sharpe(s)
            s1, s2 = s[s.index <= half], s[s.index > half]
            sh1, sh2 = sharpe(s1) if len(s1) > 15 else np.nan, sharpe(s2) if len(s2) > 15 else np.nan
            if best_all is None or (not np.isnan(sh) and sh > best_all[1]):
                best_all = (th, sh)
            if not np.isnan(sh1) and (best_h1 is None or sh1 > best_h1[1]):
                best_h1 = (th, sh1)
            if not np.isnan(sh2) and (best_h2 is None or sh2 > best_h2[1]):
                best_h2 = (th, sh2)
            print(f"  {th:>7.2f} {sh:>11.2f} {sh1:>10.2f} {sh2:>10.2f}")
        print(f"  best overall: {best_all[0]}  |  best h1: {best_h1[0]}  |  best h2: {best_h2[0]}  "
              f"{'<-- STABLE' if best_h1[0]==best_h2[0] else '<-- FLIPS, not stable'}\n")

    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
