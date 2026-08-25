"""
Pick the best mean-reversion ranker for commodities before wiring it in.
Prior test showed momentum is inverted (mom126 IC -0.113); this confirms the
reversal works and picks the signal. Candidates (higher signal = expect
higher fwd return, so reversion signals are NEGATED recent return / oversold):

  rev_21   - negated 21d return (buy biggest 1mo losers)
  rev_63   - negated 63d return
  rev_126  - negated 126d return
  rsi14_lo - low RSI14 (oversold), signal = -(RSI14)
  z_dist   - negated distance above 63d mean (buy most below-average)
  mom63    - momentum (control - should be negative)

Eval: 63d + 21d forward horizons, IC + top-1 excess vs median + top-1 hit,
strided non-overlapping, 2007-26. Commodities downloaded alone.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import COMMODITIES


def rsi(series, period=14):
    d = series.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))


def main():
    t0 = time.time()
    print("Downloading commodities ...")
    raw = yf.download(list(COMMODITIES.keys()), start="2004-01-01", end="2026-08-21",
                      progress=False, auto_adjust=True)["Close"]
    close = raw[[c for c in raw.columns if raw[c].notna().sum() > 2000]].dropna(how="all")
    print(f"names: {close.shape[1]}\n")

    sig = {}
    sig["rev_21"] = -close.pct_change(21)
    sig["rev_63"] = -close.pct_change(63)
    sig["rev_126"] = -close.pct_change(126)
    sig["rsi14_lo"] = pd.DataFrame({c: -rsi(close[c]) for c in close.columns}, index=close.index)
    mean63 = close.rolling(63).mean()
    sig["z_dist"] = -(close - mean63) / mean63
    sig["mom63"] = close.pct_change(63)

    for H in [21, 63]:
        fwd = close.pct_change(H).shift(-H)
        dates = close.index[close.index >= "2007-01-01"][::H]
        print(f"== horizon {H}d ==")
        print(f"  {'ranker':>10s} {'IC':>7s} {'top1 xs%':>9s} {'top1 hit%':>9s} {'n':>5s}")
        rows = []
        for name, S in sig.items():
            ics = []; xs = []; hit = []
            for d in dates:
                if d not in S.index or d not in fwd.index:
                    continue
                sv = S.loc[d]; fv = fwd.loc[d]
                m = sv.notna() & fv.notna()
                if m.sum() < 6:
                    continue
                sv = sv[m]; fv = fv[m]
                ics.append(sv.rank().corr(fv.rank()))
                top = sv.idxmax()
                e = fv[top] - fv.median()
                xs.append(e); hit.append(e > 0)
            if ics:
                rows.append((name, np.mean(ics), np.mean(xs) * 100, np.mean(hit) * 100, len(ics)))
        for name, ic, x, h, n in sorted(rows, key=lambda r: -r[1]):
            print(f"  {name:>10s} {ic:>7.3f} {x:>8.2f}% {h:>8.0f}% {n:>5d}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
