"""
Is the RSI2 timing an edge, or just the 200-SMA filter + partial exposure?
Per ETF (universe B, performance-blind), 2012+:
  MR      : RSI2 rule (long RSI2<=10 & >200SMA, exit >=70, flat <200SMA)
  TREND   : long whenever >200SMA, scaled by the MR book's average exposure
  SHUFFLE : same ETF, same >200SMA gate, random entry days with matched
            number of in-market days (200 draws)  -> tests the RSI2 timing itself
Excess-over-cash Sharpe, equal-weight across ETFs, 5bps/switch.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from backtest.backtest_mr_etf_selection import etf_positions
from backtest.backtest_mr_autocorr_robust import B

raw = yf.download(B + ["BIL"], start="2004-01-01", end="2026-11-20", progress=False, auto_adjust=True)["Close"].ffill()
cash = raw["BIL"].pct_change().fillna(0)
px = raw[[t for t in B if t in raw.columns]]
px = px.loc[:, px.notna().sum() > 800]
rets = px.pct_change()
sma_ok = (px > px.rolling(200, min_periods=200).mean()).astype(float)
pos = pd.DataFrame({c: etf_positions(px[c].dropna()).reindex(px.index).fillna(0) for c in px})
T0 = pd.Timestamp("2012-01-01"); m = px.index >= T0


def book(p, cost=5.0):
    p1 = p.shift(1).fillna(0)
    g = (p1 * rets).sub(0, axis=0) + (1 - p1).mul(cash, axis=0)
    g = g - p1.diff().abs().fillna(0) * cost / 1e4
    return g.where(px.notna()).mean(axis=1)


def stats(r, label):
    x = (r[m] - cash[m]).dropna()
    sh = x.mean() * 252 / (x.std() * np.sqrt(252))
    eq = (1 + r[m]).cumprod()
    print(f"{label:34s} excess-Sharpe={sh:5.2f}  CAGR={eq.iloc[-1]**(252/m.sum())-1:6.1%}  maxDD={(eq/eq.cummax()-1).min():6.1%}")
    return sh


mr = book(pos)
expo = pos[m].mean().mean()
print(f"MR avg exposure: {expo:.1%} of days in market\n")
s_mr = stats(mr, "MR (RSI2 + 200SMA)")
trend_full = book(sma_ok)
scale = expo / sma_ok[m].mean().mean()
trend = trend_full * scale + cash * (1 - scale)
s_tr = stats(trend, "Trend-only, exposure-matched")
stats(book(sma_ok), "Trend-only, full exposure")
stats(rets.mean(axis=1), "Buy&hold basket (no gate)")

rng = np.random.default_rng(1)
sh = []
for _ in range(200):
    p = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for c in px:
        ok = sma_ok[c].values.astype(bool)
        n = int(pos[c].values[m & ok].sum()) if True else 0
        n = int(pos[c][m].sum())
        cand = np.where(ok & m)[0]
        if len(cand) == 0: continue
        pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
        arr = np.zeros(len(px)); arr[pick] = 1; p[c] = arr
    r = book(p)
    x = (r[m] - cash[m]).dropna(); sh.append(x.mean() * 252 / (x.std() * np.sqrt(252)))
sh = np.array(sh)
print(f"{'Shuffled entries (same gate/exposure)':34s} excess-Sharpe mean={sh.mean():.2f}  5-95%=[{np.percentile(sh,5):.2f},{np.percentile(sh,95):.2f}]")
print(f"\nRSI2 timing beats {(sh < s_mr).mean():.0%} of shuffled-entry draws; beats exposure-matched trend: {s_mr > s_tr}")
