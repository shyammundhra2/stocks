"""
Does vol-managed exposure beat exposure-matched STATIC exposure on an equal-weight ETF basket?
Universe B (performance-blind), 2012+. Two baskets:
  RAW    : equal-weight buy&hold of the ETFs
  GATED  : each ETF held only when >200SMA (the book's gate), else cash
Vol-managed: exposure = min(1, target_vol / realized_vol), realized = 63d vol of the
basket's own return (lagged 1 day), re-set weekly; remainder in BIL; 5bps on |d exposure|.
Static comparator: constant exposure equal to the vol-managed average exposure.
Targets 8/10/12% all reported (no cherry-picking). Split into 2012-19 and 2020-26.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from backtest.backtest_mr_autocorr_robust import B

raw = yf.download(B + ["BIL"], start="2004-01-01", end="2026-11-20", progress=False, auto_adjust=True)["Close"].ffill()
cash = raw["BIL"].pct_change().fillna(0)
px = raw[[t for t in B if t in raw.columns]]
px = px.loc[:, px.notna().sum() > 800]
rets = px.pct_change()
gate = (px > px.rolling(200, min_periods=200).mean()).astype(float).shift(1)

baskets = {
    "RAW": rets.mean(axis=1),
    "GATED": (rets * gate).where(px.notna()).sum(axis=1) / px.notna().sum(axis=1)
             + (1 - gate.where(px.notna()).mean(axis=1)) * cash,
}


def metrics(r, mask):
    r = r[mask]; x = r - cash[mask]
    sh = x.mean() * 252 / (x.std() * np.sqrt(252))
    eq = (1 + r).cumprod(); yrs = mask.sum() / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    return sh, cagr, dd, (cagr / -dd if dd < 0 else np.nan)


def managed(b, target, cost=5.0):
    rv = b.rolling(63).std().shift(1) * np.sqrt(252)
    e = (target / rv).clip(upper=1.0)
    wk = pd.Series(e.index.to_period("W"), index=e.index)
    e = e.groupby(wk).transform("first").fillna(0)          # reset weekly
    e1 = e
    r = e1 * b + (1 - e1) * cash - e1.diff().abs().fillna(0) * cost / 1e4
    return r, e1


def static(b, expo, cost=0.0):
    return expo * b + (1 - expo) * cash


periods = {"full 2012-26": px.index >= "2012-01-01",
           "2012-19": (px.index >= "2012-01-01") & (px.index < "2020-01-01"),
           "2020-26": px.index >= "2020-01-01"}
for name, b in baskets.items():
    print(f"\n######## {name} basket ########")
    for pname, mask in periods.items():
        print(f"-- {pname}")
        print(f"   {'':26s} exSharpe   CAGR    maxDD  Calmar")
        for tgt in (0.08, 0.10, 0.12):
            vm, e = managed(b, tgt)
            ae = e[mask].mean()
            st = static(b, ae)
            for lab, r in ((f"vol-managed {tgt:.0%}", vm), (f"static @{ae:.0%} expo", st)):
                sh, c, dd, cal = metrics(r, mask)
                print(f"   {lab:26s} {sh:7.2f}  {c:6.1%}  {dd:6.1%}  {cal:5.2f}")
