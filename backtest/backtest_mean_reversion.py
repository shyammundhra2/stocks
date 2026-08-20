"""
Backtest the two mean-reversion signals already coded in macro/indicators.py,
using their EXACT production rules, over a long multi-regime window (2007-2026,
incl. 2008 + 2022 bears - the out-of-regime test).

  RSI2-QQQ  (mirrors get_mean_reversion): long QQQ when RSI(2) <= 10 AND price >
            200-SMA; exit when RSI(2) >= 70; flat ("RISK OFF") when price < 200-SMA.
  VIX-SPY   (mirrors get_vix_signal): VIX z-score vs trailing 50d. Buy-the-fear:
            z>4 AGGRESSIVE_BUY / z>2 SCALE_IN -> long SPY; z<-1.5 AGGRESSIVE_TRIM
            or (z<-1 and breadth failing) -> flat; else hold. Same SPY trend /
            RSP-SPY breadth gates as production.

Positions are 0/1 (long or flat); idle capital earns BIL (1-3mo T-bills, total
return). Reported vs QQQ and SPY buy & hold. Costless (signals trade a few
times/month); a small per-switch cost is applied as a sensitivity.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.helpers import compute_RSI
from macro.indicators import _trend_stats

START = "2007-01-01"
END = "2026-07-10"
COST_BPS = 0.0        # per position switch (set >0 for sensitivity)


def perf(r, label):
    r = pd.Series(r).dropna()
    n = len(r)
    yrs = n / 252
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    print(f"{label:26s} CAGR={cagr:6.1%}  vol={vol:5.1%}  Sharpe={sharpe:5.2f}  "
          f"maxDD={dd:6.1%}  n={n}")
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd)


def apply_cost(pos, gross):
    if COST_BPS <= 0:
        return gross
    switches = pos.diff().abs().fillna(0)
    return gross - switches * (COST_BPS / 1e4)


def main():
    t0 = time.time()
    tickers = ["QQQ", "SPY", "^VIX", "RSP", "BIL"]
    print(f"Downloading {tickers} {START}..{END} ...")
    raw = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"].ffill()
    qqq, spy, vix, rsp, bil = (close[c] for c in ["QQQ", "SPY", "^VIX", "RSP", "BIL"])
    idx = close.index

    cash = bil.pct_change()               # BIL total-return = cash yield
    qqq_ret = qqq.pct_change()
    spy_ret = spy.pct_change()

    # ---- RSI2-QQQ (get_mean_reversion rules) ----
    rsi2 = compute_RSI(qqq, 2)
    sma200 = qqq.rolling(200, min_periods=200).mean()
    pos_rsi = pd.Series(0.0, index=idx)
    cur = 0.0
    for t in range(len(idx)):
        price, r2v, sma = qqq.iloc[t], rsi2.iloc[t], sma200.iloc[t]
        if np.isnan(sma) or np.isnan(r2v):
            cur = 0.0
        elif price < sma:
            cur = 0.0                      # RISK OFF
        elif r2v <= 10:
            cur = 1.0                      # BUY
        elif r2v >= 70:
            cur = 0.0                      # EXIT
        # else HOLD -> keep cur
        pos_rsi.iloc[t] = cur

    # ---- VIX-SPY (get_vix_signal rules) ----
    vix_mean = vix.rolling(50).mean()
    vix_std = vix.rolling(50).std()
    z = (vix - vix_mean) / vix_std
    spy_ma50 = spy.rolling(50).mean()
    spy_ma200 = spy.rolling(200).mean()
    ratio = rsp / spy
    ratio_ma50 = ratio.rolling(50).mean()
    # pos_vix  = production rules as-is
    # pos_vixf = same, but EVERY buy also requires price > 200-SMA (crash filter)
    pos_vix = pd.Series(0.0, index=idx)
    pos_vixf = pd.Series(0.0, index=idx)
    cur = curf = 0.0
    for t in range(len(idx)):
        zt = z.iloc[t]
        if np.isnan(zt) or np.isnan(spy_ma200.iloc[t]):
            cur = curf = 0.0
            pos_vix.iloc[t] = cur
            pos_vixf.iloc[t] = curf
            continue
        spy_last = spy.iloc[t]
        above_200 = spy_last > spy_ma200.iloc[t]
        in_uptrend = spy_last > spy_ma50.iloc[t] and above_200
        breadth_failing = ratio.iloc[t] < ratio_ma50.iloc[t]

        if zt > 4.0:
            sig = "BUY"
        elif zt > 2.0:
            sig = "BUY" if in_uptrend else "HOLD"
        elif zt < -1.5:
            sig = "TRIM"
        elif zt < -1.0 and breadth_failing:
            sig = "TRIM"
        else:
            sig = "HOLD"

        if sig == "BUY":
            cur = 1.0
        elif sig == "TRIM":
            cur = 0.0
        pos_vix.iloc[t] = cur

        # filtered variant: buys gated on 200-SMA; also exit if it drops below
        if sig == "BUY":
            curf = 1.0 if above_200 else 0.0
        elif sig == "TRIM" or not above_200:
            curf = 0.0
        pos_vixf.iloc[t] = curf

    # ---- returns: position earns next-day asset return, else cash ----
    def strat(pos, asset_ret):
        p = pos.shift(1).fillna(0.0)        # act on close t, earn t->t+1 (no lookahead)
        gross = p * asset_ret + (1 - p) * cash
        return apply_cost(pos.shift(1).fillna(0.0), gross)

    r_rsi = strat(pos_rsi, qqq_ret)
    r_vix = strat(pos_vix, spy_ret)
    r_vixf = strat(pos_vixf, spy_ret)
    r_combo = 0.5 * r_rsi + 0.5 * r_vixf

    print(f"\nPeriods: {len(idx)} ({idx[0].date()} -> {idx[-1].date()})")
    print(f"Time in market: RSI2-QQQ {pos_rsi.mean():.0%}, VIX-SPY {pos_vix.mean():.0%}, "
          f"VIX-SPY+200SMA {pos_vixf.mean():.0%}\n")
    perf(qqq_ret, "QQQ buy & hold")
    perf(spy_ret, "SPY buy & hold")
    perf(cash, "BIL cash")
    print("-" * 72)
    perf(r_rsi, "RSI2-QQQ mean reversion")
    perf(r_vix, "VIX-SPY (as-is)")
    perf(r_vixf, "VIX-SPY + 200SMA filter")
    perf(r_combo, "Combined RSI2 + VIXf")

    # sub-period edge (does it survive each regime?)
    print("\nBy period (Sharpe):")
    print(f"{'window':12s} {'QQQ':>7s} {'RSI2':>7s} {'SPY':>7s} {'VIX':>7s} {'VIX+flt':>8s}")
    for lo, hi in [("2007", "2009"), ("2010", "2015"), ("2016", "2019"),
                   ("2020", "2021"), ("2022", "2022"), ("2023", "2026")]:
        m = (idx >= f"{lo}-01-01") & (idx <= f"{hi}-12-31")
        def sh(x):
            xx = pd.Series(x)[m].dropna()
            v = xx.std() * np.sqrt(252)
            return (xx.mean() * 252) / v if v > 0 else np.nan
        print(f"{lo+'-'+hi:12s} {sh(qqq_ret):>7.2f} {sh(r_rsi):>7.2f} "
              f"{sh(spy_ret):>7.2f} {sh(r_vix):>7.2f} {sh(r_vixf):>8.2f}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
