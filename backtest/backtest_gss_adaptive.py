"""
Is there an edge in DETECTING mean-reversion vs trend per asset and applying
the matching rule?

Detector: Kaufman efficiency ratio ER = |net move| / sum|daily moves| over 20d
  ER >= 0.50  -> TRENDING   -> momentum rule
  ER <= 0.35  -> CHOPPY     -> mean-reversion rule
Rules (long-only, all require price > 200-SMA = don't catch knives):
  momentum  : long if price>50-SMA and 20d slope>0
  reversion : long if RSI(2) < 15 (buy the oversold dip)

Compare, weekly (hold 5d), equal-weight the longs, 2020-2026, train/test:
  mom       momentum rule on every asset (ignore state)
  rev       reversion rule on every asset (ignore state)
  union     long if EITHER rule fires (no detection)
  adaptive  ER-gated: momentum in trend state, reversion in chop state
If detection has edge, adaptive > mom, rev, and union.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

DATA_START, TRADE_START, END = "2019-01-01", "2020-01-01", "2026-08-21"
SPLIT = pd.Timestamp("2023-06-01")
WIN, FWD, ER_L = 20, 5, 20
ER_HI, ER_LO, RSI_BUY = 0.50, 0.35, 15.0


def roll_slope(prices, win):
    n = len(prices); slope = np.full(n, np.nan)
    lp = np.log(prices)
    if n < win or sliding_window_view is None:
        return slope
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); denom = float(xc @ xc)
    slope[win - 1:] = ((W - W.mean(1)[:, None]) @ xc / denom) * 1000.0
    return slope


def rsi(prices, p):
    d = np.diff(prices, prepend=prices[0])
    up = pd.Series(np.where(d > 0, d, 0.0)); dn = pd.Series(np.where(d < 0, -d, 0.0))
    ru = up.rolling(p).mean(); rd = dn.rolling(p).mean()
    return (100 - 100 / (1 + ru / rd.replace(0, np.nan))).values


def eff_ratio(prices, L):
    s = pd.Series(prices)
    net = (s - s.shift(L)).abs()
    path = s.diff().abs().rolling(L).sum()
    return (net / path.replace(0, np.nan)).values


def perf(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * (252 / FWD)) / (x.std() * np.sqrt(252 / FWD)) if len(x) > 10 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** ((252 / FWD) / len(r)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh(np.ones(len(r), bool)), sh(dates < SPLIT), sh(dates >= SPLIT), cagr, dd


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {}; ma50 = {}; ma200 = {}; sl = {}; r2i = {}; er = {}
    for c in present:
        v = close[c].dropna()
        reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        pv[c] = close[c].values
        ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        sl[c] = reidx(roll_slope(v.values, WIN))
        r2i[c] = reidx(rsi(v.values, 2))
        er[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def mom(c, i):
        return pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i] and sl[c][i] > 0

    def rev(c, i):
        return pv[c][i] > ma200[c][i] and fin(r2i[c][i]) and r2i[c][i] < RSI_BUY

    def longs(c, i, scheme):
        e = er[c][i]
        if scheme == "mom":
            return mom(c, i)
        if scheme == "rev":
            return rev(c, i)
        if scheme == "union":
            return mom(c, i) or rev(c, i)
        # adaptive
        if fin(e) and e >= ER_HI:
            return mom(c, i)
        if fin(e) and e <= ER_LO:
            return rev(c, i)
        return False

    def run(scheme):
        dates, rr, nm = [], [], []
        for i in range(max(start_i, 200), n - FWD, FWD):
            el = [c for c in present if longs(c, i, scheme) and fin(pv[c][i]) and fin(pv[c][i + FWD])]
            dates.append(idx[i]); nm.append(len(el))
            rr.append(np.mean([pv[c][i + FWD] / pv[c][i] - 1 for c in el]) if el else 0.0)
        return np.array(rr), pd.DatetimeIndex(dates), np.mean(nm)

    print(f"detector ER{ER_L} (>= {ER_HI} trend / <= {ER_LO} chop), hold {FWD}d, SPLIT {SPLIT.date()}\n")
    print(f"{'scheme':10s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'names':>6s}")
    print("-" * 56)
    spy = close["SPY"]; sp = []
    for i in range(max(start_i, 200), n - FWD, FWD):
        sp.append(spy.values[i + FWD] / spy.values[i] - 1)
    f, tr, te, cg, dd = perf(np.array(sp), pd.DatetimeIndex([idx[i] for i in range(max(start_i, 200), n - FWD, FWD)]))
    print(f"{'SPY B&H':10s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {'-':>6s}")
    for scheme in ["mom", "rev", "union", "adaptive"]:
        rr, d, avgn = run(scheme)
        f, tr, te, cg, dd = perf(rr, d)
        print(f"{scheme:10s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {avgn:>6.1f}")
    print("\n(gross; all reconstitute weekly so turnover/costs hit them ~equally.)")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
