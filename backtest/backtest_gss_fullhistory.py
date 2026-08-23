"""
OUT-OF-REGIME test: does the adaptive turn-of-month book survive 2007-2019 -
the period it was NOT developed on, including 2008 GFC, 2015-16, 2018 Q4?

Everything so far is 2020-26 (one regime, dozens of exploratory cuts). If the
defensive profile (>1 Sharpe, shallow DD) holds through 2008, it's a real
product; if it collapses, the 2020s numbers are an artifact.

Same rules: ER router (TREND>=0.40 MOM / CHOP<=0.35 REV), last-trading-day
rebalance, 15% cap, net@5bps. Two funding models (full ~100% / 50%+BIL).
Reports full period + the three stress windows. Universe shrinks pre-2010 (many
ETFs post-date it) - avg names shown so breadth is visible.
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

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, CAP, COST, DEPLOY_CAP = 20, 20, 0.40, 0.35, 15.0, 0.15, 5.0, 0.50
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"),
           ("2015-16 selloff", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"),
           ("2020 COVID", "2020-02-01", "2020-04-30"),
           ("2022 bear", "2022-01-01", "2022-10-31"),
           ("PRE 2007-2019 (OOS)", "2007-01-01", "2020-01-01"),
           ("DEV 2020-2026", "2020-01-01", END),
           ("FULL 2007-2026", TRADE_START, END)]


def roll_sr(prices, win):
    n = len(prices); slope = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < win or sliding_window_view is None:
        return slope, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); denom = float(xc @ xc)
    ym = W.mean(1); slc = (W - ym[:, None]) @ xc / denom
    pred = slc[:, None] * x[None, :] + (ym - slc * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pred) ** 2).sum(1)
    slope[win - 1:] = slc * 1000.0
    r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return slope, r2


def rsi(prices, p):
    d = np.diff(prices, prepend=prices[0])
    up = pd.Series(np.where(d > 0, d, 0.0)); dn = pd.Series(np.where(d < 0, -d, 0.0))
    return (100 - 100 / (1 + up.rolling(p).mean() / dn.rolling(p).mean().replace(0, np.nan))).values


def eff_ratio(prices, L):
    s = pd.Series(prices)
    return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def stats(r, dates, lo, hi):
    m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
    x = np.asarray(r, float)[m]; d = x[np.isfinite(x)]
    if len(d) < 20 or d.std() == 0:
        return np.nan, np.nan, np.nan
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(x[np.isfinite(x)]))
    cagr = eq[-1] ** (252 / len(d)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh, cagr, dd


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    dl = tickers + (["BIL"] if "BIL" not in tickers else [])
    print(f"Downloading {len(dl)} tickers (+BIL) 2005-2020 ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    bil = close["BIL"].pct_change().fillna(0).values if "BIL" in close.columns else np.zeros(n)
    print(f"Universe with any history: {len(present)}\n")

    pv = {}; ma50 = {}; ma200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values; ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    fin = np.isfinite

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return 0.0
        a200 = pv[c][i] > ma200[c][i]
        if e >= ER_HI and a200 and pv[c][i] > ma50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.0)
        if e <= ER_LO and a200 and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return 0.0

    dser = pd.Series(idx); rebal_days = {g.index[-1] for _, g in dser.groupby(dser.dt.to_period("M"))
                                        if g.index[-1] >= start_i}

    w = {c: 0.0 for c in present}; eqret = np.zeros(n); turn = np.zeros(n); invested = np.zeros(n); nn = []
    for j in range(start_i, n - 1):
        if j in rebal_days:
            sel = {c: route(c, j) for c in present}
            sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
            tot = sum(sel.values())
            tgt = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
            s2 = sum(tgt.values()); tgt = {c: v / s2 for c, v in tgt.items()} if s2 > 0 else {}
            full = {c: tgt.get(c, 0.0) for c in present}
            turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            nn.append(sum(1 for v in w.values() if v > 0))
        s = 0.0
        for c in present:
            if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
        eqret[j + 1] = s; invested[j + 1] = sum(w.values())

    sc = slice(start_i + 1, n); dts = idx[sc].values
    eq_net = eqret[sc] - turn[sc] * (COST / 1e4); inv = invested[sc]; bilc = bil[sc]
    full_ret = eq_net + bilc * (1.0 - inv)
    cap_ret = eq_net * DEPLOY_CAP + bilc * (1.0 - DEPLOY_CAP * inv)
    spy_ret = close["SPY"].pct_change().values[sc]

    print(f"FULL HISTORY 2007-2026. last-trading-day rebalance, net@{COST}bps")
    print(f"deployment: invested {float((inv>0).mean()):.0%} of days, avg exposure {inv.mean():.0%}, "
          f"avg {np.mean(nn):.1f} names\n")
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ({lo} -> {hi}) ==")
        print(f"  {'series':22s} {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for nm, r in [("adaptive full ~100%", full_ret), ("adaptive 50%+BIL", cap_ret), ("SPY buy&hold", spy_ret)]:
            sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {nm:22s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
