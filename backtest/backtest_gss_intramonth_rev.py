"""
Does catching ALL reversion buys intramonth (on top of the monthly momentum
core) help? REV-only-daily failed (perpetually long falling knives), but here
momentum is the always-on driver and REV is an opportunistic overlay that grabs
oversold-in-uptrend dips the monthly scan misses.

  MOM only        : monthly rebalance, momentum sleeve only          [core]
  MOM+REV monthly : both sleeves, monthly only (current design)      [1.38 base]
  MOM + REV intra : MOM membership monthly, REV membership DAILY.
                    Combined book conviction-weighted, 15% cap, renormalized
                    daily; REV exits on RSI2>70 / <200DMA / 21d maxhold.

Daily equity, net@5bps, 2020-26, train/test, turnover so the churn cost shows.
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
WIN, ER_L, ER_HI, ER_LO = 20, 20, 0.40, 0.35
RSI_BUY, RSI_EXIT, REBAL, MAXHOLD, CAP, COST = 15.0, 70.0, 21, 21, 0.15, 5.0


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


def perf(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** (252 / len(r)) - 1
    return sh(np.ones(len(r), bool)), sh(dates < SPLIT), sh(dates >= SPLIT), cagr, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {}; ma50 = {}; ma200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values; ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 200)
    fin = np.isfinite

    def mom_conv(c, i):
        if fin(ER[c][i]) and ER[c][i] >= ER_HI and fin(pv[c][i]) and pv[c][i] > ma200[c][i] \
                and pv[c][i] > ma50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.0)
        return 0.0

    def rev_conv(c, i):
        if fin(ER[c][i]) and ER[c][i] <= ER_LO and fin(pv[c][i]) and pv[c][i] > ma200[c][i] \
                and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return float(np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i])
        return 0.0

    def book(convs):
        sel = {c: v for c, v in convs.items() if v > 0}
        tot = sum(sel.values())
        w = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
        s2 = sum(w.values())
        return {c: v / s2 for c, v in w.items()} if s2 > 0 else {}

    def sim(mode):
        w = {}; mom_fixed = {}; rev_hold = {}; gross = np.zeros(n); turn = np.zeros(n)
        for j in range(start_i, n - 1):
            if (j - start_i) % REBAL == 0:
                mom_fixed = {c: mom_conv(c, j) for c in present}
                mom_fixed = {c: v for c, v in mom_fixed.items() if v > 0 and fin(pv[c][j])}
                if mode != "intra":                                   # REV also refreshed monthly only
                    rev_hold = {c: rev_conv(c, j) for c in present}
                    rev_hold = {c: v for c, v in rev_hold.items() if v > 0 and fin(pv[c][j])}
            if mode == "intra":
                for c in list(rev_hold):                              # daily REV exits
                    if RS2[c][j] > RSI_EXIT or pv[c][j] < ma200[c][j] or (j - rev_hold[c][1]) >= MAXHOLD:
                        del rev_hold[c]
                for c in present:                                     # daily REV entries
                    if c not in rev_hold:
                        cv = rev_conv(c, j)
                        if cv > 0 and fin(pv[c][j]):
                            rev_hold[c] = (cv, j)
            if mode == "mom":
                convs = dict(mom_fixed)
            elif mode == "monthly":
                convs = {**mom_fixed, **rev_hold}
            else:  # intra
                convs = {**mom_fixed, **{c: v[0] for c, v in rev_hold.items()}}
            neww = book(convs)
            turn[j + 1] += sum(abs(neww.get(c, 0.0) - w.get(c, 0.0)) for c in set(neww) | set(w))
            w = neww
            s = 0.0
            for c in w:
                if fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
            gross[j + 1] = s
        sc = slice(start_i + 1, n)
        return gross[sc] - turn[sc] * (COST / 1e4), idx[sc], turn[sc].sum() * 252 / (n - start_i - 1)

    spy = close["SPY"].pct_change().values[start_i + 1:]; dts = idx[start_i + 1:]
    print(f"monthly ({REBAL}d, offset 0), 15% cap, net@{COST}bps, SPLIT {SPLIT.date()}  (daily equity)\n")
    print(f"{'config':18s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s}")
    print("-" * 64)
    f, tr, te, cg, dd = perf(spy, dts)
    print(f"{'SPY buy&hold':18s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {'-':>8s}")
    for mode, lab in [("mom", "MOM only"), ("monthly", "MOM+REV monthly"), ("intra", "MOM + REV intra")]:
        r, d, ty = sim(mode)
        f, tr, te, cg, dd = perf(r, d)
        print(f"{lab:18s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {ty:>7.1f}x")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
