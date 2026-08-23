"""
What works in choppy markets? The long-only trend book whipsaws (2011, 2015,
2018). Test the direction-NEUTRAL alternative on the same years: a dollar-neutral
cross-sectional slope*r2 book (long top third / short bottom third), which
doesn't need the market to go anywhere. Compare to the long-only trend book and
SPY, focused on the choppy windows. Monthly (last-trading-day) rebalance,
net@5bps both legs, 2007-2026.

If the market-neutral sort holds up in 2011/2015/2018 while the trend book
bleeds, that's the empirical answer to 'what works in chop'.
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
WIN, ER_HI, ER_LO, RSI_BUY, CAP, COST = 20, 0.40, 0.35, 15.0, 0.15, 5.0
WINDOWS = [("2011 chop", "2011-05-01", "2011-12-31"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END)]


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan); lp = np.log(p)
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def stats(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return np.nan, np.nan, np.nan
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def route(c, i):                                    # long-only adaptive trend book
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return 0.
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.)
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return 0.

    def sim_trend():
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                sel = {c: route(c, j) for c in present}; sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
                tot = sum(sel.values()); tg = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
                s2 = sum(tg.values()); tg = {c: v / s2 for c, v in tg.items()} if s2 > 0 else {}
                full = {c: tg.get(c, 0.) for c in present}; turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s
        sc = slice(si + 1, n); return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    def sim_ls(rebal):                                   # dollar-neutral slope*r2 long-short
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rebal:
                sc_ = {c: SL[c][j] * R2[c][j] for c in present
                       if fin(SL[c][j]) and fin(R2[c][j]) and fin(pv[c][j]) and fin(pv[c][j + 1])}
                names = sorted(sc_, key=sc_.get); k = max(1, len(names) // 3)
                longs, shorts = names[-k:], names[:k]
                new = {c: 0. for c in present}
                for c in longs:
                    new[c] = 0.5 / len(longs)
                for c in shorts:
                    new[c] = -0.5 / len(shorts)
                turn[j + 1] += sum(abs(new[c] - w[c]) for c in present); w = new
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s
        sc = slice(si + 1, n); return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    rd10 = set(range(si, n - 1, 10))                     # ~2-week rebalance (native horizon of the signal)
    tr, dts = sim_trend(); ls_m, _ = sim_ls(rd); ls_2w, _ = sim_ls(rd10)
    spy = close["SPY"].pct_change().values[si + 1:]
    series = [("trend (long-only)", tr), ("slope*r2 L/S monthly", ls_m),
              ("slope*r2 L/S 2-week", ls_2w), ("SPY", spy)]
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for nm, r in series:
            sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {nm:24s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
