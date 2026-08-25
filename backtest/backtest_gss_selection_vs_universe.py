"""
Is the fully-deployed adaptive book's ~30% CAGR a SELECTION edge, or just the
effect of a wide high-beta universe? Fair benchmarks (no hindsight QQQ):
  SPY            - the naive default you'd hold with no view
  equalwt_hold   - equal-weight buy-and-hold of the SAME 51-name universe,
                   rebalanced monthly. "Own the universe, no selection/timing."
  adaptive_full  - ER router selection (MOM/REV), equal-weight among selected,
                   renormalized to ~100% deployed (cash only when nothing
                   qualifies), monthly ToM rebalance. The ~30% version.

If adaptive_full beats equalwt_hold -> the trend SELECTION adds value.
If it only matches -> the edge is the universe, not the timing.
Focus 2020-2026 (the regime that matters, per the user) + 2020-23/2024-26
splits (is it all crypto?) + 2011-2019 shown small as the honest OOS.
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

DATA_START, END = "2010-06-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY, COST = 20, 0.40, 0.35, 15.0, 5.0
BASE = list(TREND_ASSETS.keys())


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
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


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return (np.nan,) * 4
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    dn_ = d[d < 0]
    so = (d.mean() * 252) / (np.sqrt((dn_ ** 2).mean()) * np.sqrt(252)) if len(dn_) > 4 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, so, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(BASE + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in BASE if t in close.columns and close[t].notna().sum() > 200]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; avail = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].reindex(idx).values
        m50[c] = close[c].rolling(50).mean().reindex(idx).values
        m200[c] = close[c].rolling(200).mean().reindex(idx).values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        avail[c] = close[c].reindex(idx).notna().values
    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return None
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return "MOM"
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return "REV"
        return None

    def conv(c, j, kind):
        if kind == "REV":
            return max((RSI_BUY - RS2[c][j]) / RSI_BUY, 0.0) * R2[c][j]
        return max(SL[c][j] * R2[c][j], 0.0)

    def sim_cost(mode):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                if mode == "equalwt_hold":
                    sel = [c for c in present if avail[c][j] and fin(pv[c][j])]
                    neww = {c: (1.0 / len(sel) if c in sel else 0.0) for c in present} if sel else {c: 0. for c in present}
                else:
                    routed = {c: route(c, j) for c in present}
                    sel = [c for c in present if routed[c] and fin(pv[c][j])]
                    if not sel:
                        neww = {c: 0. for c in present}
                    elif mode == "adaptive_equalwt":
                        neww = {c: (1.0 / len(sel) if c in sel else 0.0) for c in present}
                    else:  # adaptive_conv: weight ~ conviction (slope*r2 / oversold depth)
                        cv = {c: max(conv(c, j, routed[c]), 0.0) for c in sel}
                        tot = sum(cv.values())
                        neww = {c: (cv.get(c, 0.0) / tot if tot > 0 else 0.0) for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    spy = close["SPY"].pct_change().values[si + 1:]; dts = idx[si + 1:].values
    res = {"adaptive_conv": sim_cost("adaptive_conv"),
           "adaptive_equalwt": sim_cost("adaptive_equalwt"),
           "equalwt_hold": sim_cost("equalwt_hold")}

    windows = [("2020-2026 (regime that matters)", "2020-01-01", END),
               ("  2020-2023", "2020-01-01", "2023-12-31"),
               ("  2024-2026", "2024-01-01", END),
               ("2011-2019 (honest OOS)", "2011-01-01", "2019-12-31")]
    print(f"{'window':>34s} {'strategy':>14s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s}")
    print("-" * 88)
    for lab, lo, hi in windows:
        for name in ["adaptive_conv", "adaptive_equalwt", "equalwt_hold"]:
            r, d = res[name]; sh, so, cg, dd = perf(r, d, lo, hi)
            print(f"{lab:>34s} {name:>16s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, so, cg, dd = perf(spy, dts, lo, hi)
        print(f"{'':>34s} {'SPY':>16s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
