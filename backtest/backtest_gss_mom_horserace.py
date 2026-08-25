"""
Momentum SIGNAL horse-race on the live universe. All strategies share the SAME
machinery (universe, >200DMA trend filter, turn-of-month rebalance, top-K equal
weight, net@5bps) and differ ONLY in the ranking signal. Question: does any
literature momentum variant beat the current slope*r2 in BOTH 2011-19 and
2020-26 (regime robustness), not just the recent window?

Signals tested:
  mom126   total-return momentum, 126d skip last 21d   (Jegadeesh-Titman baseline)
  slope_r2 rolling OLS slope * r2, 63d                 (CURRENT engine metric)
  resid    residual/idiosyncratic momentum, 126d skip 21 (Blitz-Huij-Martens 2011)
  hi52     price / 252d high                            (George-Hwang 2004)
  sharpe   risk-adjusted momentum = mean/std of 126d daily ret

Honest read: a signal only 'wins' if it beats slope_r2 net-of-cost in both
sub-periods AND doesn't blow up maxDD. One-window wins are regime luck.
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

UNIV = list(TREND_ASSETS.keys())
DATA_START, END = "2010-06-01", "2026-08-21"
TOPK, COST = 8, 5.0            # hold top-8, 5bps per unit turnover
SL_WIN = 63                    # slope*r2 window
MOM_LOOK, MOM_SKIP = 126, 21   # 6mo momentum, skip last month
BETA_WIN = 126                 # rolling beta for residual momentum


def roll_slope_r2(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn
    pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


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
    dl = sorted(set(UNIV + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    spy_ret = close["SPY"].pct_change().reindex(idx).values

    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]
    pv = {}; m200 = {}; SLR2 = {}; MOM = {}; HI52 = {}; SHP = {}; RESID = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m200[c] = px.rolling(200).mean().values
        s, r = roll_slope_r2(px.values, SL_WIN); SLR2[c] = s * r
        ret = px.pct_change().values
        # total-return momentum, skip last MOM_SKIP days
        MOM[c] = pv[c] / np.concatenate([np.full(MOM_LOOK, np.nan), pv[c][:-MOM_LOOK]]) - 1.0
        MOM[c] = np.concatenate([np.full(MOM_SKIP, np.nan), MOM[c][:-MOM_SKIP]])
        HI52[c] = pv[c] / px.rolling(252).max().values
        SHP[c] = (pd.Series(ret).rolling(MOM_LOOK).mean() /
                  pd.Series(ret).rolling(MOM_LOOK).std().replace(0, np.nan)).values
        # residual momentum: resid = ret - beta*spy over rolling window, then
        # cumulate residual over MOM_LOOK skip MOM_SKIP
        rr = pd.Series(ret); ss = pd.Series(spy_ret)
        cov = rr.rolling(BETA_WIN).cov(ss); var = ss.rolling(BETA_WIN).var()
        beta = (cov / var.replace(0, np.nan)).values
        resid = ret - beta * spy_ret
        cumres = pd.Series(resid).rolling(MOM_LOOK).sum().values
        RESID[c] = np.concatenate([np.full(MOM_SKIP, np.nan), cumres[:-MOM_SKIP]])

    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    SIG = {"mom126": MOM, "slope_r2": SLR2, "resid": RESID, "hi52": HI52, "sharpe": SHP}

    def sim(sig):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                # eligible: above 200DMA and has a finite signal
                elig = [(sig[c][j], c) for c in present
                        if fin(pv[c][j]) and fin(m200[c][j]) and pv[c][j] > m200[c][j] and fin(sig[c][j])]
                elig.sort(reverse=True)
                sel = [c for _, c in elig[:TOPK]]
                neww = {c: (1.0 / len(sel) if c in sel else 0.0) for c in present} if sel else {c: 0. for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    res = {name: sim(SIG[name]) for name in SIG}
    spy = spy_ret[si + 1:]; dts = idx[si + 1:].values
    print(f"\nUniverse {len(present)}, top-{TOPK} equal-wt, >200DMA filter, ToM rebal, net@{COST:.0f}bps\n")
    for lab, lo, hi in [("2011-2019", "2011-01-01", "2019-12-31"),
                        ("2020-2026", "2020-01-01", END),
                        ("FULL 2011-26", "2011-01-01", END)]:
        print(f"== {lab} ==   {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s}")
        for name in SIG:
            r, d = res[name]; sh, so, cg, dd = perf(r, d, lo, hi)
            star = " *" if name == "slope_r2" else "  "
            print(f"  {name:10s}{star} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, so, cg, dd = perf(spy, dts, lo, hi)
        print(f"  {'SPY':10s}   {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
