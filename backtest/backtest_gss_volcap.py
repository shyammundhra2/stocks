"""
What's the best portfolio vol cap? This is the actual Kelly-fraction lever
(f* ~ Sharpe as a vol target; Sharpe~0.75 -> full Kelly ~75% vol, quarter
~19%, eighth ~9.4%). Current ceiling 13.5% (~1/5.5 Kelly). Sweep the vol
ceiling on the shipped config and see where risk-adjusted return peaks and
what it costs in drawdown - the survival objective cares about the DD, not
just the Sharpe.

Sweep max_portfolio_vol in {8,10,11.5,13.5,16,20}% (min floor scales with it).
Shipped config: adaptive router, ToM, equity-only gate, inverse-vol, MOM cap
7.5%/REV 5%, vol-target scalar, BIL cash, net@5bps. 2007-26 + crises,
Sharpe/Sortino/CAGR/maxDD + realized vol + deployment.
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
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

DATA_START, TRADE_START, END = "2004-01-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
MOM_CAP, REV_CAP, COST, BUF = 0.075, 0.05, 5.0, 0.03
# vol caps to sweep (ceiling); floor set to 0.6*ceiling
VOLCAPS = [0.08, 0.10, 0.115, 0.135, 0.16, 0.20]
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2020 COVID", "2020-02-01", "2020-04-30"),
           ("2022 bear", "2022-01-01", "2022-12-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END)]


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
        return (np.nan,) * 5
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    dn_ = d[d < 0]
    so = (d.mean() * 252) / (np.sqrt((dn_ ** 2).mean()) * np.sqrt(252)) if len(dn_) > 4 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    rv = d.std() * np.sqrt(252)
    return sh, so, cg, dd, rv


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys()); dl = tickers + ["BIL"]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    rets_np = close[present].pct_change().values
    col = {c: k for k, c in enumerate(present)}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    rv21 = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    rv_med = pd.Series(rv21).rolling(252, min_periods=60).median().values

    def scalar(j):
        if not fin(rv21[j]) or not fin(rv_med[j]) or rv21[j] <= 0:
            return 1.0
        return float(np.clip(rv_med[j] / rv21[j], 0.25, 1.0))

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

    def book_vol(tg, j):
        if not tg:
            return 0.0
        wv = np.zeros(len(present))
        for c, x in tg.items():
            wv[col[c]] = x
        window = rets_np[j - 62:j + 1]
        br = np.nansum(window * wv[None, :], axis=1)
        return float(np.std(br) * np.sqrt(252))

    def sim(vol_cap):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                rs = scalar(j); sel = {}; kind = {}
                if not riskoff:
                    for c in present:
                        sg = route(c, j)
                        if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                            sel[c] = 1.0 / VOL[c][j]; kind[c] = sg
                tot = sum(sel.values()); tg = {}
                if tot > 0:
                    for c in sel:
                        tg[c] = min(sel[c] / tot, REV_CAP if kind[c] == "REV" else MOM_CAP)
                    s2 = sum(tg.values())
                    if s2 > 0:
                        tg = {c: v / s2 for c, v in tg.items()}
                        tg = {c: min(v, REV_CAP if kind[c] == "REV" else MOM_CAP) for c, v in tg.items()}
                    tg = {c: (v * rs if kind[c] == "MOM" else v) for c, v in tg.items()}
                    # enforce vol cap: scale the whole book down if 63d vol exceeds ceiling
                    bv = book_vol(tg, j)
                    if bv > vol_cap and bv > 0:
                        tg = {c: v * (vol_cap / bv) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    results = {v: sim(v) for v in VOLCAPS}
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s} {'realVol':>8s}")
        for v in VOLCAPS:
            r, dts = results[v]; sh, so, cg, dd, rv = perf(r, dts, lo, hi)
            print(f"  cap {int(v*1000)/10:>4.1f}%       {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%} {rv*100:>7.1f}%")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
