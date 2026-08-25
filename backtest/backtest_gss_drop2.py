"""
If we keep 50 of the 52, which 2 to drop? Leave-one-out: drop each name, measure
the change in full-cycle Sharpe/CAGR/maxDD on the shipped config. Also report how
often each name is actually HELD. The best drops improve (or least harm) the book
- redundant or chronically-unheld names. 2011-26 (all 52 have data since ~2010).
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

DATA_START, TRADE_START, END = "2009-06-01", "2011-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
MOM_CAP, REV_CAP, COST, BUF = 0.075, 0.05, 5.0, 0.03
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


def perf(r):
    d = np.asarray(r, float); d = d[np.isfinite(d)]
    if len(d) < 20 or d.std() == 0:
        return np.nan, np.nan, np.nan
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + d); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(BASE + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    present = [t for t in BASE if t in close.columns and close[t].notna().sum() > 260]
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    spy = close["SPY"].reindex(idx).values; spy200 = close["SPY"].rolling(200).mean().reindex(idx).values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    rv21 = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    rv_med = pd.Series(rv21).rolling(252, min_periods=60).median().values

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].reindex(idx).values
        m50[c] = close[c].rolling(50).mean().reindex(idx).values
        m200[c] = close[c].rolling(200).mean().reindex(idx).values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().reindex(idx).values

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

    def sim(drop=None, track_hold=None):
        uni = [c for c in present if c != drop]
        w = {c: 0. for c in uni}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        hold = {c: 0 for c in uni}; nreb = 0
        for j in range(si, n - 1):
            if j in rd:
                nreb += 1
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                rs = scalar(j); sel = {}; kind = {}
                for c in uni:
                    if riskoff and c not in DEFENSIVE_ASSETS:
                        continue
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
                full = {c: tg.get(c, 0.) for c in uni}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in uni); w = full
                if track_hold is not None:
                    for c in uni:
                        if w[c] > 0.001:
                            hold[c] += 1
            s = 0.
            for c in uni:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        r = ret[sc] - turn[sc] * (COST / 1e4)
        return (r, {c: hold[c] / max(nreb, 1) for c in uni}) if track_hold is not None else r

    base_r, hold_frac = sim(track_hold=True)
    b_sh, b_cg, b_dd = perf(base_r)
    print(f"\nbaseline (52): Sharpe {b_sh:.3f}  CAGR {b_cg:.1%}  maxDD {b_dd:.1%}\n")

    rows = []
    for c in present:
        r = sim(drop=c); sh, cg, dd = perf(r)
        rows.append((c, sh - b_sh, cg - b_cg, dd - b_dd, hold_frac.get(c, 0.0)))
    # best drops: removal raises Sharpe (dSharpe>0) then raises CAGR
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print(f"{'drop':>6s} {'dSharpe':>8s} {'dCAGR':>8s} {'dMaxDD':>8s} {'held%':>7s}")
    print("-" * 44)
    for c, dsh, dcg, ddd, hf in rows[:8]:
        print(f"{c:>6s} {dsh:>+8.3f} {dcg:>+8.2%} {ddd:>+8.2%} {hf:>6.0%}   (best to drop)")
    print("  ...")
    for c, dsh, dcg, ddd, hf in rows[-3:]:
        print(f"{c:>6s} {dsh:>+8.3f} {dcg:>+8.2%} {ddd:>+8.2%} {hf:>6.0%}   (worst to drop=most valuable)")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
