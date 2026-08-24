"""
What's a good ER lookback window? Sweep L in {10,15,20,25,30,40,63} on the
shipped config (adaptive router, ToM rebalance, equity-only buffered gate,
inverse-vol sizing, MOM cap 15%/REV cap 5%, BIL on cash, net@5bps).
ER_HI/ER_LO gates held fixed at 0.40/0.35 - a real caveat, since ER's
distribution shifts with L (longer L -> smoother, less noisy ratio, so a
fixed threshold effectively tightens/loosens the gate). Reported anyway so
the window question is answered on equal footing with everything else
tested this session; full 2007-26 + crisis/chop windows.
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

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
CAP, REV_CAP, COST, BUF = 0.15, 0.05, 5.0, 0.03
ER_WINDOWS = [10, 15, 20, 25, 30, 40, 63]
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("2022 bear", "2022-01-01", "2022-12-31"),
           ("FULL 2007-26", TRADE_START, END), ("DEV 2020-26", "2020-01-01", END)]


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
    tickers = list(TREND_ASSETS.keys()); dl = tickers + ["BIL"]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; VOL = {}
    ER_by_L = {L: {} for L in ER_WINDOWS}
    avg_er_by_L = {L: [] for L in ER_WINDOWS}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2))
        VOL[c] = close[c].pct_change().rolling(63).std().values
        for L in ER_WINDOWS:
            e = ri(er(v.values, L))
            ER_by_L[L][c] = e
            avg_er_by_L[L].append(np.nanmean(e))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def route(ER, c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return None
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return "MOM"
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return "REV"
        return None

    def sim(L):
        ER = ER_by_L[L]
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        n_active = []
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                sel = {}; kind = {}
                for c in present:
                    if riskoff and c not in DEFENSIVE_ASSETS:
                        continue
                    sg = route(ER, c, j)
                    if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                        sel[c] = 1.0 / VOL[c][j]; kind[c] = sg
                n_active.append(len(sel))
                tot = sum(sel.values()); tg = {}
                if tot > 0:
                    for c in sel:
                        tg[c] = min(sel[c] / tot, REV_CAP if kind[c] == "REV" else CAP)
                    s2 = sum(tg.values())
                    if s2 > 0:
                        tg = {c: v / s2 for c, v in tg.items()}
                        tg = {c: min(v, REV_CAP if kind[c] == "REV" else CAP) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, np.mean(n_active) if n_active else 0

    results = {L: sim(L) for L in ER_WINDOWS}
    print(f"ER window sweep (gates fixed at TREND>={ER_HI}/CHOP<={ER_LO}), net@{COST}bps\n")
    print("avg breadth (names selected/rebal): " +
          "  ".join(f"L{L}={results[L][2]:.1f}" for L in ER_WINDOWS))
    print()
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for L in ER_WINDOWS:
            r, dts, _ = results[L]; sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  L={L:<3d}         {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
