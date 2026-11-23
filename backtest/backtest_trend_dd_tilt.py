"""
Tilt-weight version of the drawdown-phase idea (backtest_trend_coming_out_of_correction.py
found the hard GATE version - trade ONLY names in the -15%/-5% dd zone - loses
to a plain always-on trend sleeve outside slow 2008-style grinds, because it
throws away too much opportunity breadth in fast V-recoveries (2020) and
choppy/rotational markets (2022)).

This version keeps the FULL always_trend opportunity set (any name, any dd,
mom>0) but multiplies the ranking/sizing score by a tilt factor when a name
is in the -15%/-5% zone where the IC was strongest - up-weighting conviction
there without excluding everything else. Tests tilt=1.5x and tilt=2.0x
against always_trend (tilt=1.0, i.e. no tilt) and eq_hold.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
CAP, COST, TOP_N = 0.15, 5.0, 8
TILTS = [1.0, 1.5, 2.0]   # 1.0 = always_trend baseline, no tilt
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2020 COVID", "2020-02-01", "2020-12-31"),
           ("2022 bear", "2022-01-01", "2022-12-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END)]


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
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 300]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values if "BIL" in close.columns else np.zeros(n)
    spy_ret = close["SPY"].pct_change().values
    print(f"Universe: {len(present)}\n")

    pv = {}; dd = {}; mom = {}
    for c in present:
        p = close[c].values
        pv[c] = p
        hi = pd.Series(p).rolling(252, min_periods=60).max().values
        dd[c] = p / hi - 1.0
        m = np.full(n, np.nan)
        for i in range(252, n):
            m[i] = p[i - 21] / p[i - 252] - 1.0
        mom[c] = m

    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 260)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def sim(tilt):
        w = {c: 0. for c in present}; prev = {c: 0. for c in present}
        ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                cands = []
                for c in present:
                    if not (fin(pv[c][j]) and fin(dd[c][j]) and fin(mom[c][j])) or mom[c][j] <= 0:
                        continue
                    in_zone = -0.15 < dd[c][j] <= -0.05
                    score = mom[c][j] * (tilt if in_zone else 1.0)
                    cands.append((c, score))
                cands.sort(key=lambda x: x[1], reverse=True)
                sel = cands[:TOP_N]
                tot = sum(v for _, v in sel)
                tg = {c: min(v / tot, CAP) for c, v in sel} if tot > 0 else {}
                s2 = sum(tg.values())
                tg = {c: v / s2 for c, v in tg.items()} if s2 > 0 else {}
                w = {c: tg.get(c, 0.) for c in present}
            turn[j + 1] += sum(abs(w[c] - prev[c]) for c in present)
            prev = dict(w)
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            cash = 1.0 - sum(w.values())
            ret[j + 1] = s + cash * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    def eq_hold_sim():
        w = {c: 0. for c in present}; prev = {c: 0. for c in present}
        ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                elig = [c for c in present if fin(pv[c][j])]
                tg = {c: 1.0 / len(elig) for c in elig} if elig else {}
                w = {c: tg.get(c, 0.) for c in present}
            turn[j + 1] += sum(abs(w[c] - prev[c]) for c in present)
            prev = dict(w)
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            cash = 1.0 - sum(w.values())
            ret[j + 1] = s + cash * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    results = {f"tilt_{t:.1f}x": sim(t) for t in TILTS}
    results["eq_hold"] = eq_hold_sim()
    print("tilt_1.0x = always_trend baseline (no tilt); tilt_1.5x/2.0x up-weight names")
    print("currently in the -15%/-5% dd zone by that multiplier, full universe still eligible\n")
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for k in list(results.keys()):
            r, dts = results[k]; sh, cg, mdd = stats(r, dts, lo, hi)
            print(f"  {k:14s} {sh:>7.2f} {cg:>7.1%} {mdd:>7.1%}")
        sh, cg, mdd = stats(spy_ret[si + 1:], results['eq_hold'][1], lo, hi)
        print(f"  {'SPY':14s} {sh:>7.2f} {cg:>7.1%} {mdd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
