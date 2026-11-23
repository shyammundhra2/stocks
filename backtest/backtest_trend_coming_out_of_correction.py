"""
Turns the trend-vs-MR-by-phase IC finding (backtest_trend_vs_mr_by_dd_phase.py)
into an actual strategy: go long names specifically in the "moderate recovery"
drawdown phase (-15% < dd <= -5% off their own 252d high - the "coming out of
a severe correction, not yet at highs" zone where trend's IC peaked at +0.046,
clearly ahead of every other phase/signal combination), sized by trend strength,
rest of book in cash. Compares against:
  always_trend - the SAME trend signal/sizing with NO phase gate (long whenever
                  mom>0, any dd) - isolates whether the phase gate itself adds
                  value over just always following trend
  eq_hold       - equal-weight buy-and-hold across the same 42-name universe
  spy           - SPY buy-and-hold

Monthly rebalance, top-N by trend-signal rank among phase-eligible names each
month, equal-weight among selected (capped 15%/name), net@5bps. Full 2007-26,
same TREND_ASSETS universe and crisis windows as the sibling backtests.
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

    def sim(mode):
        """mode: 'phase_gated' | 'always_trend' | 'eq_hold'"""
        w = {c: 0. for c in present}; prev = {c: 0. for c in present}
        ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                if mode == "eq_hold":
                    elig = [c for c in present if fin(pv[c][j])]
                    tg = {c: 1.0 / len(elig) for c in elig} if elig else {}
                else:
                    cands = []
                    for c in present:
                        if not (fin(pv[c][j]) and fin(dd[c][j]) and fin(mom[c][j])) or mom[c][j] <= 0:
                            continue
                        if mode == "phase_gated" and not (-0.15 < dd[c][j] <= -0.05):
                            continue
                        cands.append((c, mom[c][j]))
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

    configs = ["phase_gated", "always_trend", "eq_hold"]
    results = {k: sim(k) for k in configs}
    print(f"phase_gated: long top-{TOP_N} by momentum among names in the -15%/-5% dd zone only")
    print(f"always_trend: same momentum signal/sizing, no dd-zone gate")
    print(f"eq_hold: equal-weight buy-and-hold, same universe\n")
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for k in configs:
            r, dts = results[k]; sh, cg, mdd = stats(r, dts, lo, hi)
            print(f"  {k:14s} {sh:>7.2f} {cg:>7.1%} {mdd:>7.1%}")
        sh, cg, mdd = stats(spy_ret[si + 1:], results['eq_hold'][1], lo, hi)
        print(f"  {'SPY':14s} {sh:>7.2f} {cg:>7.1%} {mdd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
