"""
Lighter version of the iso-curve test (backtest_trend_sleeve_isocurve_strategy.py,
which required conv=|slope|xR^2>=4 and found that too narrow - cash-drag
from only ~6% of events qualifying made it a clear reject as a full
strategy). This tests a much milder filter on the standalone 12-1 momentum
sleeve: m12>0 AND the 1-month (20d) slope is ALSO positive - no conviction
threshold, just sign agreement between the long (12-1) and short (20d)
trend measures.

Variants, same monthly-rebalance/cap/cost rigor as backtest_trend_dd_tilt.py:
  baseline      - m12>0 only, top-8 by momentum (today's live sleeve rule)
  slope_filter  - m12>0 AND slope(20,scale=10)>0, top-8 by momentum among
                  that eligible set (no backfill if <8 qualify)
  slope_ddtilt  - slope_filter's eligibility + the validated 2x dd-zone tilt
                  on the ranking score, to check whether they stack
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators.mathstats import _trend_stats

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
CAP, COST, TOP_N = 0.15, 5.0, 8
DD_TILT_ZONE, DD_TILT_FACTOR = (-0.15, -0.05), 2.0
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2020 COVID", "2020-02-01", "2020-12-31"),
           ("2022 bear", "2022-01-01", "2022-12-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END), ("Last 5yr", "2021-11-23", END)]


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
    ds = pd.Series(idx)
    rd_pos = sorted({g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si})
    rd_set = set(rd_pos)
    rd_dates = [idx[p] for p in rd_pos]

    slope_at = {c: {} for c in present}
    for c in present:
        s = pd.Series(pv[c], index=idx).dropna()
        pos = {d_: i for i, d_ in enumerate(s.index)}
        for d_ in rd_dates:
            if d_ not in pos or pos[d_] < 20:
                continue
            j = pos[d_]
            sl, _r2 = _trend_stats(s.iloc[max(0, j - 19):j + 1], 20, 10)
            slope_at[c][d_] = sl

    def sim(variant):
        w = {c: 0. for c in present}; prev = {c: 0. for c in present}
        ret = np.zeros(n); turn = np.zeros(n)
        n_eligible = []
        for j in range(si, n - 1):
            if j in rd_set:
                d_ = idx[j]
                cands = []
                for c in present:
                    if not (fin(pv[c][j]) and fin(mom[c][j])) or mom[c][j] <= 0:
                        continue
                    if variant in ("slope_filter", "slope_ddtilt"):
                        sl = slope_at[c].get(d_)
                        if sl is None or sl <= 0:
                            continue
                    score = mom[c][j]
                    if variant == "slope_ddtilt" and fin(dd[c][j]) and DD_TILT_ZONE[0] < dd[c][j] <= DD_TILT_ZONE[1]:
                        score *= DD_TILT_FACTOR
                    cands.append((c, score))
                n_eligible.append(len(cands))
                cands.sort(key=lambda x: x[1], reverse=True)
                sel = cands[:TOP_N]
                tot = sum(v for _, v in sel)
                tg = {c: min(v / tot, CAP) for c, v in sel} if tot > 0 else {}
                s2 = sum(tg.values())
                tg = {c: v / s2 for c, v in tg.items()} if s2 > 0 else {}
                w = {c: tg.get(c, 0.) for c in present}
            turn[j + 1] += sum(abs(w[c] - prev[c]) for c in present)
            prev = dict(w)
            s_ = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s_ += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            cash = 1.0 - sum(w.values())
            ret[j + 1] = s_ + cash * bil[j + 1]
        sc = slice(si + 1, n)
        avg_elig = np.mean(n_eligible) if n_eligible else float("nan")
        pct_lt8 = np.mean([e < TOP_N for e in n_eligible]) if n_eligible else float("nan")
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, avg_elig, pct_lt8

    configs = ["baseline", "slope_filter", "slope_ddtilt"]
    results = {k: sim(k) for k in configs}
    print("baseline = m12>0, top-8 by momentum (today's live sleeve rule)")
    print("slope_filter = baseline + 1-month (20d) slope > 0 eligibility (no backfill if <8 qualify)")
    print("slope_ddtilt = slope_filter's eligibility + the validated 2x dd-zone tilt on the ranking score\n")
    for k in configs:
        _, _, avg_elig, pct_lt8 = results[k]
        print(f"  {k:14s} avg eligible/month: {avg_elig:.1f}   months with <{TOP_N} eligible: {pct_lt8:.0%}")
    print()
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for k in configs:
            r, dts, _, _ = results[k]; sh, cg, mdd = stats(r, dts, lo, hi)
            print(f"  {k:14s} {sh:>7.2f} {cg:>7.1%} {mdd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
