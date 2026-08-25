"""
Robustness / anti-data-mining check on the horse-race winner (hi52, the
52-week-high proximity rank). Is hi52's edge over slope_r2 stable, or a
top-K artifact? And is a COMBINED rank (slope_r2 = path smoothness, hi52 =
level anchoring - two different trend proxies) more robust than either alone?

For TOPK in {5,8,12} report Sharpe in BOTH sub-windows for:
  slope_r2   current engine metric (baseline)
  hi52       price / 252d high
  combo      average of cross-sectional PERCENTILE ranks of slope_r2 & hi52

A real edge beats slope_r2 in BOTH windows at ALL K. Anything that only wins
at one K or one window is noise (McLean-Pontiff: assume decay, demand margin).
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
COST, SL_WIN = 5.0, 63


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
        return (np.nan, np.nan, np.nan)
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(UNIV + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values

    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]
    pv = {}; m200 = {}; SLR2 = {}; HI52 = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m200[c] = px.rolling(200).mean().values
        s, r = roll_slope_r2(px.values, SL_WIN); SLR2[c] = s * r
        HI52[c] = pv[c] / px.rolling(252).max().values

    # cross-sectional percentile ranks per day for the combo signal
    slr2_df = pd.DataFrame({c: SLR2[c] for c in present})
    hi52_df = pd.DataFrame({c: HI52[c] for c in present})
    combo_df = (slr2_df.rank(axis=1, pct=True) + hi52_df.rank(axis=1, pct=True)) / 2.0
    COMBO = {c: combo_df[c].values for c in present}

    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    SIG = {"slope_r2": SLR2, "hi52": HI52, "combo": COMBO}

    def sim(sig, topk):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                elig = [(sig[c][j], c) for c in present
                        if fin(pv[c][j]) and fin(m200[c][j]) and pv[c][j] > m200[c][j] and fin(sig[c][j])]
                elig.sort(reverse=True)
                sel = [c for _, c in elig[:topk]]
                neww = {c: (1.0 / len(sel) if c in sel else 0.0) for c in present} if sel else {c: 0. for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    print(f"\nUniverse {len(present)}, >200DMA filter, ToM rebal, net@{COST:.0f}bps")
    print("Sharpe (2011-19 / 2020-26)  |  full CAGR  |  full maxDD\n")
    for topk in (5, 8, 12):
        print(f"== TOP-{topk} ==")
        for name in SIG:
            r, d = sim(SIG[name], topk)
            s1, _, _ = perf(r, d, "2011-01-01", "2019-12-31")
            s2, _, _ = perf(r, d, "2020-01-01", END)
            _, cg, dd = perf(r, d, "2011-01-01", END)
            star = " *base" if name == "slope_r2" else "      "
            print(f"  {name:9s}{star}  {s1:>5.2f} / {s2:>5.2f}   {cg:>6.1%}   {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
