"""
Does the production slope*r2 sort order have alpha - and does the
self-normalized persistence filter improve it?

get_trends() ranks assets by slope*r2 (20-day OLS on log price, the same
_trend_stats(c, 20, 10) used live). This tests that ranking as a tradable
sort, and cleanly separates alpha from beta by using a dollar-neutral
top-third-minus-bottom-third long/short (equal names each side, so market
beta nets out - the fix the previous strategy test was missing).

Per rebalance date t (non-overlapping FWD-day holds; no lookahead):
  sr    = slope*r2 from _trend_stats(trailing prices, 20, 10)
  znorm = VR(q=21) self-normalized vs the asset's own trailing VR history

Strategies:
  long_all          +1 everything                       (beta benchmark)
  sr_ls_tertile     long top-3rd / short bottom-3rd by sr   (beta-neutral)
  sr_long_top       long-only top-3rd by sr                 (has beta)
  mom_ls_tertile    same L/S but ranked by 63d return    (sort-key control)
  sr_ls_trendfilt   L/S by sr, universe = self-norm TREND names only
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import _variance_ratio, _trend_stats

END = pd.Timestamp("2026-08-20")
TEST_START = END - pd.DateOffset(years=2)
LOOKBACK = 252
FWD = 10
NORM_WIN = 126
Z_THR = 0.5
MOM_LB = 63
TRW, TRSCALE = 20, 10       # production _trend_stats window / scale


def ann_stats(r, ppy):
    r = np.asarray(r, float)
    if len(r) < 3:
        return (float("nan"),) * 4
    mu, sd = r.mean(), r.std(ddof=1)
    ar, av = mu * ppy, sd * np.sqrt(ppy)
    return ar, av, (ar / av if av > 0 else float("nan")), float(np.mean(r > 0))


def ls_tertile(sr, fret, mask=None):
    """Dollar-neutral long top-third / short bottom-third by sr. Returns
    (period_return, n_per_side) or (nan, 0) if the cross-section is too thin."""
    idx = np.where(mask)[0] if mask is not None else np.arange(len(sr))
    idx = [j for j in idx if np.isfinite(sr[j]) and np.isfinite(fret[j])]
    if len(idx) < 6:
        return np.nan, 0
    order = sorted(idx, key=lambda j: sr[j])
    k = len(order) // 3
    if k < 1:
        return np.nan, 0
    short, long_ = order[:k], order[-k:]
    return float(np.mean([fret[j] for j in long_]) - np.mean([fret[j] for j in short])), k


def main(fwd=FWD, step=None):
    step = step or fwd
    t0 = time.time()
    print(f"Holding {fwd}d | rebalance {step}d | sort=slope*r2 (win {TRW})\n")
    tickers = list(TREND_ASSETS.keys())
    ds = (TEST_START - pd.DateOffset(days=620)).date()
    print(f"Downloading {len(tickers)} tickers {ds} -> {END.date()} ...")
    raw = yf.download(tickers, start=str(ds), end=str((END + pd.Timedelta(days=1)).date()),
                      progress=False, group_by="column", auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all")
    present = [t for t in tickers if t in close.columns]
    skipped = [t for t in tickers if t not in present]
    if skipped:
        print(f"Skipped (no data): {', '.join(skipped)}")
    idx = close.index
    n = len(idx)
    print(f"Universe: {len(present)}, {n} bars\n")

    print("Precomputing rolling VR (for the persistence filter) ...")
    vr = pd.DataFrame(index=idx, columns=present, dtype=float)
    for sym in present:
        vals = close[sym].values
        col = np.full(n, np.nan)
        for d in range(LOOKBACK, n):
            w = vals[d - LOOKBACK:d + 1]
            if not np.isnan(w).any():
                col[d] = _variance_ratio(w, q=21)
        vr[sym] = col

    logc = np.log(close)
    start_i = max(LOOKBACK + NORM_WIN, MOM_LB, TRW)
    positions = [i for i in range(start_i, n - fwd) if idx[i] >= TEST_START][::step]

    strategies = ["long_all", "sr_ls_tertile", "sr_long_top", "mom_ls_tertile", "sr_ls_trendfilt"]
    rets = {s: [] for s in strategies}
    nside = {s: [] for s in strategies}

    for i in positions:
        syms, srs, moms, frets, trend = [], [], [], [], []
        for sym in present:
            lp = logc[sym]
            p0, pf, pm = lp.iloc[i], lp.iloc[i + fwd], lp.iloc[i - MOM_LB]
            if not np.isfinite([p0, pf, pm]).all():
                continue
            sl, r2 = _trend_stats(close[sym].iloc[:i + 1], TRW, TRSCALE)
            vn = vr[sym].iloc[i]
            hist = vr[sym].iloc[i - NORM_WIN:i].dropna()
            zt = ((vn - hist.mean()) / hist.std()) if (len(hist) >= 20 and np.isfinite(vn) and hist.std() > 0) else np.nan
            syms.append(sym)
            srs.append(sl * r2)
            moms.append(p0 - pm)
            frets.append(np.expm1(pf - p0))
            trend.append(np.isfinite(zt) and zt > Z_THR)
        if len(syms) < 6:
            continue
        srs = np.array(srs); moms = np.array(moms); frets = np.array(frets); trend = np.array(trend)

        rets["long_all"].append(float(np.mean(frets))); nside["long_all"].append(len(frets))

        r, k = ls_tertile(srs, frets)
        if k:
            rets["sr_ls_tertile"].append(r); nside["sr_ls_tertile"].append(k)

        order = np.argsort(srs); ktop = len(order) // 3
        if ktop >= 1:
            top = order[-ktop:]
            rets["sr_long_top"].append(float(np.mean(frets[top]))); nside["sr_long_top"].append(ktop)

        r, k = ls_tertile(moms, frets)
        if k:
            rets["mom_ls_tertile"].append(r); nside["mom_ls_tertile"].append(k)

        r, k = ls_tertile(srs, frets, mask=trend)
        if k:
            rets["sr_ls_trendfilt"].append(r); nside["sr_ls_trendfilt"].append(k)

    ppy = 252 / fwd
    npd = len(rets["long_all"])
    se = np.sqrt(ppy / npd) if npd else float("nan")
    print(f"Periods: {npd} ({idx[positions[0]].date()} -> {idx[positions[-1]].date()}), {ppy:.1f}/yr")
    print(f"Sharpe standard error ~ +/-{se:.2f}\n")

    print(f"{'strategy':16s} {'ann_ret':>8s} {'ann_vol':>8s} {'Sharpe':>7s} {'hit':>5s} {'per-side':>8s}")
    print("-" * 58)
    sh = {}
    for s in strategies:
        ar, av, s_, hit = ann_stats(rets[s], ppy)
        sh[s] = s_
        nm = np.mean(nside[s]) if nside[s] else float("nan")
        print(f"{s:16s} {ar:>8.1%} {av:>8.1%} {s_:>7.2f} {hit:>5.0%} {nm:>8.1f}")

    print()
    print(f"slope*r2 sort (beta-neutral) Sharpe: {sh['sr_ls_tertile']:+.2f}  "
          f"(vs 63d-momentum sort {sh['mom_ls_tertile']:+.2f})")
    print(f"persistence-filtered slope*r2 sort:  {sh['sr_ls_trendfilt']:+.2f}  "
          f"({'better' if sh['sr_ls_trendfilt'] > sh['sr_ls_tertile'] else 'not better'} than unfiltered)")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=FWD)
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    main(fwd=a.fwd, step=a.step)
