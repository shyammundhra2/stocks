"""
Does mean reversion pay inside any slope*r2 bracket?

Reversion failed on the whole universe (backtest_persistence_strategy.py),
but it might work in a specific slice - e.g. flat/choppy names (slope*r2 ~ 0)
where there is no trend to fight, while it loses in strong trends.

Each rebalance date, rank names into slope*r2 quintiles (Q1 = most negative
slope*r2 / strongest downtrend, Q5 = strongest uptrend) and, inside each
bracket, apply a weekly mean-reversion rule:

    rev = -sign(trailing 5-day return)      long recent losers, short winners

held FWD days forward (non-overlapping, no lookahead). Reported per bracket:
  rev_sharpe   Sharpe of the reversion rule inside the bracket
  rev_ret      annualized return of that rule
  hit          fraction of periods positive
  fwd_ret      raw long-only forward return of the bracket (its beta/drift)

If reversion works anywhere, rev_sharpe is positive in that bracket.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import _trend_stats

END = pd.Timestamp("2026-08-20")
TEST_START = END - pd.DateOffset(years=2)
FWD = 10
REV_LB = 5
MOM_LB = 63              # momentum lookback for the within-bracket trend test
NQ = 5                    # number of slope*r2 brackets
TRW, TRSCALE = 20, 10


def ann_stats(r, ppy):
    r = np.asarray(r, float)
    if len(r) < 3:
        return (float("nan"),) * 4
    mu, sd = r.mean(), r.std(ddof=1)
    ar, av = mu * ppy, sd * np.sqrt(ppy)
    return ar, av, (ar / av if av > 0 else float("nan")), float(np.mean(r > 0))


def main(fwd=FWD, step=None):
    step = step or fwd
    t0 = time.time()
    print(f"Holding {fwd}d | rebalance {step}d | {NQ} slope*r2 brackets | reversion lb {REV_LB}d\n")
    tickers = list(TREND_ASSETS.keys())
    ds = (TEST_START - pd.DateOffset(days=420)).date()
    print(f"Downloading {len(tickers)} tickers {ds} -> {END.date()} ...")
    raw = yf.download(tickers, start=str(ds), end=str((END + pd.Timedelta(days=1)).date()),
                      progress=False, group_by="column", auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all")
    present = [t for t in tickers if t in close.columns]
    idx = close.index
    n = len(idx)
    print(f"Universe: {len(present)}, {n} bars\n")

    logc = np.log(close)
    start_i = max(252, TRW, REV_LB, MOM_LB)
    positions = [i for i in range(start_i, n - fwd) if idx[i] >= TEST_START][::step]

    rev_ret = {q: [] for q in range(NQ)}     # reversion rule inside bracket q
    mom_ret = {q: [] for q in range(NQ)}     # within-bracket beta-neutral momentum L/S
    fwd_ret = {q: [] for q in range(NQ)}     # raw long-only bracket return
    sr_avg = {q: [] for q in range(NQ)}
    breadth = {q: [] for q in range(NQ)}

    for i in positions:
        srs, revs, moms, frets = [], [], [], []
        for sym in present:
            lp = logc[sym]
            p0, pf, pr, pm = lp.iloc[i], lp.iloc[i + fwd], lp.iloc[i - REV_LB], lp.iloc[i - MOM_LB]
            if not np.isfinite([p0, pf, pr, pm]).all():
                continue
            sl, r2 = _trend_stats(close[sym].iloc[:i + 1], TRW, TRSCALE)
            srs.append(sl * r2)
            revs.append(-np.sign(p0 - pr))
            moms.append(p0 - pm)                       # 63d log return = momentum score
            frets.append(np.expm1(pf - p0))
        if len(srs) < NQ * 2:
            continue
        srs = np.array(srs); revs = np.array(revs); moms = np.array(moms); frets = np.array(frets)
        order = np.argsort(srs)                       # ascending: Q0 = lowest slope*r2
        buckets = np.array_split(order, NQ)
        for q, b in enumerate(buckets):
            if len(b) == 0:
                continue
            rev_ret[q].append(float(np.mean(revs[b] * frets[b])))
            fwd_ret[q].append(float(np.mean(frets[b])))
            sr_avg[q].append(float(np.mean(srs[b])))
            breadth[q].append(len(b))
            # within-bracket momentum: long higher-momentum half, short lower half
            # (dollar-neutral inside the bracket -> strips the bracket's beta)
            if len(b) >= 4:
                o = np.argsort(moms[b]); h = len(o) // 2
                lo, hi = b[o[:h]], b[o[-h:]]
                mom_ret[q].append(float(np.mean(frets[hi]) - np.mean(frets[lo])))

    ppy = 252 / fwd
    npd = len(rev_ret[0])
    se = np.sqrt(ppy / npd) if npd else float("nan")
    print(f"Periods: {npd} ({idx[positions[0]].date()} -> {idx[positions[-1]].date()}), {ppy:.1f}/yr")
    print(f"Sharpe standard error ~ +/-{se:.2f}\n")

    print(f"{'bracket':10s} {'avg sr':>8s} {'rev_ret':>8s} {'rev_shrp':>8s} {'hit':>5s} "
          f"{'fwd_ret':>8s} {'names':>6s}")
    print("-" * 62)
    labels = {0: "Q1 (dn)", NQ - 1: f"Q{NQ} (up)"}
    for q in range(NQ):
        ar, av, sh, hit = ann_stats(rev_ret[q], ppy)
        far, _, _, _ = ann_stats(fwd_ret[q], ppy)
        lab = labels.get(q, f"Q{q + 1}")
        print(f"{lab:10s} {np.mean(sr_avg[q]):>8.2f} {ar:>8.1%} {sh:>8.2f} {hit:>5.0%} "
              f"{far:>8.1%} {np.mean(breadth[q]):>6.1f}")

    best = max(range(NQ), key=lambda q: ann_stats(rev_ret[q], ppy)[2])
    bsh = ann_stats(rev_ret[best], ppy)[2]
    lab = labels.get(best, f"Q{best + 1}")
    verdict = "reversion pays here" if bsh > se else "no bracket beats noise"
    print(f"\nBest reversion bracket: {lab} (Sharpe {bsh:+.2f}) -> {verdict}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=FWD)
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    main(fwd=a.fwd, step=a.step)
