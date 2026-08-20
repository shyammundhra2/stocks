"""
Within a slope*r2 bracket, does the persistence LABEL pick the right rule?

For Q1 (strongest-downtrend slope*r2 quintile) and Q4 (mild-uptrend quintile),
split names by what the production persistence classifier says
(_persistence_classify: TREND / NEUTRAL / MEAN-REVERT, from the Hurst + VR +
autocorr blend), and inside each cell run BOTH a reversion rule and a trend
(momentum) rule. If the label has value, the label-matching rule should win:

    "it says MEAN-REVERT" -> reversion should beat momentum
    "it says TREND"       -> momentum should beat reversion

Rules (position in {-1,+1}, forward FWD-day hold, non-overlapping, no lookahead):
    reversion  rev = -sign(trailing 5d return)     (fade recent move)
    trend      mom =  sign(trailing 63d return)    (ride medium momentum)

Returns are pooled per period (mean over the cell's names) then annualized.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import (
    _trend_stats, _hurst_exponent, _variance_ratio,
    _return_autocorr, _persistence_classify,
)

END = pd.Timestamp("2026-08-20")
TEST_START = END - pd.DateOffset(years=2)
LOOKBACK = 252
FWD = 5
REV_LB, MOM_LB = 5, 63
NQ = 5
TRW, TRSCALE = 20, 10
MIN_CELL = 2              # min names in a (bracket,label) cell to record the period


def ann_stats(r, ppy):
    r = np.asarray(r, float)
    if len(r) < 3:
        return float("nan"), float("nan")
    mu, sd = r.mean(), r.std(ddof=1)
    sh = (mu * ppy) / (sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    return mu * ppy, sh


def main(fwd=FWD, step=None):
    step = step or fwd
    t0 = time.time()
    print(f"Holding {fwd}d | rebalance {step}d | brackets Q1 & Q4 x persistence label\n")
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
    start_i = max(LOOKBACK, MOM_LB)
    positions = [i for i in range(start_i, n - fwd) if idx[i] >= TEST_START][::step]

    # cells: bracket in {Q1,Q4} x label in {TREND,MEAN-REVERT} x rule in {rev,mom}
    cells = {(brk, lab, rule): []
             for brk in ("Q1", "Q4")
             for lab in ("MEAN-REVERT", "TREND")
             for rule in ("rev", "mom")}
    counts = {(brk, lab): [] for brk in ("Q1", "Q4") for lab in ("MEAN-REVERT", "TREND")}

    print("Scoring (slope*r2 + persistence label per name per date) ...")
    for i in positions:
        rows = []
        for sym in present:
            lp = logc[sym]
            p0, pf, pr, pm = lp.iloc[i], lp.iloc[i + fwd], lp.iloc[i - REV_LB], lp.iloc[i - MOM_LB]
            if not np.isfinite([p0, pf, pr, pm]).all():
                continue
            s = close[sym].iloc[:i + 1]
            if s.iloc[-LOOKBACK:].isna().any() or len(s) < LOOKBACK:
                continue
            win = s.iloc[-LOOKBACK:]
            sl, r2 = _trend_stats(s, TRW, TRSCALE)
            h = _hurst_exponent(win)
            vr = _variance_ratio(win, q=21)
            ac = _return_autocorr(win)
            lab = _persistence_classify(h, vr, ac)["persistence_label"]
            rows.append({
                "sr": sl * r2,
                "lab": lab,
                "rev": -np.sign(p0 - pr),
                "mom": np.sign(p0 - pm),
                "fret": np.expm1(pf - p0),
            })
        if len(rows) < NQ * 2:
            continue
        rows.sort(key=lambda d: d["sr"])
        buckets = np.array_split(np.arange(len(rows)), NQ)
        for brk, bidx in (("Q1", buckets[0]), ("Q4", buckets[3])):
            grp = [rows[j] for j in bidx]
            for lab in ("MEAN-REVERT", "TREND"):
                sub = [d for d in grp if d["lab"] == lab]
                counts[(brk, lab)].append(len(sub))
                if len(sub) < MIN_CELL:
                    continue
                cells[(brk, lab, "rev")].append(float(np.mean([d["rev"] * d["fret"] for d in sub])))
                cells[(brk, lab, "mom")].append(float(np.mean([d["mom"] * d["fret"] for d in sub])))

    ppy = 252 / fwd
    npd = len(positions)
    se = np.sqrt(ppy / max(npd, 1))
    print(f"Periods: {npd} ({idx[positions[0]].date()} -> {idx[positions[-1]].date()}), "
          f"{ppy:.1f}/yr | Sharpe SE ~ +/-{se:.2f}\n")

    print(f"{'bracket':8s} {'label says':12s} {'rev Sharpe':>11s} {'mom Sharpe':>11s} "
          f"{'winner':>10s} {'avg N':>6s} {'obs':>5s}")
    print("-" * 68)
    for brk in ("Q1", "Q4"):
        for lab in ("MEAN-REVERT", "TREND"):
            rev = cells[(brk, lab, "rev")]
            mom = cells[(brk, lab, "mom")]
            _, rsh = ann_stats(rev, ppy)
            _, msh = ann_stats(mom, ppy)
            want = "rev" if lab == "MEAN-REVERT" else "mom"
            got = "rev" if (np.nan_to_num(rsh, nan=-9) > np.nan_to_num(msh, nan=-9)) else "mom"
            mark = "OK" if got == want else "no"
            avgn = np.mean(counts[(brk, lab)]) if counts[(brk, lab)] else 0.0
            print(f"{brk:8s} {lab:12s} {rsh:>11.2f} {msh:>11.2f} "
                  f"{got + ' (' + mark + ')':>10s} {avgn:>6.1f} {len(rev):>5d}")
    print("\nlabel-matching rule should win: MEAN-REVERT->rev, TREND->mom")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=int, default=FWD)
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    main(fwd=a.fwd, step=a.step)
