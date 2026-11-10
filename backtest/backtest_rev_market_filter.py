"""
Can the REV leg be improved by conditioning on a MARKET-level signal?

Motivated by backtest_rev_signal_quality: z20 carries TIMING information, not
selection information. If REV's value is timing, a better timing signal should
improve it - so test the obvious candidates.

Candidates (all market-level, all free):
    vix_z        VIX z-score vs 50d, the gauge the capitulation tile uses
    vix_ts       VIX / VIX3M term structure, the validated deploy throttle
    spy_rsi14    price washout
    spy_dd       SPY drawdown from its 252d high
    hmm_pcrisis  regime HMM crisis probability (loaded if the walk-forward
                 CSV exists; skipped with a note otherwise)

TWO THINGS THIS CONTROLS FOR, both learned the hard way:

 1. Overlap. 20d forward returns sampled daily overlap ~20x. Everything is
    reported at the DATE level (one row per signal date) and the headline
    conclusion uses NON-OVERLAPPING dates, every 20 sessions.

 2. "Improves REV" vs "is just market timing". A filter that helps REV might
    help everything equally - that is not an improvement to REV, it is a market
    call you could apply anywhere. So every bucket reports the REV return AND
    the eligible-pool return on the same dates. What matters is the DIFFERENCE
    moving, not REV's absolute number moving.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
import backtest.backtest_rev_signal_quality as Q

HORIZON = 20
HMM_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmm_backtest_output.csv")


def market_signals(index):
    spy = yf.download("SPY", period="15y", auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", period="15y", auto_adjust=True, progress=False)["Close"].squeeze()
    spy.index = spy.index.tz_localize(None)
    vix.index = vix.index.tz_localize(None)

    d = pd.DataFrame(index=spy.index)
    d["vix_z"] = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    r = spy.diff()
    g = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["spy_rsi14"] = 100 - 100 / (1 + g / l)
    d["spy_dd"] = spy / spy.rolling(252, min_periods=60).max() - 1.0

    try:  # VIX3M only has a short yfinance history; use what there is
        v3 = yf.download("^VIX3M", period="15y", auto_adjust=True, progress=False)["Close"].squeeze()
        v3.index = v3.index.tz_localize(None)
        d["vix_ts"] = (vix / v3).reindex(d.index)
    except Exception:
        d["vix_ts"] = np.nan

    if os.path.exists(HMM_CSV):
        h = pd.read_csv(HMM_CSV, index_col=0, parse_dates=True)
        col = next((c for c in h.columns if "crisis" in c.lower()), None)
        if col:
            h.index = h.index.tz_localize(None) if h.index.tz is not None else h.index
            d["hmm_pcrisis"] = h[col].reindex(d.index).ffill()
    return d.reindex(index).ffill()


def bucket_report(dates_df, col, cuts, label):
    """dates_df: one row per signal date with rev_fwd, pool_fwd and the signal."""
    sub = dates_df.dropna(subset=[col])
    if len(sub) < 60:
        print(f"\n--- {label}: only {len(sub)} dates, skipped ---")
        return
    print(f"\n--- {label} (n={len(sub)} signal dates) ---")
    print(f"{'bucket':>18} {'n':>5} {'REV fwd':>9} {'pool fwd':>9} {'REV-pool':>9}")
    for lo, hi in cuts:
        b = sub[(sub[col] > lo) & (sub[col] <= hi)]
        if len(b) < 20:
            continue
        print(f"  {lo:>7.2f}..{hi:<7.2f} {len(b):>5} {b.rev_fwd.mean():>+9.2%} "
              f"{b.pool_fwd.mean():>+9.2%} {b.rev_fwd.mean()-b.pool_fwd.mean():>+9.2%}")


def main():
    t0 = time.time()
    d = Q.build_panel()
    pool = d[(d.er <= 0.35) & d.above200].copy()
    sig = pool[pool.z20 < -1.0]

    # collapse to one row per date: what REV earned, and what the pool earned
    dates = pd.DataFrame({
        "rev_fwd": sig.groupby("date").fwd.mean(),
        "pool_fwd": pool.groupby("date").fwd.mean(),
        "n_rev": sig.groupby("date").size(),
    }).dropna(subset=["rev_fwd"])

    ms = market_signals(dates.index)
    dates = dates.join(ms)
    print(f"\nREV fired on {len(dates)} distinct dates, {dates.index.min().date()} .. "
          f"{dates.index.max().date()}")
    print(f"baseline:  REV {dates.rev_fwd.mean():+.2%}   pool {dates.pool_fwd.mean():+.2%}   "
          f"REV-pool {dates.rev_fwd.mean()-dates.pool_fwd.mean():+.2%}")
    print("\n(REV-pool is the column that matters. If a filter lifts REV and the")
    print(" pool equally, it is a market call, not an improvement to REV.)")

    bucket_report(dates, "vix_z", [(-9, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)], "VIX z-score")
    bucket_report(dates, "spy_rsi14", [(0, 35), (35, 45), (45, 55), (55, 100)], "SPY RSI14")
    bucket_report(dates, "spy_dd", [(-1, -0.10), (-0.10, -0.05), (-0.05, -0.02), (-0.02, 0)], "SPY drawdown")
    bucket_report(dates, "vix_ts", [(0, 0.90), (0.90, 1.0), (1.0, 1.1), (1.1, 9)], "VIX/VIX3M term structure")
    if "hmm_pcrisis" in dates:
        bucket_report(dates, "hmm_pcrisis", [(-0.01, 0.05), (0.05, 0.3), (0.3, 0.7), (0.7, 1.01)],
                      "HMM p_crisis")
    else:
        print("\n--- HMM p_crisis: no walk-forward CSV found, skipped ---")

    # non-overlapping confirmation on whichever filter looked best
    print("\n=== NON-OVERLAPPING check (every 20th signal date) ===")
    nd = dates.iloc[::HORIZON]
    print(f"  n={len(nd)} independent dates   REV {nd.rev_fwd.mean():+.2%}   "
          f"pool {nd.pool_fwd.mean():+.2%}   REV-pool {nd.rev_fwd.mean()-nd.pool_fwd.mean():+.2%}")
    for col, thr, lbl in [("vix_z", 1.0, "vix_z > 1"), ("spy_dd", -0.05, "SPY >5% off high"),
                          ("spy_rsi14", 40, "RSI14 < 40")]:
        if col not in nd:
            continue
        m = nd[col] > thr if col == "vix_z" else nd[col] < thr
        sub = nd[m.fillna(False)]
        if len(sub) >= 10:
            print(f"  {lbl:22s} n={len(sub):>3}  REV {sub.rev_fwd.mean():>+7.2%}  "
                  f"pool {sub.pool_fwd.mean():>+7.2%}  diff {sub.rev_fwd.mean()-sub.pool_fwd.mean():>+7.2%}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
