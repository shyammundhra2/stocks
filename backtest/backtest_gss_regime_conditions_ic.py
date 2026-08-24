"""
Does the EXISTING regime scalar's logic actually predict forward SPY/QQQ
returns? The live regime_scalar (macro/indicators/regime.py _regime_details +
mathstats.get_regime_scalar) counts 6 pass/fail conditions and scales
deployment 0-100% by how many pass, assuming MORE PASSES = MORE BULLISH.

This tests the two conditions that overlap with the breadth/credit IC test
(which used raw ratio LEVEL - different from what's live): here, the actual
production framing - ratio ABOVE its own 50d moving average (momentum-of-
ratio, not level):
  breadth_pass = RSP/SPY  > its 50d MA
  credit_pass  = HYG/IEF  > its 50d MA
  trend_pass   = SPY > its 200d MA           (for comparison - the other conditions)
  vix_pass     = VIX<20 and MOVE<110

For each, forward SPY/QQQ return at 5/10/21d, PASS vs FAIL (matching the
live boolean, not quintiles) - directly checks whether "pass" truly precedes
better forward returns, i.e. whether the direction assumed by
get_regime_scalar (more passes -> scale up) is empirically supported.
2007-2026.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
HORIZONS = [5, 10, 21]


def stats_for_bool(cond, fwd, si):
    m = np.isfinite(fwd) & np.isfinite(cond.astype(float))
    m[:si] = False
    c = cond[m].astype(bool); f = fwd[m]
    p = f[c]; q = f[~c]
    return (p > 0).mean() * 100, (q > 0).mean() * 100, p.mean() * 100, q.mean() * 100, len(p), len(q)


def main():
    t0 = time.time()
    tickers = ["SPY", "QQQ", "RSP", "HYG", "IEF", "^VIX", "^MOVE"]
    print(f"Downloading {tickers} ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"]
    idx = close.index; n = len(idx)
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 260)

    spy = close["SPY"].values; qqq = close["QQQ"].values
    rsp_spy = (close["RSP"] / close["SPY"])
    hyg_ief = (close["HYG"] / close["IEF"])
    breadth_pass = (rsp_spy > rsp_spy.rolling(50).mean()).values
    credit_pass = (hyg_ief > hyg_ief.rolling(50).mean()).values
    trend_pass = (close["SPY"] > close["SPY"].rolling(200).mean()).values
    vix_pass = ((close["^VIX"] < 20) & (close["^MOVE"] < 110)).values
    both_pass = breadth_pass & credit_pass                    # breadth AND credit agree

    conditions = {
        "breadth_pass (RSP/SPY>50MA)": breadth_pass,
        "credit_pass (HYG/IEF>50MA)": credit_pass,
        "both_pass (breadth&credit)": both_pass,
        "trend_pass (SPY>200MA)": trend_pass,
        "vix_pass (VIX<20&MOVE<110)": vix_pass,
    }
    targets = {"SPY": spy, "QQQ": qqq}

    print(f"{'target':>5s} {'horizon':>7s} {'condition':>28s} {'PASS hit%':>9s} {'FAIL hit%':>9s} "
          f"{'PASS fwd%':>9s} {'FAIL fwd%':>9s} {'nP':>6s} {'nF':>6s}")
    print("-" * 100)
    for tname, tpx in targets.items():
        for h in HORIZONS:
            fwd = np.full(n, np.nan)
            fwd[si:n - h] = tpx[si + h:n] / tpx[si:n - h] - 1.0
            for cname, cond in conditions.items():
                ph, fh, pf, ff, nP, nF = stats_for_bool(cond, fwd, si)
                print(f"{tname:>5s} {h:>6d}d {cname:>28s} {ph:>8.1f}% {fh:>8.1f}% "
                      f"{pf:>8.2f}% {ff:>8.2f}% {nP:>6d} {nF:>6d}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
