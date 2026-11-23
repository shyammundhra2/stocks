"""
Does trend-following work better coming out of a severe correction, while
mean-reversion works better once price is back near its old highs?

Tests the hypothesis directly rather than answering from priors: for every
ticker-day in the TREND_ASSETS universe, bucket by CURRENT drawdown from the
trailing 252d high (severe / moderate recovery / near highs), then compute
the pooled rank-IC of a trend signal (12-1mo momentum) and a mean-reversion
signal (5-day reversal) against the FORWARD 20-trading-day return, per
bucket. If the hypothesis holds, trend's IC should be highest in the
"severe" bucket and MR's IC should be highest in "near highs".

This is a market-level/per-asset regime split (distance from a prior high),
a different axis from core_route()'s existing high-ER=trend/low-ER=MR split
(trendiness of the current move, not where it sits relative to history) -
deliberately kept separate rather than assumed to be the same thing.

Full available history per ticker (yfinance auto_adjust), no survivorship
adjustment beyond what yfinance itself provides, net of nothing (this is a
pure signal-IC test, not a strategy backtest - no costs/sizing to apply).
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

FWD = 20          # forward return horizon (trading days)
BUCKETS = [("Severe (dd <= -15%)", -1.0, -0.15),
           ("Moderate recovery (-15% < dd <= -5%)", -0.15, -0.05),
           ("Near highs (dd > -5%)", -0.05, 1.0)]


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start="2005-01-01", progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 300]
    print(f"Universe with enough history: {len(present)}\n")

    rows = []   # (bucket_label, trend_signal, mr_signal, fwd_ret)
    for t in present:
        s = close[t].dropna()
        p = s.values
        n = len(p)
        if n < 280:
            continue
        hi = pd.Series(p).rolling(252, min_periods=60).max().values
        dd = p / hi - 1.0
        # Trend: 12-1 month momentum (skip the most recent month, standard).
        mom = np.full(n, np.nan)
        for i in range(252, n):
            mom[i] = p[i - 21] / p[i - 252] - 1.0
        # Mean-reversion: 5-day reversal, sign-flipped so higher = "more oversold = more bullish".
        rev5 = np.full(n, np.nan)
        rev5[5:] = -(p[5:] / p[:-5] - 1.0)
        fwd = np.full(n, np.nan)
        fwd[:-FWD] = p[FWD:] / p[:-FWD] - 1.0

        for i in range(252, n - FWD):
            if not (np.isfinite(dd[i]) and np.isfinite(mom[i]) and np.isfinite(rev5[i]) and np.isfinite(fwd[i])):
                continue
            for lab, lo, hi_b in BUCKETS:
                if lo < dd[i] <= hi_b:
                    rows.append((lab, mom[i], rev5[i], fwd[i]))
                    break

    df = pd.DataFrame(rows, columns=["bucket", "trend_sig", "mr_sig", "fwd_ret"])
    print(f"Pooled ticker-days: {len(df)}\n")
    print(f"{'Bucket':40s} {'n':>8s} {'IC(trend)':>10s} {'IC(MR)':>10s} {'winner':>10s}")
    for lab, _, _ in BUCKETS:
        sub = df[df["bucket"] == lab]
        if len(sub) < 50:
            print(f"{lab:40s} {'(insufficient n)':>8s}")
            continue
        ic_trend, _ = spearmanr(sub["trend_sig"], sub["fwd_ret"])
        ic_mr, _ = spearmanr(sub["mr_sig"], sub["fwd_ret"])
        winner = "trend" if ic_trend > ic_mr else "MR"
        print(f"{lab:40s} {len(sub):>8d} {ic_trend:>10.3f} {ic_mr:>10.3f} {winner:>10s}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
