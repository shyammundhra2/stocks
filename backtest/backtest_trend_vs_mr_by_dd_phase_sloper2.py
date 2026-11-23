"""
Same test as backtest_trend_vs_mr_by_dd_phase.py, but with the trend signal
swapped from 12-1 month momentum to slope x R^2 (the 21-day trend-quality
measure used in core_route()'s high-ER branch and the rotation map
rankings) - checking whether the "trend's IC peaks in the 5-15% drawdown
zone" finding holds for this signal too, before considering swapping it
into the live trendselection/predict_top_trends.py ranker. MR signal (5-day
reversal) and dd-phase buckets unchanged, so this isolates the effect of
the trend-signal definition itself.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

WIN = 21   # matches core_route()'s WIN and the rotation-map convention
BUCKETS = [("Severe (dd <= -15%)", -1.0, -0.15),
           ("Moderate recovery (-15% < dd <= -5%)", -0.15, -0.05),
           ("Near highs (dd > -5%)", -0.05, 1.0)]


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan); lp = np.log(p)
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start="2005-01-01", progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 300]
    print(f"Universe with enough history: {len(present)}\n")

    rows = []
    for t in present:
        s = close[t].dropna()
        p = s.values
        n = len(p)
        if n < 280:
            continue
        hi = pd.Series(p).rolling(252, min_periods=60).max().values
        dd = p / hi - 1.0
        slope, r2 = roll_sr(p, WIN)
        trend_sig = slope * r2   # signed trend-quality score, same as core_route()'s usage before its max(.,0) clip
        rev5 = np.full(n, np.nan)
        rev5[5:] = -(p[5:] / p[:-5] - 1.0)
        fwd = np.full(n, np.nan)
        fwd[:-20] = p[20:] / p[:-20] - 1.0

        for i in range(252, n - 20):
            if not (np.isfinite(dd[i]) and np.isfinite(trend_sig[i]) and np.isfinite(rev5[i]) and np.isfinite(fwd[i])):
                continue
            for lab, lo, hi_b in BUCKETS:
                if lo < dd[i] <= hi_b:
                    rows.append((lab, trend_sig[i], rev5[i], fwd[i]))
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
