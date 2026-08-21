"""
Do BUY signals whose trailing slope*R2 path traces a "U" (dipped to a
mid-month trough, now re-accelerating) outperform BUYs without a U-path?

For every (asset, day) across TREND_ASSETS, 2007-26:
  BUY proxy   = price > 200SMA and > 50SMA and slope > 0 and R2 > 0.6
                (the production "buy/hold zone" core condition)
  U path      = over the last ~21 days of slope*R2: a trough in the middle
                (below the start), now recovered above it, currently positive
                and rising into today.
Then compare the forward return of U-BUYs vs non-U-BUYs at a few horizons.

Rolling 20-bar slope/R2 is vectorized (closed-form OLS) to match
_trend_stats(c, 20, 10). Adjusted (total-return) prices.
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

START, END = "2007-01-01", "2026-08-21"
WIN = 20            # trend-stats window (matches production)
PATH = 21          # ~1 month trailing path
FWDS = [10, 21]    # forward horizons (trading days)
# U = down->up reversal in the (slope, R2) plane: start at negative slope +
# high R2, dip through ~(0,0), end at positive slope + high R2.
EPS_S = 1.0        # |slope| (x1000 scale) to count as a real down/up leg
R2_HI = 0.5        # "clean trend" R2 at both ends of the U
R2_LO = 0.30       # R2 collapses in the middle as the trend breaks/turns


def roll_slope_r2(prices):
    """Vectorized slope (scaled x1000) and R2 of a 20-bar OLS on log price,
    ending at each index. Matches _trend_stats(c, 20, 10). NaN for the first
    WIN-1 bars."""
    n = len(prices)
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < WIN or sliding_window_view is None:
        return slope, r2
    W = sliding_window_view(lp, WIN)               # (n-WIN+1, WIN)
    x = np.arange(WIN)
    xc = x - x.mean()
    denom = float(xc @ xc)                         # 665 for WIN=20
    ymean = W.mean(axis=1)
    sl_c = (W - ymean[:, None]) @ xc / denom       # per-window OLS slope
    intercept = ymean - sl_c * x.mean()
    pred = sl_c[:, None] * x[None, :] + intercept[:, None]
    sstot = ((W - ymean[:, None]) ** 2).sum(axis=1)
    ssres = ((W - pred) ** 2).sum(axis=1)
    r2w = np.where(sstot > 0, 1.0 - ssres / sstot, 0.0)
    slope[WIN - 1:] = sl_c * 10 * 100              # production scale
    r2[WIN - 1:] = np.clip(r2w, 0.0, 1.0)
    return slope, r2


def is_U(sp, rp):
    """Down->up reversal in the (slope, R2) plane (the SLV-style U):
    sp, rp = trailing slope and R2 paths (len PATH), oldest..today.
    Start with negative slope, pass through ~(0,0) (slope ~0 AND R2 collapses),
    end with a strong positive slope and clean fit."""
    L = len(sp)
    if L < 15 or not (np.isfinite(sp).all() and np.isfinite(rp).all()):
        return False
    a, b = L // 5, (4 * L) // 5
    if not (sp[0] < 0 and sp[-1] > EPS_S):          # negative slope -> strong positive
        return False
    mid_break = np.any((np.abs(sp[a:b]) < EPS_S) & (rp[a:b] < R2_LO))  # through ~(0,0)
    return bool(mid_break) and rp[-1] > R2_HI


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {START}..{END} ...")
    raw = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    print(f"Universe with >260 bars: {len(present)}\n")

    # collect (is_U, fwd_ret) per BUY observation, per horizon
    recs = {f: {"U": [], "N": []} for f in FWDS}
    for sym in present:
        c = close[sym].dropna()
        p = c.values
        n = len(p)
        if n < 260:
            continue
        slope, r2 = roll_slope_r2(p)
        sr = slope * r2
        ser = pd.Series(p)
        ma50 = ser.rolling(50).mean().values
        ma200 = ser.rolling(200).mean().values

        for j in range(220, n - max(FWDS)):
            if not (p[j] > ma200[j] and p[j] > ma50[j] and slope[j] > 0 and r2[j] > 0.6):
                continue
            u = is_U(slope[j - PATH + 1:j + 1], r2[j - PATH + 1:j + 1])
            for f in FWDS:
                fret = p[j + f] / p[j] - 1.0
                recs[f]["U" if u else "N"].append(fret)

    print(f"{'horizon':>8s} {'group':>6s} {'n':>7s} {'mean%':>8s} {'median%':>8s} "
          f"{'hit%':>6s} {'per-trade Sharpe':>17s}")
    print("-" * 62)
    for f in FWDS:
        for g, lab in (("U", "U-BUY"), ("N", "non-U")):
            r = np.array(recs[f][g], dtype=float)
            if len(r) < 5:
                print(f"{f:>8d} {lab:>6s} {len(r):>7d}  (too few)")
                continue
            sh = r.mean() / r.std() if r.std() > 0 else float("nan")
            print(f"{f:>8d} {lab:>6s} {len(r):>7d} {r.mean()*100:>8.2f} "
                  f"{np.median(r)*100:>8.2f} {np.mean(r>0)*100:>6.0f} {sh:>17.3f}")
        u = np.array(recs[f]["U"]); nn = np.array(recs[f]["N"])
        if len(u) >= 5 and len(nn) >= 5:
            diff = u.mean() - nn.mean()
            # rough two-sample t (obs overlap -> significance is optimistic)
            se = np.sqrt(u.var()/len(u) + nn.var()/len(nn))
            t = diff / se if se > 0 else float("nan")
            print(f"{'':>8s} {'diff':>6s} U-nonU mean = {diff*100:+.2f}%  (rough t={t:+.2f})")
        print()

    print("Caveat: overlapping daily observations are autocorrelated, so the")
    print("per-trade Sharpe and t-stat overstate significance. Read the U-vs-nonU")
    print("*difference* in mean/hit, not the absolute levels.")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
