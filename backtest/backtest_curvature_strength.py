"""
Is a curvature-aware trend strength a better SIZING factor than the linear
slope*R2*strength - while keeping breadth (unlike reversal-only u_fit)?

Curvature strength fits log-price to a QUADRATIC over a window (price ~ t + t^2),
so it sees both the current trajectory and whether the trend is ACCELERATING
(convex up) or decelerating - the second-order info a linear OLS misses. It
scores every trend, so an established accelerating uptrend rates high (u_fit
would score it 0).

  curve_strength = max(end_slope, 0) * quad_R2 * (accel>0 ? 1.4 : 0.7)

Same portfolio framework as backtest_u_sizing: fully-invested long book among
BUY-eligible names, monthly rebalance, 2007-26. Schemes compared as the
optimizer's conviction weight:
  ew       equal weight
  linear   slope*R2*strength      (today's conviction; strength=tanh(sr/8))
  ufit     slope*R2*u_fit         (reversal-only - concentrates)
  curve    curvature strength     (quadratic, keeps breadth)
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
from backtest.backtest_u_pattern import roll_slope_r2, is_U, PATH
from backtest.backtest_u_sizing import u_fit, perf, wmean

START, END = "2007-01-01", "2026-08-21"
FWD = 21
QW = 30            # quadratic-fit window (~6 weeks, enough to see curvature)


def roll_quad(prices):
    """Per-index end-slope, acceleration, and R2 of a quadratic log-price fit
    over the trailing QW bars. NaN for the first QW-1 bars."""
    n = len(prices)
    es = np.full(n, np.nan); ac = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < QW or sliding_window_view is None:
        return es, ac, r2
    W = sliding_window_view(lp, QW)                    # (nw, QW)
    t = np.arange(QW)
    X = np.column_stack([np.ones(QW), t, t * t])       # (QW, 3)
    M = np.linalg.pinv(X)                              # (3, QW): coeffs = M @ y
    coeffs = W @ M.T                                   # (nw, 3): c0, c1, c2
    c1, c2 = coeffs[:, 1], coeffs[:, 2]
    end_slope = (c1 + 2 * c2 * (QW - 1)) * 1000.0      # trajectory now, ~slope scale
    fitted = coeffs @ X.T
    ymean = W.mean(axis=1)
    sstot = ((W - ymean[:, None]) ** 2).sum(axis=1)
    ssres = ((W - fitted) ** 2).sum(axis=1)
    r2q = np.where(sstot > 0, 1.0 - ssres / sstot, 0.0)
    es[QW - 1:] = end_slope
    ac[QW - 1:] = c2                                    # sign = accel / decel
    r2[QW - 1:] = np.clip(r2q, 0.0, 1.0)
    return es, ac, r2


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {START}..{END} ...")
    raw = yf.download(tickers, start=START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index
    n = len(idx)
    print(f"Universe with >260 bars: {len(present)}\n")

    print("Precomputing slope/R2/BUY/u_fit/curvature per asset ...")
    slope_df, r2_df, buy_df, uf_df, curve_df, r2q_df = ({} for _ in range(6))
    for c in present:
        s = close[c]; v = s.dropna()
        sl, r2 = roll_slope_r2(v.values)
        es, ac, r2q = roll_quad(v.values)
        reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        sl, r2, es, ac, r2q = map(reidx, (sl, r2, es, ac, r2q))
        pv = s.values
        ma50 = s.rolling(50).mean().values
        ma200 = s.rolling(200).mean().values
        buy = (pv > ma200) & (pv > ma50) & (sl > 0) & (r2 > 0.6)
        ufit = np.zeros(n)
        for j in range(PATH, n):
            sp, rp = sl[j - PATH + 1:j + 1], r2[j - PATH + 1:j + 1]
            if np.isfinite(sp).all() and np.isfinite(rp).all():
                ufit[j] = u_fit(sp, rp)
        accel_boost = np.where(np.nan_to_num(ac) > 0, 1.4, 0.7)
        es0, r2q0 = np.nan_to_num(es), np.nan_to_num(r2q)
        # Up-trajectory: end_slope * quad_R2 * accel bonus. Down-trajectory:
        # don't zero it - keep quad_R2 as the floor strength.
        curve = np.where(es0 > 0, es0 * r2q0 * accel_boost, r2q0)
        slope_df[c], r2_df[c], buy_df[c] = sl, r2, np.nan_to_num(buy)
        uf_df[c], curve_df[c], r2q_df[c] = ufit, curve, r2q0

    pv = {c: close[c].values for c in present}
    schemes = ["ew", "linear", "srq", "curve", "ufit"]
    rets = {s: [] for s in schemes}
    breadth = {s: [] for s in schemes}

    for i in range(max(220, PATH), n - FWD, FWD):
        elig = [c for c in present if buy_df[c][i] and np.isfinite(pv[c][i]) and np.isfinite(pv[c][i + FWD])]
        if not elig:
            for s in schemes:
                rets[s].append(0.0)
            continue
        fret = {c: pv[c][i + FWD] / pv[c][i] - 1.0 for c in elig}
        sr = {c: max(slope_df[c][i] * r2_df[c][i], 0.0) for c in elig}
        w = {
            "ew": {c: 1.0 for c in elig},
            "linear": {c: sr[c] * np.tanh(sr[c] / 8.0) for c in elig},   # slope*r2*strength
            "srq": {c: sr[c] * r2q_df[c][i] for c in elig},              # slope*r2*R2_quad
            "curve": {c: curve_df[c][i] for c in elig},
            "ufit": {c: sr[c] * uf_df[c][i] for c in elig},
        }
        for s in schemes:
            tot = sum(w[s].values())
            rets[s].append(wmean(fret, w[s]) if tot > 0 else 0.0)
            breadth[s].append(sum(1 for c in elig if w[s][c] > 0))

    ppy = 252 / FWD
    npd = len(rets["ew"])
    print(f"Periods: {npd}, {ppy:.1f}/yr\n")
    print(f"{'scheme':8s} {'CAGR':>7s} {'vol':>7s} {'Sharpe':>7s} {'maxDD':>8s} {'avg names':>9s}")
    print("-" * 52)
    for s in schemes:
        cagr, vol, sh, dd = perf(rets[s], ppy)
        nm = np.mean(breadth[s]) if breadth[s] else float("nan")
        print(f"{s:8s} {cagr:>7.1%} {vol:>7.1%} {sh:>7.2f} {dd:>8.1%} {nm:>9.1f}")
    print("\ncurve keeps breadth (like linear) - does it beat linear on Sharpe,")
    print("and match ufit's edge without ufit's concentration?")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
