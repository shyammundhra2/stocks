"""
What's the best indicator of trend strength - and does ANY of them actually
predict whether an ETF will be up or down? Classification quality (does it
distinguish trend from chop) is a different question from PREDICTIVE POWER
(does today's reading forecast tomorrow's direction/magnitude).

Candidates:
  slope_r2   - current conviction score (OLS slope x R2, 20d)
  ER         - Kaufman efficiency ratio (20d)
  R2         - OLS fit alone (20d)
  ADX        - Wilder's Average Directional Index (14d), NOT yet tried this
               session - classic trend-strength indicator, needs OHLC
  pct_200dma - % above/below the 200-day moving average (simple, common)
  slope_z    - slope z-score vs its own recent history (already computed live)

For each, at horizons 5d/10d/21d: TIME-SERIES IC (corr(indicator_today,
fwd_return) within each asset, pooled) + hit-rate by quintile (does a higher
reading actually raise P(up)?). This is IC/hit-rate, not backtest P&L - it
isolates whether the indicator itself has ANY forecasting content before any
strategy is built on top of it. Universe: TREND_ASSETS, 2007-2026.
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

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
WIN, ER_L, ADX_L = 20, 20, 14
HORIZONS = [5, 10, 21]


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan); lp = np.log(p)
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


def er(p, L):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def adx(high, low, close, L=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = h.diff(); down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / L, adjust=False, min_periods=L).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1 / L, adjust=False, min_periods=L).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1 / L, adjust=False, min_periods=L).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / L, adjust=False, min_periods=L).mean().values


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers (OHLC) ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close_df = raw["Close"]; high_df = raw["High"]; low_df = raw["Low"]
    present = [t for t in tickers if t in close_df.columns and close_df[t].notna().sum() > 260]
    idx = close_df.index
    print(f"Universe: {len(present)}\n")

    rows = {h: [] for h in HORIZONS}   # per-horizon: list of (indicator_name, value, fwd_ret)
    for c in present:
        cl = close_df[c].reindex(idx); hi = high_df[c].reindex(idx); lo = low_df[c].reindex(idx)
        v = cl.values
        slope, r2 = roll_sr(v, WIN)
        slope_r2 = slope * r2
        ER = er(v, ER_L)
        ADX = adx(hi.values, lo.values, v, ADX_L)
        ma200 = cl.rolling(200).mean().values
        pct200 = (v - ma200) / ma200 * 100
        # slope z-score vs trailing 60-bar history of the same 20d slope (matches live calc)
        sl_series = pd.Series(slope)
        slope_z = ((sl_series - sl_series.rolling(60).mean()) / sl_series.rolling(60).std()).values

        n = len(v)
        for h in HORIZONS:
            fwd = np.full(n, np.nan)
            fwd[:n - h] = v[h:] / v[:n - h] - 1.0
            valid = np.isfinite(fwd)
            for name, arr in [("slope_r2", slope_r2), ("ER", ER), ("R2", r2),
                               ("ADX", ADX), ("pct_200dma", pct200), ("slope_z", slope_z)]:
                m = valid & np.isfinite(arr)
                if m.sum() > 50:
                    rows[h].append((name, arr[m], fwd[m]))

    print(f"{'horizon':>7s} {'indicator':>10s} {'IC (spearman)':>14s} {'n':>8s} "
          f"{'Q1 hit%':>8s} {'Q5 hit%':>8s} {'Q1 fwd%':>8s} {'Q5 fwd%':>8s}")
    print("-" * 80)
    for h in HORIZONS:
        by_name = {}
        for name, ind, fwd in rows[h]:
            by_name.setdefault(name, [[], []])
            by_name[name][0].append(ind); by_name[name][1].append(fwd)
        for name in ["slope_r2", "ER", "R2", "ADX", "pct_200dma", "slope_z"]:
            if name not in by_name:
                continue
            ind = np.concatenate(by_name[name][0]); fwd = np.concatenate(by_name[name][1])
            n = len(ind)
            ic = pd.Series(ind).rank().corr(pd.Series(fwd).rank())
            q = pd.qcut(pd.Series(ind), 5, labels=False, duplicates="drop")
            df = pd.DataFrame({"q": q, "fwd": fwd})
            g = df.groupby("q")["fwd"]
            hit = g.apply(lambda x: (x > 0).mean())
            mean_ = g.mean()
            qlo, qhi = hit.index.min(), hit.index.max()
            print(f"{h:>6d}d {name:>10s} {ic:>13.3f} {n:>8d} "
                  f"{hit.get(qlo, np.nan)*100:>7.1f}% {hit.get(qhi, np.nan)*100:>7.1f}% "
                  f"{mean_.get(qlo, np.nan)*100:>7.2f}% {mean_.get(qhi, np.nan)*100:>7.2f}%")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
