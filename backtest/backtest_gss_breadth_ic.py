"""
Does market breadth / credit-risk-appetite predict the BROAD market's (SPY,
QQQ) forward return? Different shape of predictor than the per-asset trend-
strength test: these are AGGREGATE signals about market health, not a single
ETF's own technical reading. Also structurally different statistically - one
time series of ~5000 days, not 36 assets pooled - so fewer independent obs;
read the IC/hit-rate with that in mind.

Candidates:
  breadth_200   - % of the 36-name universe with price > its own 200DMA
  breadth_slope - % of the universe with a positive 20d OLS slope
  hyg_lqd       - HYG/LQD ratio level (junk vs IG credit - risk appetite)
  hyg_lqd_chg   - 20d rate of change of that ratio (momentum in credit RA)
  rsp_spy       - RSP/SPY ratio level (equal-wt vs cap-wt - breadth/conc.)
  rsp_spy_chg   - 20d rate of change of that ratio

Targets: forward SPY and QQQ return at 5d/10d/21d. Same IC (rank corr) +
quintile hit-rate/mean-return framework as the trend-strength IC test.
2007-2026.
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
WIN = 20
HORIZONS = [5, 10, 21]


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); lp = np.log(p)
    if n < win or sliding_window_view is None:
        return sl
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn
    sl[win - 1:] = s * 1000
    return sl


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    extra = ["HYG", "LQD", "RSP"]
    dl = tickers + [t for t in extra if t not in tickers]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe for breadth: {len(present)}\n")

    # --- breadth: % above 200DMA, % positive 20d slope ---
    above200 = np.zeros(n); slope_pos = np.zeros(n); counted = np.zeros(n)
    for c in present:
        v = close[c].reindex(idx)
        ma200 = v.rolling(200).mean()
        sl = roll_sr(v.values, WIN)
        valid = v.notna().values & np.isfinite(ma200.values) & np.isfinite(sl)
        above200[valid] += (v.values[valid] > ma200.values[valid])
        slope_pos[valid] += (sl[valid] > 0)
        counted[valid] += 1
    breadth_200 = np.where(counted > 5, above200 / np.maximum(counted, 1), np.nan) * 100
    breadth_slope = np.where(counted > 5, slope_pos / np.maximum(counted, 1), np.nan) * 100

    # --- credit / concentration ratios ---
    hyg = close["HYG"].reindex(idx); lqd = close["LQD"].reindex(idx)
    rsp = close["RSP"].reindex(idx); spy = close["SPY"].reindex(idx); qqq = close["QQQ"].reindex(idx)
    hyg_lqd = (hyg / lqd).values
    rsp_spy = (rsp / spy).values
    hyg_lqd_chg = pd.Series(hyg_lqd).pct_change(20).values * 100
    rsp_spy_chg = pd.Series(rsp_spy).pct_change(20).values * 100

    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)

    indicators = {
        "breadth_200": breadth_200, "breadth_slope": breadth_slope,
        "hyg_lqd": hyg_lqd, "hyg_lqd_chg": hyg_lqd_chg,
        "rsp_spy": rsp_spy * 100, "rsp_spy_chg": rsp_spy_chg,
    }
    targets = {"SPY": spy.values, "QQQ": qqq.values}

    print(f"{'target':>5s} {'horizon':>7s} {'indicator':>13s} {'IC':>7s} {'n':>6s} "
          f"{'Q1 hit%':>8s} {'Q5 hit%':>8s} {'Q1 fwd%':>8s} {'Q5 fwd%':>8s}")
    print("-" * 82)
    for tname, tpx in targets.items():
        for h in HORIZONS:
            fwd = np.full(n, np.nan)
            fwd[si:n - h] = tpx[si + h:n] / tpx[si:n - h] - 1.0
            for name, arr in indicators.items():
                m = np.isfinite(fwd) & np.isfinite(arr)
                m[:si] = False
                if m.sum() < 100:
                    continue
                ind = arr[m]; f = fwd[m]; nobs = len(ind)
                ic = pd.Series(ind).rank().corr(pd.Series(f).rank())
                q = pd.qcut(pd.Series(ind), 5, labels=False, duplicates="drop")
                df = pd.DataFrame({"q": q, "fwd": f})
                g = df.groupby("q")["fwd"]
                hit = g.apply(lambda x: (x > 0).mean()); mean_ = g.mean()
                qlo, qhi = hit.index.min(), hit.index.max()
                print(f"{tname:>5s} {h:>6d}d {name:>13s} {ic:>7.3f} {nobs:>6d} "
                      f"{hit.get(qlo, np.nan)*100:>7.1f}% {hit.get(qhi, np.nan)*100:>7.1f}% "
                      f"{mean_.get(qlo, np.nan)*100:>7.2f}% {mean_.get(qhi, np.nan)*100:>7.2f}%")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
