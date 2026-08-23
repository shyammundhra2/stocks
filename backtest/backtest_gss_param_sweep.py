"""
Disciplined parameter search for the GSS trend strategy, since 2020.

Sweeps the core SIGNAL parameters (they matter more than sizing, per earlier
tests): the trend-fit window, the BUY R2 threshold, and the holding period.
Book = equal-weight the BUY-eligible names (price>200/50-SMA, slope>0,
R2>thresh), rebalanced every `hold` days, non-overlapping, 2020-2026.

The point is NOT the in-sample best (that overfits). Every combo is scored on
a TRAIN half (< 2023-06) and a TEST half (>= 2023-06). An "edge" only counts
if it holds in BOTH halves. Baselines: SPY buy&hold and the production default
(window 20, R2 0.6, hold ~21).
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

DATA_START = "2019-01-01"     # burn-in for 200-SMA into 2020
TRADE_START = "2020-01-01"
END = "2026-08-21"
SPLIT = pd.Timestamp("2023-06-01")

WINDOWS = [10, 15, 20, 30, 40]
R2S = [0.4, 0.5, 0.6, 0.7]
HOLDS = [5, 10, 21, 42]


def roll_sr(prices, win):
    n = len(prices)
    slope = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < win or sliding_window_view is None:
        return slope, r2
    W = sliding_window_view(lp, win)
    x = np.arange(win); xc = x - x.mean(); denom = float(xc @ xc)
    ym = W.mean(1)
    slc = (W - ym[:, None]) @ xc / denom
    b = ym - slc * x.mean()
    pred = slc[:, None] * x[None, :] + b[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1)
    ssr = ((W - pred) ** 2).sum(1)
    r2w = np.where(sst > 0, 1.0 - ssr / sst, 0.0)
    slope[win - 1:] = slc * 1000.0
    r2[win - 1:] = np.clip(r2w, 0.0, 1.0)
    return slope, r2


def sharpe(rets, ppy):
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 5 or r.std() == 0:
        return np.nan
    return (r.mean() * ppy) / (r.std() * np.sqrt(ppy))


def maxdd(rets):
    r = np.asarray(rets, float); r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    eq = np.cumprod(1 + r)
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {DATA_START}..{END} ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {c: close[c].values for c in present}
    ma50 = {c: close[c].rolling(50).mean().values for c in present}
    ma200 = {c: close[c].rolling(200).mean().values for c in present}
    sr_cache = {}   # window -> {sym: (slope, r2)}
    for w in WINDOWS:
        sr_cache[w] = {}
        for c in present:
            v = close[c].dropna()
            s, r = roll_sr(v.values, w)
            reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
            sr_cache[w][c] = (reidx(s), reidx(r))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    split_i_date = SPLIT

    def eval_combo(w, r2t, hold):
        sl = {c: sr_cache[w][c][0] for c in present}
        r2 = {c: sr_cache[w][c][1] for c in present}
        dates, rr = [], []
        for i in range(max(start_i, 200), n - hold, hold):
            elig = [c for c in present
                    if pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i]
                    and sl[c][i] > 0 and r2[c][i] > r2t
                    and np.isfinite(pv[c][i + hold])]
            dates.append(idx[i])
            rr.append(np.mean([pv[c][i + hold] / pv[c][i] - 1.0 for c in elig]) if elig else 0.0)
        rr = np.array(rr); dates = pd.DatetimeIndex(dates)
        ppy = 252 / hold
        tr = rr[dates < split_i_date]; te = rr[dates >= split_i_date]
        return sharpe(rr, ppy), sharpe(tr, ppy), sharpe(te, ppy), maxdd(rr), len(rr)

    # SPY benchmark
    spy = close["SPY"] if "SPY" in close.columns else None
    results = []
    for w in WINDOWS:
        for r2t in R2S:
            for h in HOLDS:
                full, tr, te, dd, npd = eval_combo(w, r2t, h)
                results.append((w, r2t, h, full, tr, te, dd, npd))

    def spy_sharpe(hold):
        p = spy.values; dates, rr = [], []
        for i in range(max(start_i, 200), n - hold, hold):
            if np.isfinite(p[i]) and np.isfinite(p[i + hold]):
                dates.append(idx[i]); rr.append(p[i + hold] / p[i] - 1.0)
        rr = np.array(rr); dates = pd.DatetimeIndex(dates); ppy = 252 / hold
        return sharpe(rr, ppy), sharpe(rr[dates < split_i_date], ppy), sharpe(rr[dates >= split_i_date], ppy)

    print(f"Rebalances scored; SPLIT at {SPLIT.date()}  (train < split, test >= split)\n")
    hdr = f"{'win':>4s} {'R2':>4s} {'hold':>5s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'min(tr,te)':>10s} {'maxDD':>7s}"

    # 1) production default
    print("PRODUCTION DEFAULT (win 20, R2 0.6, hold 21):")
    f, tr, te, dd, npd = eval_combo(20, 0.6, 21)
    print(hdr)
    print(f"{20:>4d} {0.6:>4.1f} {21:>5d} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {min(tr,te):>10.2f} {dd:>7.1%}")
    if spy is not None:
        sf, st, se = spy_sharpe(21)
        print(f"SPY buy&hold (hold 21):        FULL {sf:.2f}  TRAIN {st:.2f}  TEST {se:.2f}")

    # 2) top by robustness = min(train, test)
    robust = sorted(results, key=lambda x: -min(x[4], x[5]))[:12]
    print("\nMOST ROBUST (ranked by the WORSE of train/test - overfit-resistant):")
    print(hdr)
    for w, r2t, h, f, tr, te, dd, npd in robust:
        print(f"{w:>4d} {r2t:>4.1f} {h:>5d} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {min(tr,te):>10.2f} {dd:>7.1%}")

    # 3) top by full (the overfit temptation) - for contrast
    best_full = sorted(results, key=lambda x: -x[3])[:5]
    print("\nBEST IN-SAMPLE (FULL) - the overfit trap, shown for contrast:")
    print(hdr)
    for w, r2t, h, f, tr, te, dd, npd in best_full:
        print(f"{w:>4d} {r2t:>4.1f} {h:>5d} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {min(tr,te):>10.2f} {dd:>7.1%}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
