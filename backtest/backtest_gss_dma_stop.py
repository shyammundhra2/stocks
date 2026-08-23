"""
Find a DMA (moving-average) stop where the trend edge SURVIVES daily risk
discipline. A tight ATR stop whipsawed the edge away (backtest_gss_hold_exits);
a wider DMA stop should let trend pullbacks breathe while still cutting real
breaks - keeping most of the return AND controlling drawdown.

Same book as before (equal-weight BUY-eligible: price>200/50-SMA, slope>0,
R2>0.6, window 20). Enter every REBAL days; EXIT DAILY when close < N-day SMA.
Sweep N in {20,50,100,150,200} x REBAL in {21,42}. Daily equity, 2020-2026,
train/test split. Baselines: blind hold (no stop) and SPY.
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

DATA_START, TRADE_START, END = "2019-01-01", "2020-01-01", "2026-08-21"
SPLIT = pd.Timestamp("2023-06-01")
WIN, R2T = 20, 0.6
DMAS = [20, 50, 100, 150, 200]
REBALS = [21, 42]


def roll_sr(prices, win):
    n = len(prices); slope = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < win or sliding_window_view is None:
        return slope, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); denom = float(xc @ xc)
    ym = W.mean(1); slc = (W - ym[:, None]) @ xc / denom
    pred = slc[:, None] * x[None, :] + (ym - slc * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pred) ** 2).sum(1)
    slope[win - 1:] = slc * 1000.0
    r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return slope, r2


def stats(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r))
    cagr = eq[-1] ** (252 / len(r)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh(np.ones(len(r), bool)), sh(dates < SPLIT), sh(dates >= SPLIT), cagr, dd


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {DATA_START}..{END} ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {c: close[c].values for c in present}
    ma50 = {c: close[c].rolling(50).mean().values for c in present}
    ma200 = {c: close[c].rolling(200).mean().values for c in present}
    dma = {p: {c: close[c].rolling(p).mean().values for c in present} for p in DMAS}
    sl = {}; r2 = {}
    for c in present:
        v = close[c].dropna()
        s, r = roll_sr(v.values, WIN)
        reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        sl[c], r2[c] = reidx(s), reidx(r)

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))

    def buy(c, i):
        return (pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i]
                and sl[c][i] > 0 and r2[c][i] > R2T)

    def sim(REBAL, dma_p):
        dret = np.zeros(n); tim = 0; days = 0; i = start_i
        while i < n - 1:
            elig = [c for c in present if buy(c, i) and np.isfinite(pv[c][i])]
            N = len(elig); held = set(elig)
            for j in range(i, min(i + REBAL, n - 1)):
                if N:
                    s = 0.0
                    for c in elig:
                        if c in held and np.isfinite(pv[c][j + 1]) and np.isfinite(pv[c][j]):
                            s += pv[c][j + 1] / pv[c][j] - 1.0
                    dret[j + 1] = s / N
                    tim += len(held) / N; days += 1
                if dma_p is not None:
                    d = dma[dma_p]
                    for c in list(held):
                        p = pv[c][j + 1]
                        if np.isfinite(p) and np.isfinite(d[c][j + 1]) and p < d[c][j + 1]:
                            held.discard(c)
            i += REBAL
        return dret[start_i + 1:], idx[start_i + 1:], (tim / days if days else 0)

    spy_r = close["SPY"].pct_change().values[start_i + 1:]; dts = idx[start_i + 1:]

    print(f"win {WIN}, R2 {R2T}, SPLIT {SPLIT.date()}  (daily equity; exits always on)\n")
    print(f"{'config':24s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'inMkt':>6s}")
    print("-" * 68)
    sf, st, se, sc, sd = stats(spy_r, dts)
    print(f"{'SPY buy&hold':24s} {sf:>6.2f} {st:>6.2f} {se:>6.2f} {sc:>7.1%} {sd:>7.1%} {'100%':>6s}")
    for REBAL in REBALS:
        r, d, tm = sim(REBAL, None)
        f, tr, te, cg, dd = stats(r, d)
        print(f"{'hold '+str(REBAL)+' blind (no stop)':24s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {tm:>6.0%}")
        for p in DMAS:
            r, d, tm = sim(REBAL, p)
            f, tr, te, cg, dd = stats(r, d)
            print(f"{'hold '+str(REBAL)+'  stop<'+str(p)+'DMA':24s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {tm:>6.0%}")
        print()

    print("inMkt = avg fraction of the book actually held (rest stopped to cash).")
    print("Edge survives if a DMA stop keeps Sharpe near blind-hold while cutting maxDD.")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
