"""
Does the "hold ~42 days" edge survive keeping DAILY risk discipline?

Decouples the two cadences:
  * ENTRY / re-weight: every REBAL days, equal-weight the BUY-eligible names
    (price>200/50-SMA, slope>0, R2>0.6, window 20).
  * EXIT: checked DAILY. A held name is dropped to cash the day it breaches
    either an ATR stop (entry - 2.5*ATR) or a SELL(MA50) break (price<50-SMA
    and slope<0). Its weight sits in cash until the next rebalance.

Simulated day-by-day, so Sharpe AND max drawdown are measured on the DAILY
equity curve (the earlier sweep's DD across holds was coarse/misleading).
Compares blind-hold (no intra-hold exits) vs daily-exit, at REBAL 21 and 42,
2020-2026, with a train/test split. Benchmark: SPY daily buy&hold.
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
WIN, R2T, ATR_MULT = 20, 0.6, 2.5


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


def stats(rets, dates):
    r = np.asarray(rets, float)
    def sh(mask):
        x = r[mask]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r))
    cagr = eq[-1] ** (252 / len(r)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    tr = dates < SPLIT; te = dates >= SPLIT
    return sh(np.ones(len(r), bool)), sh(tr), sh(te), cagr, dd


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers {DATA_START}..{END} ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"]; high = raw["High"]; low = raw["Low"]
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {c: close[c].values for c in present}
    ma50 = {c: close[c].rolling(50).mean().values for c in present}
    ma200 = {c: close[c].rolling(200).mean().values for c in present}
    sl = {}; r2 = {}; atr = {}
    for c in present:
        v = close[c].dropna()
        s, r = roll_sr(v.values, WIN)
        reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        sl[c], r2[c] = reidx(s), reidx(r)
        tr = pd.concat([high[c] - low[c],
                        (high[c] - close[c].shift()).abs(),
                        (low[c] - close[c].shift()).abs()], axis=1).max(axis=1)
        atr[c] = tr.rolling(14).mean().values

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))

    def buy(c, i):
        return (pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i]
                and sl[c][i] > 0 and r2[c][i] > R2T)

    def sim(REBAL, use_exits):
        dret = np.zeros(n); i = start_i
        while i < n - 1:
            elig = [c for c in present if buy(c, i) and np.isfinite(pv[c][i]) and np.isfinite(atr[c][i])]
            N = len(elig)
            stop = {c: pv[c][i] - ATR_MULT * atr[c][i] for c in elig}
            held = set(elig)
            for j in range(i, min(i + REBAL, n - 1)):
                if N:
                    s = 0.0
                    for c in elig:
                        if c in held and np.isfinite(pv[c][j + 1]) and np.isfinite(pv[c][j]):
                            s += pv[c][j + 1] / pv[c][j] - 1.0
                    dret[j + 1] = s / N
                if use_exits:
                    for c in list(held):
                        p = pv[c][j + 1]
                        if not np.isfinite(p):
                            continue
                        if p < stop[c] or (p < ma50[c][j + 1] and sl[c][j + 1] < 0):
                            held.discard(c)
            i += REBAL
        return dret[start_i + 1:], idx[start_i + 1:]

    spy = close["SPY"].pct_change().values
    spy_r = spy[start_i + 1:]; dts = idx[start_i + 1:]

    print(f"win {WIN}, R2 {R2T}, ATR stop {ATR_MULT}x, SPLIT {SPLIT.date()}  (daily equity)\n")
    print(f"{'config':22s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s}")
    print("-" * 60)
    sf, st, se, sc, sd = stats(spy_r, dts)
    print(f"{'SPY buy&hold':22s} {sf:>6.2f} {st:>6.2f} {se:>6.2f} {sc:>7.1%} {sd:>7.1%}")
    for REBAL in (21, 42):
        for use_exits in (False, True):
            r, d = sim(REBAL, use_exits)
            f, tr, te, cg, dd = stats(r, d)
            tag = f"hold {REBAL}  {'+daily exits' if use_exits else 'blind hold'}"
            print(f"{tag:22s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%}")

    print("\nBlind = held to the rebalance boundary (ignores stops/sells intra-hold).")
    print("+daily exits = ATR stop + SELL(MA50) checked every day, weight -> cash on exit.")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
