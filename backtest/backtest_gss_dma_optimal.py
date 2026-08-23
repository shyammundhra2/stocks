"""
Find the optimal DMA stop period (a "new 50") for the GSS trend book, on a
NET-of-cost basis - because tighter stops cut drawdown but churn, and cost is
what sets the real optimum.

Book: equal-weight BUY-eligible names (price>200/50-SMA, slope>0, R2>0.6,
window 20), rebalanced every 21 days; EXIT DAILY when close < N-day SMA.
Weights tracked day-by-day so turnover is real; net = gross - turnover*cost.
Fine DMA grid, 2020-2026, daily equity, train/test split. Ranked by net@5bps.
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
WIN, R2T, REBAL = 20, 0.6, 21
DMAS = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100]
COSTS = [0.0, 5.0, 10.0]   # one-way bps per unit turnover


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


def perf(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** (252 / len(r)) - 1
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
    fin = np.isfinite

    def buy(c, i):
        return (pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i] and sl[c][i] > 0 and r2[c][i] > R2T)

    def sim(dma_p):
        w = {c: 0.0 for c in present}
        gross = np.zeros(n); turn = np.zeros(n); inmkt = 0.0; days = 0
        for j in range(start_i, n - 1):
            if (j - start_i) % REBAL == 0:
                elig = [c for c in present if buy(c, j) and fin(pv[c][j])]
                N = len(elig)
                tgt = {c: (1.0 / N if c in elig else 0.0) for c in present} if N else {c: 0.0 for c in present}
                turn[j + 1] += sum(abs(tgt[c] - w[c]) for c in present)
                w = tgt
            r = 0.0; wsum = 0.0
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    r += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0); wsum += w[c]
            gross[j + 1] = r; inmkt += wsum; days += 1
            if dma_p is not None:
                d = dma[dma_p]
                for c in present:
                    if w[c] > 0 and fin(pv[c][j + 1]) and fin(d[c][j + 1]) and pv[c][j + 1] < d[c][j + 1]:
                        turn[j + 1] += w[c]; w[c] = 0.0
        s = slice(start_i + 1, n)
        return gross[s], turn[s], idx[s], (inmkt / days if days else 0), turn[s].sum() * 252 / days

    print(f"win {WIN}, R2 {R2T}, rebalance {REBAL}, SPLIT {SPLIT.date()}  (net of cost)\n")
    print(f"{'stop':>7s} {'gross':>6s} {'net@5':>6s} {'net@10':>6s} {'TR@5':>6s} {'TE@5':>6s} "
          f"{'CAGR@5':>7s} {'DD@5':>7s} {'inMkt':>6s} {'turn/yr':>7s}")
    print("-" * 76)
    rows = []
    for p in DMAS:
        gross, turn, dts, inmkt, turnyr = sim(p)
        def netstats(bps):
            net = gross - turn * (bps / 1e4)
            return perf(net, dts)
        g = perf(gross, dts)[0]
        f5, tr5, te5, cg5, dd5 = netstats(5)
        f10 = netstats(10)[0]
        rows.append((p, g, f5, f10, tr5, te5, cg5, dd5, inmkt, turnyr))
        print(f"{p:>6d}D {g:>6.2f} {f5:>6.2f} {f10:>6.2f} {tr5:>6.2f} {te5:>6.2f} "
              f"{cg5:>7.1%} {dd5:>7.1%} {inmkt:>6.0%} {turnyr:>6.1f}x")

    best = max(rows, key=lambda x: (x[2] if np.isfinite(x[2]) else -9))
    robust = max(rows, key=lambda x: (min(x[4], x[5]) if np.isfinite(x[4]) and np.isfinite(x[5]) else -9))
    print(f"\nBest net@5bps Sharpe: {best[0]}D (net {best[2]:.2f}, DD {best[7]:.1%}, inMkt {best[8]:.0%})")
    print(f"Most robust (min train/test @5bps): {robust[0]}D "
          f"(train {robust[4]:.2f}, test {robust[5]:.2f})")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
