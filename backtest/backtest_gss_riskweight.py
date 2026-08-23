"""
Test two cheap, robust sizing edges on the GSS trend book (hold 21, 20-DMA
stop, exits daily, net of 5bps), vs equal-weight:
  inv-vol      weight ~ 1/trailing-vol (risk parity - stop overweighting the
               volatile metals cluster)
  vol-target   scale total exposure to a 12% vol target (cash the rest);
               de-risk when the book's realized vol spikes
  both         inv-vol + vol-target
2020-2026, daily equity, train/test split.
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
WIN, R2T, REBAL, STOP_DMA, COST = 20, 0.6, 21, 20, 5.0
TGT_VOL = 0.12 / np.sqrt(252)   # 12% annualized -> daily


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
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {c: close[c].values for c in present}
    ret = {c: close[c].pct_change().values for c in present}
    vol30 = {c: pd.Series(ret[c]).rolling(30).std().values for c in present}
    ma50 = {c: close[c].rolling(50).mean().values for c in present}
    ma200 = {c: close[c].rolling(200).mean().values for c in present}
    dma = {c: close[c].rolling(STOP_DMA).mean().values for c in present}
    sl = {}; r2 = {}
    for c in present:
        v = close[c].dropna(); s, r = roll_sr(v.values, WIN)
        reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        sl[c], r2[c] = reidx(s), reidx(r)

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def buy(c, i):
        return (pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i] and sl[c][i] > 0 and r2[c][i] > R2T)

    def sim(scheme):
        w = {c: 0.0 for c in present}
        gross = np.zeros(n); turn = np.zeros(n)
        for j in range(start_i, n - 1):
            if (j - start_i) % REBAL == 0:
                elig = [c for c in present if buy(c, j) and fin(pv[c][j]) and fin(vol30[c][j]) and vol30[c][j] > 0]
                if scheme in ("invvol", "both"):
                    raw_w = {c: 1.0 / vol30[c][j] for c in elig}
                else:
                    raw_w = {c: 1.0 for c in elig}
                tot = sum(raw_w.values())
                tgt = {c: (raw_w[c] / tot if tot > 0 else 0.0) for c in elig}
                if scheme in ("voltgt", "both") and tot > 0:
                    lo = max(j - 30, start_i)
                    pr = np.array([sum(tgt[c] * ret[c][k] for c in elig if fin(ret[c][k])) for k in range(lo, j)])
                    pv_ = pr.std() if len(pr) > 5 else 0.0
                    scal = min(1.0, TGT_VOL / pv_) if pv_ > 0 else 1.0
                    tgt = {c: tgt[c] * scal for c in elig}
                full = {c: tgt.get(c, 0.0) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present)
                w = full
            s = 0.0
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
            gross[j + 1] = s
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(dma[c][j + 1]) and pv[c][j + 1] < dma[c][j + 1]:
                    turn[j + 1] += w[c]; w[c] = 0.0
        sc = slice(start_i + 1, n)
        net = gross[sc] - turn[sc] * (COST / 1e4)
        return net, idx[sc]

    print(f"hold {REBAL}, {STOP_DMA}-DMA stop, net@{COST}bps, vol-tgt {TGT_VOL*np.sqrt(252):.0%}\n")
    print(f"{'scheme':12s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s}")
    print("-" * 50)
    for scheme, lab in [("equal", "equal-wt"), ("invvol", "inv-vol"),
                        ("voltgt", "vol-target"), ("both", "invvol+vtgt")]:
        r, d = sim(scheme)
        f, tr, te, cg, dd = perf(r, d)
        print(f"{lab:12s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
