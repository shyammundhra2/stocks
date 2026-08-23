"""
How often to rebalance the adaptive router strategy (TREND>=0.40 -> momentum,
CHOP<=0.35 -> RSI2 reversion, regime-aware conviction)? Sweep the hold/rebalance
period. Net of 5bps, 2020-2026, train/test, with turnover/yr so you see the
cost side. Reversion needs speed; momentum tolerates slower - this finds the
balance.
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
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, COST = 20, 20, 0.40, 0.35, 15.0, 5.0
HOLDS = [2, 3, 5, 10, 21]


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


def rsi(prices, p):
    d = np.diff(prices, prepend=prices[0])
    up = pd.Series(np.where(d > 0, d, 0.0)); dn = pd.Series(np.where(d < 0, -d, 0.0))
    return (100 - 100 / (1 + up.rolling(p).mean() / dn.rolling(p).mean().replace(0, np.nan))).values


def eff_ratio(prices, L):
    s = pd.Series(prices)
    return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dates, ppy):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * ppy) / (x.std() * np.sqrt(ppy)) if len(x) > 10 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** (ppy / len(r)) - 1
    return sh(np.ones(len(r), bool)), sh(dates < SPLIT), sh(dates >= SPLIT), cagr, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {}; ma50 = {}; ma200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values; ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return 0.0
        a200 = pv[c][i] > ma200[c][i]
        if e >= ER_HI and a200 and pv[c][i] > ma50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.0)
        if e <= ER_LO and a200 and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return 0.0

    def run(hold):
        prev = {}; rr = []; dts = []; turn_tot = 0.0; rebs = 0
        for i in range(max(start_i, 200), n - hold, hold):
            sel = {}
            for c in present:
                if not fin(pv[c][i + hold]):
                    continue
                base = route(c, i)
                if base > 0:
                    sel[c] = base                        # 15% cap applied at weight stage below
            tot = sum(sel.values())
            w = {c: sel[c] / tot for c in sel} if tot > 0 else {}
            # 15% per-name cap, renormalize
            w = {c: min(x, 0.15) for c, x in w.items()}
            s2 = sum(w.values())
            if s2 > 0:
                w = {c: x / s2 for c, x in w.items()}
            gross = sum(w[c] * (pv[c][i + hold] / pv[c][i] - 1.0) for c in w)
            turn = sum(abs(w.get(c, 0.0) - prev.get(c, 0.0)) for c in set(w) | set(prev))
            rr.append(gross - turn * (COST / 1e4)); dts.append(idx[i]); prev = w
            turn_tot += turn; rebs += 1
        turnyr = (turn_tot / rebs) * (252 / hold) if rebs else 0
        return np.array(rr), pd.DatetimeIndex(dts), turnyr

    print(f"adaptive (TREND>={ER_HI}/CHOP<={ER_LO}), 15% cap, net@{COST}bps, SPLIT {SPLIT.date()}\n")
    print(f"{'hold':>5s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s}")
    print("-" * 52)
    for h in HOLDS:
        r, d, ty = run(h)
        f, tr, te, cg, mdd = perf(r, d, 252 / h)
        print(f"{h:>4d}d {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {mdd:>7.1%} {ty:>7.1f}x")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
