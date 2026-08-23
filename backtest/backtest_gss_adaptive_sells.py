"""
Does adding DAILY sell signals to the monthly-rebalanced adaptive router help?
(The rebalance sweep held blindly to the boundary - no sells.)

Entry: monthly (21d), router TREND>=0.40 -> MOM (slope*r2), CHOP<=0.35 -> REV
       (oversold depth), regime-aware conviction, 15% cap.
Daily sells (sleeve-specific), weight -> cash until next rebalance:
  MOM: exit on close < 20-DMA            (trend break)
  REV: exit on RSI2 > 70 (bounce done)   OR close < 20-DMA (stop)
Daily equity so Sharpe AND max drawdown are honest. Net@5bps, train/test.
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
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, RSI_EXIT, REBAL, COST = 20, 20, 0.40, 0.35, 15.0, 70.0, 21, 5.0


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


def perf(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** (252 / len(r)) - 1
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

    pv = {}; ma50 = {}; ma200 = {}; dma20 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values; ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values; dma20[c] = close[c].rolling(20).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return None, 0.0
        a200 = pv[c][i] > ma200[c][i]
        if e >= ER_HI and a200 and pv[c][i] > ma50[c][i] and SL[c][i] > 0:
            return "MOM", max(SL[c][i] * R2[c][i], 0.0)
        if e <= ER_LO and a200 and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return "REV", np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return None, 0.0

    def sim(sell_mode):
        w = {c: 0.0 for c in present}; sleeve = {}; gross = np.zeros(n); turn = np.zeros(n)
        for j in range(start_i, n - 1):
            if (j - start_i) % REBAL == 0:
                sel = {}; slv = {}
                for c in present:
                    sg, base = route(c, j)
                    if sg and base > 0 and fin(pv[c][j]):
                        sel[c] = base; slv[c] = sg
                tot = sum(sel.values())
                tgt = {c: min(sel[c] / tot, 0.15) for c in sel} if tot > 0 else {}
                s2 = sum(tgt.values())
                tgt = {c: v / s2 for c, v in tgt.items()} if s2 > 0 else {}
                full = {c: tgt.get(c, 0.0) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present)
                w = full; sleeve = slv
            s = 0.0
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
            gross[j + 1] = s
            if sell_mode != "none":
                for c in list(present):
                    if w[c] <= 0 or not fin(pv[c][j + 1]):
                        continue
                    p = pv[c][j + 1]
                    if sell_mode == "200dma":
                        hit = p < ma200[c][j + 1]                                # wide trend-break exit
                    elif sleeve.get(c) == "MOM":
                        hit = p < dma20[c][j + 1]
                    else:
                        hit = (fin(RS2[c][j + 1]) and RS2[c][j + 1] > RSI_EXIT) or p < ma200[c][j + 1]
                    if hit:
                        turn[j + 1] += w[c]; w[c] = 0.0
        sc = slice(start_i + 1, n)
        net = gross[sc] - turn[sc] * (COST / 1e4)
        return net, idx[sc], turn[sc].sum() * 252 / (n - start_i - 1)

    spy = close["SPY"].pct_change().values[start_i + 1:]; dts = idx[start_i + 1:]
    print(f"monthly rebalance ({REBAL}d), 15% cap, net@{COST}bps, SPLIT {SPLIT.date()}  (daily equity)\n")
    print(f"{'config':22s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s}")
    print("-" * 66)
    f, tr, te, cg, dd = perf(spy, dts)
    print(f"{'SPY buy&hold':22s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {'-':>8s}")
    for mode, lab in [("none", "blind hold"), ("sleeve", "+sleeve sells"), ("200dma", "+200-DMA sell")]:
        r, d, ty = sim(mode)
        f, tr, te, cg, dd = perf(r, d)
        print(f"{('monthly '+lab):22s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {ty:>7.1f}x")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
