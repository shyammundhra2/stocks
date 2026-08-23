"""
Which day to rebalance the adaptive router (monthly, blind hold)?

Two questions:
  1. PHASE ROBUSTNESS - does the strategy depend on WHICH 21-day phase you
     happen to start on? Sweep all 21 offsets; a robust edge is flat across
     them, a fragile one has a lucky phase.
  2. CALENDAR RULES - is there a turn-of-month effect? Compare:
       first-of-month, last-of-month (turn-of-month), mid-month, 4th Tuesday.

Adaptive: TREND>=0.40 -> MOM (slope*r2), CHOP<=0.35 -> REV, 15% cap, blind hold
to next rebalance. Daily equity, net@5bps, train/test.
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
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, REBAL, CAP, COST = 20, 20, 0.40, 0.35, 15.0, 21, 0.15, 5.0


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

    pv = {}; ma50 = {}; ma200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values; ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 200)
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

    def sim(rebal_days):
        w = {c: 0.0 for c in present}; gross = np.zeros(n); turn = np.zeros(n)
        for j in range(start_i, n - 1):
            if j in rebal_days:
                sel = {c: route(c, j) for c in present}
                sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
                tot = sum(sel.values())
                tgt = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
                s2 = sum(tgt.values()); tgt = {c: v / s2 for c, v in tgt.items()} if s2 > 0 else {}
                full = {c: tgt.get(c, 0.0) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.0
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
            gross[j + 1] = s
        sc = slice(start_i + 1, n)
        return gross[sc] - turn[sc] * (COST / 1e4), idx[sc], turn[sc].sum() * 252 / (n - start_i - 1)

    # ---- 1. phase robustness: every 21-day offset ----
    print("PHASE SWEEP (21-day rebalance, each starting offset)\n")
    print(f"{'offset':>7s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s}")
    print("-" * 46)
    fulls = []
    for off in range(REBAL):
        rd = set(range(start_i + off, n - 1, REBAL))
        r, d, ty = sim(rd); f, tr, te, cg, dd = perf(r, d); fulls.append(f)
        print(f"{off:>7d} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%}")
    fulls = np.array(fulls)
    print(f"\n  phase Sharpe: mean {fulls.mean():.2f}  min {fulls.min():.2f}  "
          f"max {fulls.max():.2f}  std {fulls.std():.2f}  (low std = robust to day choice)\n")

    # ---- 2. calendar rules ----
    dser = pd.Series(idx)
    ym = dser.dt.to_period("M")
    first = set(np.where(ym.values != np.roll(ym.values, 1))[0])          # first trading day of month
    last = set(np.where(ym.values != np.roll(ym.values, -1))[0])          # last trading day of month
    mid = set()
    for _, g in dser.groupby(ym):
        pos = g.index.tolist(); mid.add(pos[len(pos) // 2])               # ~mid month
    tue4 = set()                                                          # 4th Tuesday
    for _, g in dser.groupby(ym):
        tues = [i for i in g.index if dser.iloc[i].weekday() == 1]
        if len(tues) >= 4:
            tue4.add(tues[3])
    rules = [("first-of-month", first), ("last-of-month(ToM)", last),
             ("mid-month", mid), ("4th Tuesday", tue4)]
    print("CALENDAR RULES (monthly)\n")
    print(f"{'rule':20s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s}")
    print("-" * 62)
    for lab, days in rules:
        rd = {j for j in days if j >= start_i}
        r, d, ty = sim(rd); f, tr, te, cg, dd = perf(r, d)
        print(f"{lab:20s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {ty:>7.1f}x")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
