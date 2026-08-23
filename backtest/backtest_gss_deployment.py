"""
Reality check on the ~30% CAGR: how DEPLOYED is the backtest, and does idle
cash earn anything?

The other GSS backtests renormalize weights to sum to 1 -> they are ~100%
INVESTED whenever any name qualifies (cash only on flat days, earning 0%). The
LIVE Kelly/vol-capped book deploys only ~38-50% and parks the rest. So the 30%
CAGR is a fully-invested figure, not a 50%-deployed one. This measures actual
deployment and reprices under two realistic funding models with T-bill (BIL,
total return) credited on idle cash:

  full (~100%)  : current backtest, BIL on the rare flat days
  50% cap       : scale equity exposure to 50%, BIL on the other ~50%

Last-trading-day (turn-of-month) rebalance, net@5bps, 2020-26, train/test.
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
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, CAP, COST, DEPLOY_CAP = 20, 20, 0.40, 0.35, 15.0, 0.15, 5.0, 0.50


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
    dl = tickers + (["BIL"] if "BIL" not in tickers else [])
    print(f"Downloading {len(dl)} tickers (+BIL cash proxy) ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    bil = close["BIL"].pct_change().fillna(0).values
    bil_yield = (np.prod(1 + bil[np.isfinite(bil)]) ** (252 / len(bil)) - 1)
    print(f"Universe: {len(present)}   BIL cash yield ~{bil_yield:.1%}/yr\n")

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

    dser = pd.Series(idx); rebal_days = {g.index[-1] for _, g in dser.groupby(dser.dt.to_period("M"))
                                        if g.index[-1] >= start_i}

    # simulate once; capture per-day equity return and invested fraction
    w = {c: 0.0 for c in present}; eqret = np.zeros(n); turn = np.zeros(n); invested = np.zeros(n); nnames = []
    for j in range(start_i, n - 1):
        if j in rebal_days:
            sel = {c: route(c, j) for c in present}
            sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
            tot = sum(sel.values())
            tgt = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
            s2 = sum(tgt.values()); tgt = {c: v / s2 for c, v in tgt.items()} if s2 > 0 else {}
            full = {c: tgt.get(c, 0.0) for c in present}
            turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            nnames.append(sum(1 for v in w.values() if v > 0))
        s = 0.0
        for c in present:
            if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
        eqret[j + 1] = s; invested[j + 1] = sum(w.values())

    sc = slice(start_i + 1, n); dts = idx[sc]
    eq_net = eqret[sc] - turn[sc] * (COST / 1e4)
    inv = invested[sc]; bilc = bil[sc]
    pct_days_inv = float((inv > 0).mean()); avg_inv = float(inv.mean()); avg_names = np.mean(nnames)

    print(f"DEPLOYMENT: invested on {pct_days_inv:.0%} of days, avg gross exposure {avg_inv:.0%}, "
          f"avg {avg_names:.1f} names when rebalanced")
    print("(so the ~30% CAGR is a ~100%-deployed book, NOT 50% deployed)\n")

    # funding models
    full_ret = eq_net + bilc * (1.0 - inv)                       # BIL on the flat-day cash only
    cap_ret = eq_net * DEPLOY_CAP + bilc * (1.0 - DEPLOY_CAP * inv)  # 50% equity exposure, rest BIL

    print(f"turn-of-month adaptive, 15% cap, net@{COST}bps, SPLIT {SPLIT.date()}\n")
    print(f"{'funding model':28s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s}")
    print("-" * 66)
    for lab, r in [("full (~100% deployed)", full_ret),
                   (f"{DEPLOY_CAP:.0%} cap + BIL on rest", cap_ret),
                   ("SPY buy&hold", close["SPY"].pct_change().values[sc])]:
        f, tr, te, cg, dd = perf(r, dts)
        print(f"{lab:28s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
