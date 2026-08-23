"""
Is the monthly rebalance starving the RSI2 reversion sleeve?

RSI2 oversold (< 15) lasts 1-3 days. The live strategy only checks the router
on the 21-day rebalance day, so a REV signal can only enter if the name is
oversold on that ONE day. This isolates the reversion sleeve and compares:

  REV monthly : scan RSI2<15 only on rebalance day, hold 21d to next rebalance
  REV daily   : scan RSI2<15 EVERY day, enter intramonth, exit on RSI2>70
                (bounce done) / close<200DMA / MAXHOLD cap

Both: ER<=0.35 chop + price>200DMA gate, equal-weight active names capped 15%
(rest cash), daily equity, net@5bps, train/test. If daily >> monthly the
cadence is hiding a real edge; if similar, monthly sampling is fine.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

DATA_START, TRADE_START, END = "2019-01-01", "2020-01-01", "2026-08-21"
SPLIT = pd.Timestamp("2023-06-01")
ER_L, ER_LO, RSI_BUY, RSI_EXIT, REBAL, MAXHOLD, CAP, COST = 20, 0.35, 15.0, 70.0, 21, 21, 0.15, 5.0


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

    pv = {}; ma200 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        pv[c] = close[c].values; ma200[c] = close[c].rolling(200).mean().values
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def oversold(c, i):
        return (fin(ER[c][i]) and ER[c][i] <= ER_LO and fin(pv[c][i]) and pv[c][i] > ma200[c][i]
                and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY)

    def weights_from(active):
        if not active:
            return {}
        w = {c: min(1.0 / len(active), CAP) for c in active}   # equal-wt, 15% cap, rest cash
        return w

    def run(mode):
        w = {}; hold_since = {}; gross = np.zeros(n); turn = np.zeros(n); ntrades = 0; ndays_inv = 0
        for j in range(max(start_i, 200), n - 1):
            if mode == "monthly":
                if (j - max(start_i, 200)) % REBAL == 0:
                    active = [c for c in present if oversold(c, j)]
                    neww = weights_from(active)
                    ntrades += sum(1 for c in neww if c not in w)
                else:
                    neww = w                                           # hold to next rebalance
            else:  # daily
                held = dict(w)
                for c in list(held):                                   # exits, checked daily
                    if (RS2[c][j] > RSI_EXIT or pv[c][j] < ma200[c][j]
                            or (j - hold_since[c]) >= MAXHOLD):
                        del held[c]; hold_since.pop(c, None)
                for c in present:                                      # entries, checked daily
                    if c not in held and oversold(c, j):
                        hold_since[c] = j; held[c] = 0.0; ntrades += 1
                neww = weights_from(list(held))
            turn[j + 1] += sum(abs(neww.get(c, 0.0) - w.get(c, 0.0)) for c in set(neww) | set(w))
            w = neww
            if w:
                ndays_inv += 1
            s = 0.0
            for c in w:
                if fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.0)
            gross[j + 1] = s
        sc = slice(max(start_i, 200) + 1, n)
        net = gross[sc] - turn[sc] * (COST / 1e4)
        days = n - max(start_i, 200) - 1
        return net, idx[sc], turn[sc].sum() * 252 / days, ntrades, ndays_inv / days

    spy = close["SPY"].pct_change().values[max(start_i, 200) + 1:]; dts = idx[max(start_i, 200) + 1:]
    print(f"REV sleeve only, ER<={ER_LO} & RSI2<{RSI_BUY:.0f} & >200DMA, 15% cap, net@{COST}bps, SPLIT {SPLIT.date()}\n")
    print(f"{'config':16s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s} {'trades':>7s} {'%inv':>6s}")
    print("-" * 82)
    f, tr, te, cg, dd = perf(spy, dts)
    print(f"{'SPY buy&hold':16s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {'-':>8s} {'-':>7s} {'-':>6s}")
    for mode, lab in [("monthly", "REV monthly"), ("daily", "REV daily")]:
        r, d, ty, nt, inv = run(mode)
        f, tr, te, cg, dd = perf(r, d)
        print(f"{lab:16s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {ty:>7.1f}x {nt:>7d} {inv:>5.0%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
