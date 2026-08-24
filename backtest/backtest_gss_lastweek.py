"""
Can rebalancing anywhere in the LAST WEEK (not just the exact last trading
day) still work? Extends the earlier ToM stress test (which only covered
last-2..last-0) to the full last-5-trading-days window, on the SHIPPED
config: adaptive router, equity-only buffered gate (+/-3%), inverse-vol
sizing, MOM cap 15% / REV cap 5%, BIL on idle cash, net@5bps.

Practical framing: you may not always be able to trade on the exact last
day (travel, a bad print, wanting to dodge a single volatile session) - is
any day in the last week close enough?
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
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
CAP, REV_CAP, COST, BUF = 0.15, 0.05, 5.0, 0.03
LAST_N = 5  # last 5 trading days of the month = "last week"


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan); lp = np.log(p)
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn; pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s * 1000; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return sl, r2


def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dates, split):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * 252) / (x.std() * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** (252 / len(r)) - 1
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return sh(np.ones(len(r), bool)), sh(dates < split), sh(dates >= split), cagr, dd


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys()); dl = tickers + ["BIL"]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return None
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return "MOM"
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return "REV"
        return None

    def sim(rebal_days):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        for j in range(si, n - 1):
            if j in rebal_days:
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                sel = {}; kind = {}
                for c in present:
                    if riskoff and c not in DEFENSIVE_ASSETS:
                        continue
                    sg = route(c, j)
                    if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                        sel[c] = 1.0 / VOL[c][j]; kind[c] = sg
                tot = sum(sel.values()); tg = {}
                if tot > 0:
                    for c in sel:
                        tg[c] = min(sel[c] / tot, REV_CAP if kind[c] == "REV" else CAP)
                    s2 = sum(tg.values())
                    if s2 > 0:
                        tg = {c: v / s2 for c, v in tg.items()}
                        tg = {c: min(v, REV_CAP if kind[c] == "REV" else CAP) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, turn[sc].sum() * 252 / (n - si - 1)

    ds = pd.Series(idx); monthpos = [g.index.tolist() for _, g in ds.groupby(ds.dt.to_period("M"))]
    split = np.datetime64("2023-06-01")

    print(f"shipped config: gate + inverse-vol + REV cap 5%, net@{COST}bps\n")
    print(f"{'anchor':14s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'turn/yr':>8s}")
    print("-" * 60)
    fulls = []
    for back in range(LAST_N):
        days = {pos[-1 - back] for pos in monthpos if len(pos) > back and pos[-1 - back] >= si}
        r, dts, ty = sim(days); f, tr, te, cg, dd = perf(r, dts, split); fulls.append(f)
        lab = "last-0 (ToM)" if back == 0 else f"last-{back}"
        print(f"{lab:14s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {dd:>7.1%} {ty:>7.1f}x")

    # "anywhere in the last week" = pooled/randomized anchor: for each month
    # pick a FIXED offset (not cherry-picked per month) - test all 5 as above,
    # plus a "worst-case any single day" summary.
    fulls = np.array(fulls)
    print(f"\nlast-week band Sharpe: mean {fulls.mean():.2f}  min {fulls.min():.2f}  "
          f"max {fulls.max():.2f}  std {fulls.std():.2f}")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
