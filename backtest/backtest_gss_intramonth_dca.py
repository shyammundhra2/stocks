"""
Can you buy SMALL portions intramonth when RSI2 fires? Different from the failed
'flood' test (bought every dip full-size, renormalized daily -> -44% DD). Here:
hold a small cash RESERVE and deploy it in small TRANCHES into intramonth RSI2
dips, HELD to the next monthly rebalance (per the hold-period finding: REV wants
~21d, don't flip). Reserve idle-cash earns BIL.

  base_full     : 100% monthly core, no reserve                 [current book]
  reserve_cash  : 80% core + 20% reserve parked in BIL          [drag baseline]
  reserve_dca   : 80% core + 20% deployed intramonth into RSI2 dips (5% tranches,
                  15% per-name cap, held to month-end)

If reserve_dca beats reserve_cash AND is competitive with base_full (esp. lower
DD in chop), intramonth scale-in adds value. Full 2007-26 + chop windows.
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

DATA_START, TRADE_START, END = "2005-06-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY, CAP, COST = 20, 0.40, 0.35, 15.0, 0.15, 5.0
RESERVE, TRANCHE = 0.20, 0.05
WINDOWS = [("2011 chop", "2011-05-01", "2011-12-31"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END)]


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


def stats(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return np.nan, np.nan, np.nan
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys()); dl = tickers + (["BIL"] if "BIL" not in tickers else [])
    print(f"Downloading {len(dl)} tickers (+BIL) ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values if "BIL" in close.columns else np.zeros(n)
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def core_route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return 0.
        a = pv[c][i] > m200[c][i]
        if e >= ER_HI and a and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.)
        if e <= ER_LO and a and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return 0.

    def rev_signal(c, i):
        return (fin(ER[c][i]) and ER[c][i] <= ER_LO and fin(pv[c][i]) and pv[c][i] > m200[c][i]
                and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY)

    def sim(reserve, dca):
        core = {c: 0. for c in present}; add = {c: 0. for c in present}
        prev = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                sel = {c: core_route(c, j) for c in present}; sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
                tot = sum(sel.values())
                tg = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
                s2 = sum(tg.values()); tg = {c: v / s2 * (1 - reserve) for c, v in tg.items()} if s2 > 0 else {}
                core = {c: tg.get(c, 0.) for c in present}; add = {c: 0. for c in present}
            elif dca:
                deployed = sum(add.values())
                for c in present:
                    if deployed + TRANCHE > reserve + 1e-9:
                        break
                    if rev_signal(c, j) and (core[c] + add[c]) + TRANCHE <= CAP + 1e-9 and fin(pv[c][j]):
                        add[c] += TRANCHE; deployed += TRANCHE
            cur = {c: core[c] + add[c] for c in present}
            turn[j + 1] += sum(abs(cur[c] - prev[c]) for c in present); prev = cur
            s = 0.
            for c in present:
                if cur[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += cur[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            cash = 1.0 - sum(cur.values())
            ret[j + 1] = s + cash * bil[j + 1]
        sc = slice(si + 1, n); return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    configs = [("base_full", 0.0, False), ("reserve_cash", RESERVE, False), ("reserve_dca", RESERVE, True)]
    results = {name: sim(res, dca) for name, res, dca in configs}
    spy = close["SPY"].pct_change().values[si + 1:]
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for name, _, _ in configs:
            r, dts = results[name]; sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {name:14s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, cg, dd = stats(spy, results["base_full"][1], lo, hi)
        print(f"  {'SPY':14s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
