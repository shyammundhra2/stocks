"""
Exit-threshold sweep for the VIX-capitulation reserve (backtest_gss_capitulation.py's
"vixz" trigger - the one actually live in macro/indicators/signals.py's
get_vix_signal()). That backtest swept ENTRY triggers (dd15/dd20/vixz/rsi25) but
froze the EXIT at "SPY recovers to within 5% of its 252d high" without ever
testing alternatives - this sweeps the exit threshold itself, entry held fixed
at vixz, same data/cost/window conventions as the original.

DD_EXIT candidates: -0.03 (tight), -0.05 (current/frozen), -0.07, -0.10 (loose),
plus two ABOVE the 252d high - 0.0 (wait for an actual new high, not just "close
enough") and +0.02 (require a confirmed breakout past the old high, not just
touching it) - to check whether early-but-unconfirmed exits are giving back
real upside to whipsaws, or whether waiting for a genuine new high just means
sitting out most of the recovery.
Full 2007-26, net@5bps, vs reserve_cash baseline. Also reports whipsaw (re-entries
within 10 trading days of an exit - a false-recovery tell) since that's the real
risk on the tight end that Sharpe/CAGR alone can hide.
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
WIN, ER_HI, ER_LO, RSI_BUY, CAP, COST, RESERVE = 20, 0.40, 0.35, 15.0, 0.15, 5.0, 0.20
EXIT_CANDIDATES = [-0.03, -0.05, -0.07, -0.10, 0.0, 0.02]
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2020 COVID", "2020-02-01", "2020-12-31"),
           ("2022 bear", "2022-01-01", "2022-12-31"), ("FULL 2007-26", TRADE_START, END),
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
    spy = close["SPY"].values; spy_ret = close["SPY"].pct_change().values
    spy_hi = pd.Series(spy).rolling(252, min_periods=60).max().values
    spy_dd = spy / spy_hi - 1.0
    # Exit-side comparison uses the PRIOR day's trailing high (shifted), not
    # today's - rolling().max() is inclusive of the current row, so today's dd
    # vs today's own high can never exceed 0 by construction. Without the
    # shift, any exit_dd >= 0 candidate silently degenerates into "never
    # exits" (confirmed: ndep collapsed to 1 for the unshifted version).
    spy_hi_prior = pd.Series(spy).rolling(252, min_periods=60).max().shift(1).values
    spy_dd_exit = spy / spy_hi_prior - 1.0
    spy_rsi14 = rsi(close["SPY"].dropna().values, 14)
    rv = pd.Series(spy_ret).rolling(21).std()
    rvz = ((rv - rv.rolling(252).mean()) / rv.rolling(252).std()).values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 260)
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

    def vixz_trigger(j):
        return fin(rvz[j]) and rvz[j] > 2.0 and fin(spy_rsi14[j]) and spy_rsi14[j] < 30

    def sim(exit_dd):
        core = {c: 0. for c in present}; prev = {c: 0. for c in present}
        res_spy = 0.0; prev_res = 0.0; ret = np.zeros(n); turn = np.zeros(n)
        exit_days = []; ndep = 0
        for j in range(si, n - 1):
            if j in rd:
                sel = {c: core_route(c, j) for c in present}; sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
                tot = sum(sel.values())
                tg = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
                s2 = sum(tg.values()); tg = {c: v / s2 * (1 - RESERVE) for c, v in tg.items()} if s2 > 0 else {}
                core = {c: tg.get(c, 0.) for c in present}
            if res_spy == 0.0 and vixz_trigger(j):
                res_spy = RESERVE; ndep += 1
            elif res_spy > 0 and fin(spy_dd_exit[j]) and spy_dd_exit[j] > exit_dd:   # panic over
                res_spy = 0.0; exit_days.append(j)
            turn[j + 1] += sum(abs(core[c] - prev[c]) for c in present) + abs(res_spy - prev_res)
            prev = dict(core); prev_res = res_spy
            s = 0.
            for c in present:
                if core[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += core[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            if res_spy > 0 and fin(spy_ret[j + 1]):
                s += res_spy * spy_ret[j + 1]
            cash = 1.0 - sum(core.values()) - res_spy
            ret[j + 1] = s + cash * bil[j + 1]
        sc = slice(si + 1, n)
        # whipsaw: how many deploys re-triggered within 10 trading days of a prior exit
        whips = 0
        ex_sorted = sorted(exit_days)
        dep_days = []
        res_spy2 = 0.0
        for j in range(si, n - 1):
            if res_spy2 == 0.0 and vixz_trigger(j):
                dep_days.append(j); res_spy2 = RESERVE
            elif res_spy2 > 0 and fin(spy_dd_exit[j]) and spy_dd_exit[j] > exit_dd:
                res_spy2 = 0.0
        for d in dep_days:
            if any(0 < d - e <= 10 for e in ex_sorted):
                whips += 1
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, ndep, whips

    print("core 80% monthly adaptive + 20% reserve; entry fixed at vixz (z>2 AND RSI14<30);")
    print("sweeping the EXIT threshold only (SPY recovery vs its 252d high)\n")
    results = {ex: sim(ex) for ex in EXIT_CANDIDATES}
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'deploys':>8s} {'whipsaw':>8s}")
        for ex in EXIT_CANDIDATES:
            r, dts, ndep, whips = results[ex]
            sh, cg, dd = stats(r, dts, lo, hi)
            tag = f"exit{ex:+.0%}" + ("  <- current" if ex == -0.05 else "")
            print(f"  {tag:18s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {ndep:>8d} {whips:>8d}")
        r0, dts0, _, _ = results[-0.05]
        sh, cg, dd = stats(spy_ret[si + 1:], dts0, lo, hi)
        print(f"  {'SPY':18s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {'-':>8s} {'-':>8s}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
