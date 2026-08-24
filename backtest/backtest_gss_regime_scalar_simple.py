"""
Find a regime_scalar formula from EASY, ROBUST yfinance signals only - the
prior attempt mixed FX/yield tickers (^TNX/^IRX/JPY=X/^MOVE) into one
download and it corrupted calendar alignment across the whole frame (NaN-
poisoned every 200d rolling window - SPY>200DMA came back 0% pass, which is
impossible). Dropping those entirely. Uses only: SPY (already in the
universe) and ^VIX (one clean extra ticker, works fine combined - verified).

Candidates, wired as Lever A (regime_scalar multiplies MOM weight only,
before cap/normalize - matches the live asymmetry, REV untouched):
  none            - scalar=1.0 always (baseline / no throttle)
  trend_only      - 1.0 if SPY>200DMA else 0.25   (needs only SPY)
  vix_level       - 1.0 if VIX<20 else 0.4         (needs only VIX)
  trend_and_vix   - both pass->1.0, one->0.55, neither->0.15
  vol_target      - SPY's OWN realized vol vs its trailing median, scalar =
                     clip(median_vol/realized_vol, 0.25, 1.0) - needs ZERO
                     extra tickers, pure price-derived (simplest possible)
  continuous_vix  - smooth: clip(1-(VIX-15)/25,0.2,1) * (1.0 if trend else 0.5)
                     - avoids hard step-function cliffs (buffered-gate lesson)

Shipped config: adaptive router, ToM rebalance, equity-only buffered gate,
inverse-vol sizing, MOM cap 15%/REV cap 5%, BIL on cash, net@5bps, 2007-26.
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
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("2022 bear", "2022-01-01", "2022-12-31"),
           ("FULL 2007-26", TRADE_START, END), ("DEV 2020-26", "2020-01-01", END)]


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
    tickers = list(TREND_ASSETS.keys())
    dl = tickers + ["BIL", "^VIX"]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    vix = close["^VIX"].reindex(idx).ffill().values
    spy_rv = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    spy_rv_median = pd.Series(spy_rv).rolling(252, min_periods=60).median().values
    trend_pass_frac = float(np.nanmean(spy > spy200))
    print(f"Universe: {len(present)}   sanity: SPY>200DMA pass rate = {trend_pass_frac:.1%} (expect ~70-80%)\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 260)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def scalar_none(j):
        return 1.0

    def scalar_trend_only(j):
        if not fin(spy200[j]):
            return 1.0
        return 1.0 if spy[j] > spy200[j] else 0.25

    def scalar_vix_level(j):
        if not fin(vix[j]):
            return 1.0
        return 1.0 if vix[j] < 20 else 0.4

    def scalar_trend_and_vix(j):
        if not fin(spy200[j]) or not fin(vix[j]):
            return 1.0
        p = int(spy[j] > spy200[j]) + int(vix[j] < 20)
        return {2: 1.0, 1: 0.55, 0: 0.15}[p]

    def scalar_vol_target(j):
        rv, med = spy_rv[j], spy_rv_median[j]
        if not fin(rv) or not fin(med) or rv <= 0:
            return 1.0
        return float(np.clip(med / rv, 0.25, 1.0))

    def scalar_continuous_vix(j):
        if not fin(vix[j]) or not fin(spy200[j]):
            return 1.0
        vix_scalar = float(np.clip(1.0 - (vix[j] - 15.0) / 25.0, 0.2, 1.0))
        trend_scalar = 1.0 if spy[j] > spy200[j] else 0.5
        return vix_scalar * trend_scalar

    formulas = {"none": scalar_none, "trend_only": scalar_trend_only, "vix_level": scalar_vix_level,
                "trend_and_vix": scalar_trend_and_vix, "vol_target": scalar_vol_target,
                "continuous_vix": scalar_continuous_vix}
    # diagnostic: does exempting DEFENSIVE_ASSETS (matching the buffered gate's
    # own exemption logic) fix the crisis conflict?
    exempt_variants = {"vix_level_defexempt": scalar_vix_level,
                        "trend_and_vix_defexempt": scalar_trend_and_vix,
                        "vol_target_defexempt": scalar_vol_target}

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

    def sim(scalar_fn, exempt_defensive=False):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                rs = scalar_fn(j)
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
                # Lever A: regime_scalar throttles MOM EXPOSURE post-normalization
                # (matches live: shrinks the upper bound, so freed capital goes
                # to cash, NOT reshuffled to other names via renormalization).
                # exempt_defensive: don't throttle GLD/SLV/DBC/TLT MOM trades -
                # matches the buffered gate's own exemption; tests whether the
                # blanket throttle was hurting by shrinking crisis hedges too.
                def scale(c, v):
                    if kind[c] != "MOM":
                        return v
                    if exempt_defensive and c in DEFENSIVE_ASSETS:
                        return v
                    return v * rs
                tg = {c: scale(c, v) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    results = {name: sim(fn) for name, fn in formulas.items()}
    results.update({name: sim(fn, exempt_defensive=True) for name, fn in exempt_variants.items()})
    all_names = list(formulas) + list(exempt_variants)
    spyr = close["SPY"].pct_change().values[si + 1:]
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for name in all_names:
            r, dts = results[name]; sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {name:24s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, cg, dd = stats(spyr, results["none"][1], lo, hi)
        print(f"  {'SPY':24s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
