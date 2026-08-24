"""
Find a regime_scalar formula that actually improves the shipped GSS book,
not just one with plausible-sounding conditions. Replicates all 6 live
conditions (trend/vix/breadth/credit/curve/carry) historically, then tests
candidate scalar formulas by wiring the scalar into the ACTUAL adaptive
router backtest as Lever A (regime_scalar multiplies MOM weight before the
cap/normalize step - REV stays untouched, matching the live asymmetry in
_kelly_covariance_optimizer). This is the real test: does a different
formula move Sharpe/CAGR/maxDD on the shipped config, not just show a nicer
IC on SPY in isolation.

Candidates:
  none          - regime_scalar=1.0 always (no throttle) - baseline
  current       - live formula: 6 equal-weight conditions, lookup table
  trend_vix     - only the 2 conditions with a real protective hit-rate gap
                  (backtest_gss_regime_conditions_ic.py): trend + vix
  trend_only    - trend alone (SPY>200MA), simplest possible de-risk signal
  weighted      - all 6, but trend x2 vix x1.5 breadth/credit/curve/carry x1,
                  continuous (not stepped) scalar from the weighted fraction

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
    regime_tickers = ["RSP", "HYG", "IEF", "^VIX", "^MOVE", "^TNX", "^IRX", "JPY=X"]
    dl = tickers + ["BIL"] + [t for t in regime_tickers if t not in tickers]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Universe: {len(present)}\n")

    # --- replicate the 6 live regime conditions, historically ---
    rsp_spy = close["RSP"] / close["SPY"]
    hyg_ief = close["HYG"] / close["IEF"]
    trend_pass = (close["SPY"] > close["SPY"].rolling(200).mean()).reindex(idx).values
    vix_pass = ((close["^VIX"] < 20) & (close["^MOVE"] < 110)).reindex(idx).values
    breadth_pass = (rsp_spy > rsp_spy.rolling(50).mean()).reindex(idx).values
    credit_pass = (hyg_ief > hyg_ief.rolling(50).mean()).reindex(idx).values
    curve_pass = ((close["^TNX"] - close["^IRX"]) > 0).reindex(idx).values
    jpy_ret = close["JPY=X"].pct_change()
    jpy_vol = (jpy_ret.rolling(20).std() * np.sqrt(252)).reindex(idx)
    jpy_ma50 = close["JPY=X"].rolling(50).mean().reindex(idx)
    carry_pass = ((close["JPY=X"].reindex(idx) > jpy_ma50) & (jpy_vol < 0.15)).values

    conds = {"trend": trend_pass, "vix": vix_pass, "breadth": breadth_pass,
             "credit": credit_pass, "curve": curve_pass, "carry": carry_pass}
    for name, c in conds.items():
        print(f"  {name:8s} valid={np.isfinite(c.astype(float)).sum() if c.dtype != bool else (~pd.isna(pd.Series(c))).sum()}  "
              f"pass%={np.nanmean(c.astype(float))*100:.0f}%")

    def scalar_current(j):
        vals = [conds[k][j] for k in ["trend", "vix", "breadth", "credit", "curve", "carry"]]
        if any(pd.isna(v) for v in vals):
            return 1.0
        p = sum(bool(v) for v in vals)
        return {6: 1.0, 5: 0.85, 4: 0.70, 3: 0.50, 2: 0.30, 1: 0.15, 0: 0.0}[p]

    def scalar_trend_vix(j):
        t, v = conds["trend"][j], conds["vix"][j]
        if pd.isna(t) or pd.isna(v):
            return 1.0
        p = int(bool(t)) + int(bool(v))
        return {2: 1.0, 1: 0.55, 0: 0.15}[p]

    def scalar_trend_only(j):
        t = conds["trend"][j]
        if pd.isna(t):
            return 1.0
        return 1.0 if bool(t) else 0.25

    def scalar_weighted(j):
        w = {"trend": 2.0, "vix": 1.5, "breadth": 1.0, "credit": 1.0, "curve": 1.0, "carry": 1.0}
        tot = 0.0; got = 0.0
        for k, wt in w.items():
            v = conds[k][j]
            if pd.isna(v):
                continue
            tot += wt
            got += wt if bool(v) else 0.0
        return (got / tot) if tot > 0 else 1.0

    def scalar_none(j):
        return 1.0

    formulas = {"none": scalar_none, "current": scalar_current, "trend_vix": scalar_trend_vix,
                "trend_only": scalar_trend_only, "weighted": scalar_weighted}

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 260)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

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

    def sim(scalar_fn):
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
                        base = 1.0 / VOL[c][j]
                        sel[c] = base * (rs if sg == "MOM" else 1.0)   # Lever A: MOM only
                        kind[c] = sg
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
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    results = {name: sim(fn) for name, fn in formulas.items()}
    spyr = close["SPY"].pct_change().values[si + 1:]
    print()
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for name in formulas:
            r, dts = results[name]; sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {name:12s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, cg, dd = stats(spyr, results["none"][1], lo, hi)
        print(f"  {'SPY':12s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
