"""
Should the optimizer maximize SHARPE (Markowitz tangency) instead of the
current mu-free inverse-vol / risk-budget objective? Max-Sharpe needs per-asset
expected returns mu - which this session showed we can't estimate (no per-asset
alpha, slope*r2 IC~0). Test whether a return-forecast objective beats the
mu-free rules OOS, on the SAME selected (routed) names each rebalance.

Weighting schemes (long-only, per-name capped 7.5/5%, then vol-target scalar):
  inverse_vol   - w ~ 1/vol         (current)
  equal_weight  - w = 1/n           (mu-free, simplest)
  min_var       - min w'Cov w       (mu-free, uses covariance)
  maxsharpe_mom - max (w'mu)/sqrt(w'Cov w), mu = trailing 63d ann return
  maxsharpe_sr  - same, mu = slope*r2 (the trend signal as expected return)
Cov/mu from trailing 63d. 52-name book, ToM, gate, BIL cash, 2011-26 + crises.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

DATA_START, TRADE_START, END = "2009-06-01", "2011-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
MOM_CAP, REV_CAP, COST, BUF = 0.075, 0.05, 5.0, 0.03
BASE = list(TREND_ASSETS.keys())
SCHEMES = ["inverse_vol", "equal_weight", "min_var", "maxsharpe_mom", "maxsharpe_sr"]


def roll_sr(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
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


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return (np.nan,) * 4
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    dn_ = d[d < 0]
    so = (d.mean() * 252) / (np.sqrt((dn_ ** 2).mean()) * np.sqrt(252)) if len(dn_) > 4 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, so, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def solve(scheme, names, caps, invvol, mu, cov):
    n = len(names)
    if n == 1:
        return np.array([min(1.0, caps[0])])
    if scheme == "inverse_vol":
        w = invvol / invvol.sum()
    elif scheme == "equal_weight":
        w = np.full(n, 1.0 / n)
    else:
        bounds = [(0.0, c) for c in caps]
        x0 = np.full(n, 1.0 / n)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if scheme == "min_var":
            obj = lambda w: float(w @ cov @ w)
        else:
            m = mu.copy()
            def obj(w, m=m):
                var = float(w @ cov @ w)
                return -(w @ m) / np.sqrt(var + 1e-12)
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-9})
        w = res.x if res.success else x0
    w = np.minimum(w, caps)
    s = w.sum()
    return w / s if s > 0 else w


def main():
    t0 = time.time()
    dl = sorted(set(BASE + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in BASE if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    spy = close["SPY"].reindex(idx).values; spy200 = close["SPY"].rolling(200).mean().reindex(idx).values
    rets_np = close[present].pct_change().reindex(idx).values
    colidx = {c: k for k, c in enumerate(present)}
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].reindex(idx).values
        m50[c] = close[c].rolling(50).mean().reindex(idx).values
        m200[c] = close[c].rolling(200).mean().reindex(idx).values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().reindex(idx).values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}
    rv21 = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    rv_med = pd.Series(rv21).rolling(252, min_periods=60).median().values

    def scalar(j):
        if not fin(rv21[j]) or not fin(rv_med[j]) or rv21[j] <= 0:
            return 1.0
        return float(np.clip(rv_med[j] / rv21[j], 0.25, 1.0))

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

    def sim(scheme):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                rs = scalar(j); names = []; kinds = []
                for c in present:
                    if riskoff and c not in DEFENSIVE_ASSETS:
                        continue
                    sg = route(c, j)
                    if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                        names.append(c); kinds.append(sg)
                tg = {}
                if names:
                    caps = np.array([REV_CAP if k == "REV" else MOM_CAP for k in kinds])
                    invvol = np.array([1.0 / VOL[c][j] for c in names])
                    win_ = rets_np[j - 62:j + 1][:, [colidx[c] for c in names]]
                    cov = np.cov(np.nan_to_num(win_).T) if len(names) > 1 else np.array([[1.0]])
                    mu = np.nanmean(win_, axis=0) * 252
                    sr = np.array([max(SL[c][j] * R2[c][j], 0.0) for c in names])
                    mu_use = mu if scheme == "maxsharpe_mom" else sr
                    wv = solve(scheme, names, caps, invvol, mu_use, np.atleast_2d(cov))
                    # apply MOM vol-target scalar, then renormalize under caps
                    for i2, c in enumerate(names):
                        tg[c] = wv[i2] * (rs if kinds[i2] == "MOM" else 1.0)
                    ssum = sum(tg.values())
                    if ssum > 0:
                        tg = {c: min(v / ssum, caps[names.index(c)]) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    results = {s: sim(s) for s in SCHEMES}
    for lab, lo, hi in [("2020 COVID", "2020-02-01", "2020-04-30"),
                        ("2022 bear", "2022-01-01", "2022-12-31"),
                        ("FULL 2011-26", TRADE_START, END), ("DEV 2020-26", "2020-01-01", END)]:
        print(f"== {lab} ==   {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s}")
        for s in SCHEMES:
            r, dts = results[s]; sh, so, cg, dd = perf(r, dts, lo, hi)
            print(f"  {s:14s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
