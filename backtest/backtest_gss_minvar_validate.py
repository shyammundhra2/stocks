"""
Validate min-variance vs inverse-vol before wiring it into the live objective.
Three questions:
  1. OUT-OF-WINDOW: does min-var beat inverse-vol across sub-periods AND in the
     2007-2010 window (incl 2008), not just 2011-26?
  2. COVARIANCE ROBUSTNESS: min-var leans on the covariance estimate. Test
     sample cov (63d, 126d) and Ledoit-Wolf shrinkage - if the ranking flips
     with the estimator, it's fragile.
  3. CONCENTRATION: does min-var pile into low-vol names? Report effective N
     (1/sum w^2), avg max weight, and share going to DEFENSIVE_ASSETS.

Base-36 universe (full 2007-26 incl 2008; the +16 breadth names post-date 2010).
Shipped config: router, ToM, gate, 7.5/5% caps, vol-target scalar, BIL cash.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
try:
    from sklearn.covariance import ledoit_wolf
except Exception:
    ledoit_wolf = None

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS

# base-36: exclude the breadth names added post-2010 so 2007-26 has full history
NEW_NAMES = {"XBI","XHB","XRT","KRE","XOP","OIH","IYT","GDX","KIE","FDN","IHI",
             "SIL","REMX","LIT","GDXJ","GNR"}
BASE = [t for t in TREND_ASSETS if t not in NEW_NAMES]
DATA_START, TRADE_START, END = "2004-06-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
MOM_CAP, REV_CAP, COST, BUF = 0.075, 0.05, 5.0, 0.03


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
        return (np.nan,) * 3
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def minvar_w(cov, caps):
    n = cov.shape[0]
    if n == 1:
        return np.array([min(1.0, caps[0])])
    res = minimize(lambda w: float(w @ cov @ w), np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, c) for c in caps],
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 200, "ftol": 1e-10})
    w = res.x if res.success else np.full(n, 1.0 / n)
    w = np.minimum(w, caps); s = w.sum()
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
    print(f"Base universe: {len(present)}\n")

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

    def sim(scheme, cov_win=63, shrink=False):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        effN = []; maxw = []; defshare = []
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
                    if scheme == "inverse_vol":
                        iv = np.array([1.0 / VOL[c][j] for c in names]); wv = iv / iv.sum()
                        wv = np.minimum(wv, caps); wv = wv / wv.sum() if wv.sum() > 0 else wv
                    else:  # min_var
                        win_ = np.nan_to_num(rets_np[j - cov_win + 1:j + 1][:, [colidx[c] for c in names]])
                        if len(names) > 1:
                            cov = ledoit_wolf(win_)[0] if (shrink and ledoit_wolf) else np.cov(win_.T)
                        else:
                            cov = np.array([[1.0]])
                        wv = minvar_w(np.atleast_2d(cov), caps)
                    for i2, c in enumerate(names):
                        tg[c] = wv[i2] * (rs if kinds[i2] == "MOM" else 1.0)
                    ss = sum(tg.values())
                    if ss > 0:
                        tg = {c: min(v / ss, caps[names.index(c)]) for c, v in tg.items()}
                    wa = np.array(list(tg.values()))
                    if wa.sum() > 0:
                        p = wa / wa.sum()
                        effN.append(1.0 / np.sum(p ** 2)); maxw.append(p.max())
                        defshare.append(sum(v for c, v in tg.items() if c in DEFENSIVE_ASSETS) / wa.sum())
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return (ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values,
                np.mean(effN) if effN else 0, np.mean(maxw) if maxw else 0,
                np.mean(defshare) if defshare else 0)

    configs = [("inverse_vol", dict(scheme="inverse_vol")),
               ("min_var 63d", dict(scheme="min_var", cov_win=63)),
               ("min_var 126d", dict(scheme="min_var", cov_win=126)),
               ("min_var LW-shrink", dict(scheme="min_var", cov_win=63, shrink=True))]
    results = {name: sim(**kw) for name, kw in configs}

    print("CONCENTRATION (avg over rebalances):")
    for name, _ in configs:
        _, _, en, mw, dfs = results[name]
        print(f"  {name:18s} eff.N {en:4.1f}   max-wt {mw:4.0%}   defensive-share {dfs:4.0%}")
    print()
    windows = [("2007-2010 (OOS incl 2008)", "2007-01-01", "2010-12-31"),
               ("2011-2015", "2011-01-01", "2015-12-31"),
               ("2016-2020", "2016-01-01", "2020-12-31"),
               ("2021-2026", "2021-01-01", END),
               ("2008 GFC", "2007-10-01", "2009-06-30"),
               ("FULL 2007-26", TRADE_START, END)]
    for lab, lo, hi in windows:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for name, _ in configs:
            r, dts, _, _, _ = results[name]; sh, cg, dd = perf(r, dts, lo, hi)
            print(f"  {name:18s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
