"""
LIVE-faithful test of the hi52 blend. The horse-race used a top-K selector, but
the live book routes (ER + >200DMA) then sizes ALL selected names by INVERSE-VOL
conviction (_conviction_score = 1/vol63). So the real question is: does tilting
the inverse-vol conviction toward momentum-quality (slope*r2 + hi52 rank-blend)
beat PURE inverse-vol under the live mechanism?

Mechanism (proxy of live): ToM rebalance; MOM = ER>=0.40 & >200DMA & >50DMA &
slope>0; REV = ER<=0.35 & >200DMA & rsi2<15. Weight selected names by
conviction_i / sum(conviction), clip per-name at CAP, scale gross to DEPLOY,
rest to BIL. Net @5bps.

  base     conviction = 1/vol63                        (LIVE baseline)
  q25/50   conviction = (1/vol63) * qmult, where qmult maps the cross-sectional
           (slope*r2 rank + hi52 rank)/2 percentile into [1-a, 1+a], a=.25/.50

A tilt only ships if it beats base in BOTH 2011-19 and 2020-26 without wrecking
maxDD. (McLean-Pontiff: demand margin, assume forward decay.)
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

UNIV = list(TREND_ASSETS.keys())
DATA_START, END = "2010-06-01", "2026-08-21"
COST, SL_WIN = 5.0, 63
CAP, DEPLOY = 0.075, 0.65          # live per-name & max-deploy caps
ER_HI, ER_LO, RSI_BUY = 0.40, 0.35, 15.0


def roll_slope_r2(p, win):
    n = len(p); sl = np.full(n, np.nan); r2 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    if n < win or sliding_window_view is None:
        return sl, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); s = (W - ym[:, None]) @ xc / dn
    pr = s[:, None] * x[None, :] + (ym - s * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    sl[win - 1:] = s; r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
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
        return (np.nan, np.nan, np.nan)
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(UNIV + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values

    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]
    pv = {}; m50 = {}; m200 = {}; SLR2 = {}; HI52 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s, r = roll_slope_r2(px.values, SL_WIN); SLR2[c] = s * r
        HI52[c] = pv[c] / px.rolling(252).max().values
        RS2[c] = rsi(px.values, 2); ER[c] = er(px.values, 20)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)

    slr2_df = pd.DataFrame({c: SLR2[c] for c in present}).rank(axis=1, pct=True)
    hi52_df = pd.DataFrame({c: HI52[c] for c in present}).rank(axis=1, pct=True)
    QPCT = {c: ((slr2_df[c] + hi52_df[c]) / 2.0).values for c in present}  # in [0,1]

    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def route(c, j):
        e = ER[c][j]
        if not (fin(e) and fin(pv[c][j]) and fin(m200[c][j])):
            return None
        if pv[c][j] <= m200[c][j]:
            return None
        if e >= ER_HI and fin(m50[c][j]) and pv[c][j] > m50[c][j] and fin(SLR2[c][j]) and SLR2[c][j] > 0:
            return "MOM"
        if e <= ER_LO and fin(RS2[c][j]) and RS2[c][j] < RSI_BUY:
            return "REV"
        return None

    def sim(alpha):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                sel = [c for c in present if route(c, j) and fin(VOL[c][j]) and VOL[c][j] > 0]
                conv = {}
                for c in sel:
                    base = 1.0 / VOL[c][j]
                    qmult = 1.0 + alpha * (2.0 * QPCT[c][j] - 1.0) if (alpha > 0 and fin(QPCT[c][j])) else 1.0
                    conv[c] = base * qmult
                tot = sum(conv.values())
                if tot > 0:
                    raw = {c: conv[c] / tot for c in sel}
                    raw = {c: min(v, CAP) for c, v in raw.items()}           # per-name cap
                    g = sum(raw.values())
                    if g > DEPLOY and g > 0:
                        raw = {c: v * (DEPLOY / g) for c, v in raw.items()}  # deploy cap
                    neww = {c: raw.get(c, 0.0) for c in present}
                else:
                    neww = {c: 0. for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    variants = {"base (1/vol)": 0.0, "blend a=.25": 0.25, "blend a=.50": 0.50}
    print(f"\nLIVE-proxy: ER router, inverse-vol sizing, cap {CAP:.1%}, deploy {DEPLOY:.0%}, net@{COST:.0f}bps")
    print("Sharpe (2011-19 / 2020-26)  |  full CAGR  |  full maxDD\n")
    for lab, a in variants.items():
        r, d = sim(a)
        s1, _, _ = perf(r, d, "2011-01-01", "2019-12-31")
        s2, _, _ = perf(r, d, "2020-01-01", END)
        _, cg, dd = perf(r, d, "2011-01-01", END)
        print(f"  {lab:14s}  {s1:>5.2f} / {s2:>5.2f}   {cg:>6.1%}   {dd:>7.1%}")
    r, d = sim(0.0)
    spy = close["SPY"].pct_change().reindex(idx).values[si + 1:]
    s1, _, _ = perf(spy, d, "2011-01-01", "2019-12-31")
    s2, _, _ = perf(spy, d, "2020-01-01", END); _, cg, dd = perf(spy, d, "2011-01-01", END)
    print(f"  {'SPY':14s}  {s1:>5.2f} / {s2:>5.2f}   {cg:>6.1%}   {dd:>7.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
