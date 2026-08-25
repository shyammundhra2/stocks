"""
Should RSI2 mean-reversion be RESERVED for ETFs OUTSIDE the 3 trend-map curves?
The map draws iso-conviction hyperbolas |slope|.R2 = k at k={0.18,0.36,0.6}*xabs,
where slope = per-day-log-return*2000 (window=20, scale=20) and xabs=max(|slope|,3)
across the day's names. "Outside the 3 curves" = |slope|.R2 < 0.18*xabs: the
low-conviction, directionless names near the origin. Idea: only FADE names that
are genuinely trendless - don't mean-revert a name that's still in a clean trend.

Variants (MOM sleeve + top-8 blend cut unchanged; only REV gating changes):
  base        REV = ER<=0.35 & >200 & RSI2<15                 (current)
  and_curve   REV = base AND outside-curves (|slope|.R2<0.18*xabs)   (stricter)
  curve_only  REV = outside-curves & >200 & RSI2<15  (ER gate REPLACED by curve)

Report daily churn (total + REV-only), avg REV names/rebal, monthly turnover,
and Sharpe/CAGR/maxDD. A change earns its keep only if it cuts churn WITHOUT
giving up Sharpe/CAGR or deepening maxDD.
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

UNIV = list(TREND_ASSETS)
DATA_START, END = "2010-06-01", "2026-08-21"
COST, CAP, DEPLOY, TOPK = 5.0, 0.075, 0.65, 8
ER_HI, ER_LO, RSI_BUY = 0.40, 0.35, 15.0
MAP_WIN, MAP_MULT, CURVE_F = 20, 2000.0, 0.18   # map slope scale & innermost curve


def roll_sl_r2(p, win):
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


def rsi(p, k=2):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L=20):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return (np.nan,) * 3
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(UNIV + ["BIL"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]

    pv = {}; m50 = {}; m200 = {}; SL63 = {}; MSL = {}; MR2 = {}; HI = {}; RS = {}; ER = {}; VOL = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s63, r63 = roll_sl_r2(px.values, 63); SL63[c] = s63 * r63          # blend-cut slope*r2
        s20, r20 = roll_sl_r2(px.values, MAP_WIN)
        MSL[c] = s20 * MAP_MULT; MR2[c] = r20                              # MAP slope & r2
        HI[c] = pv[c] / px.rolling(252).max().values
        RS[c] = rsi(px.values); ER[c] = er(px.values)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)

    # daily cross-sectional xabs = max(|map slope|, 3) and curve-1 level
    MSL_df = pd.DataFrame({c: np.abs(MSL[c]) for c in present})
    XABS = np.maximum(MSL_df.max(axis=1).values, 3.0)
    K1 = CURVE_F * XABS                                                    # innermost curve
    j_index = {d: i for i, d in enumerate(range(n))}

    slxr = pd.DataFrame({c: SL63[c] for c in present}).rank(axis=1, pct=True)
    hir = pd.DataFrame({c: HI[c] for c in present}).rank(axis=1, pct=True)
    BL = {c: ((slxr[c] + hir[c]) / 2).values for c in present}
    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def outside_curves(c, j):
        conv = abs(MSL[c][j]) * MR2[c][j]
        return fin(conv) and conv < K1[j]

    def route(c, j, mode):
        e = ER[c][j]
        if not (fin(e) and fin(pv[c][j]) and fin(m200[c][j])) or pv[c][j] <= m200[c][j]:
            return None
        if e >= ER_HI and fin(m50[c][j]) and pv[c][j] > m50[c][j] and fin(SL63[c][j]) and SL63[c][j] > 0:
            return "MOM"
        rev_ok = fin(RS[c][j]) and RS[c][j] < RSI_BUY
        if mode == "base":
            if e <= ER_LO and rev_ok:
                return "REV"
        elif mode == "and_curve":
            if e <= ER_LO and rev_ok and outside_curves(c, j):
                return "REV"
        elif mode == "curve_only":
            if rev_ok and outside_curves(c, j):
                return "REV"
        return None

    def daily_churn(mode):
        prev = None; prevr = None; tot = 0; totr = 0; days = 0
        for j in range(si, n):
            cur = {c for c in present if route(c, j, mode)}
            curr = {c for c in present if route(c, j, mode) == "REV"}
            if prev is not None:
                tot += len(cur ^ prev); totr += len(curr ^ prevr); days += 1
            prev = cur; prevr = curr
        return tot / days, totr / days

    def sim(mode):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); revn = []
        for j in range(si, n - 1):
            if j in rd:
                mom = [c for c in present if route(c, j, mode) == "MOM" and fin(VOL[c][j]) and VOL[c][j] > 0]
                rev = [c for c in present if route(c, j, mode) == "REV" and fin(VOL[c][j]) and VOL[c][j] > 0]
                revn.append(len(rev))
                if len(mom) > TOPK:
                    mom = sorted(mom, key=lambda c: (BL[c][j] if fin(BL[c][j]) else -1), reverse=True)[:TOPK]
                sel = mom + rev; conv = {c: 1.0 / VOL[c][j] for c in sel}; tot = sum(conv.values())
                if tot > 0:
                    rv = {c: min(conv[c] / tot, CAP) for c in sel}; g = sum(rv.values())
                    if g > DEPLOY and g > 0:
                        rv = {c: v * (DEPLOY / g) for c, v in rv.items()}
                    nw = {c: rv.get(c, 0.0) for c in present}
                else:
                    nw = {c: 0. for c in present}
                turn[j + 1] += sum(abs(nw[c] - w[c]) for c in present); w = nw
            s = sum(w[c] * (pv[c][j + 1] / pv[c][j] - 1.) for c in present
                    if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]))
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        yrs = (idx[n - 1] - idx[si]).days / 365.25
        return ret[sc], idx[sc].values, turn[sc].sum() / 2 / yrs, np.mean(revn)

    print(f"\nREV gate variants (MOM + top-8 blend cut fixed). net@{COST:.0f}bps")
    print("             dChurn dREVchurn avgREV  turn  | Sh 11-19/20-26  CAGR/maxDD   2Y")
    for mode in ("base", "and_curve", "curve_only"):
        ch, chr_ = daily_churn(mode); r, d, tn, arv = sim(mode)
        s1, _, _ = perf(r, d, "2011-01-01", "2019-12-31")
        s2, _, _ = perf(r, d, "2020-01-01", END)
        _, cgf, ddf = perf(r, d, "2011-01-01", END)
        _, cg2, _ = perf(r, d, "2024-01-01", END)
        print(f"  {mode:11s}  {ch:>5.2f}  {chr_:>5.2f}   {arv:>4.1f}  {tn:>4.1f}x | "
              f"{s1:>4.2f}/{s2:>4.2f}  {cgf:>5.1%}/{ddf:>6.1%} {cg2:>5.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
