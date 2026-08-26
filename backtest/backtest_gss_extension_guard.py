"""
Does an EXTENSION GUARD on the live ER-router MOM entry cut the chop whipsaw
(2013-19, the book's weak regime) without giving up the trend-regime edge? Guard
= skip a MOM entry when the name is overbought/extended (RSI14>70 or ATR14>2x its
126d avg) - Trendguard's conditions, which made it chop-robust.

Full live-proxy config: ER router + top-8 slope*r2/hi52 blend cut on MOM +
curve-gated REV (ER<=.35 & rsi2<15 & outside innermost iso-curve) + inverse-vol
sizing + 7.5%/5% caps + 65% deploy cap + buffered SPY-200DMA gate, net@5bps.
(VIX throttle omitted - it's a deploy scalar, orthogonal to the entry guard.)

  base        current MOM entry (no guard)
  guard_rsi   MOM also requires RSI14 <= 70
  guard_atr   MOM also requires ATR14 <= 2 * mean(ATR14,126)
  guard_both  MOM requires both
"""
import sys
import time

import numpy as np
import pandas as pd

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/tg_ohlc.parquet"
COST, CAP, DEPLOY, TOPK, BUF = 5.0, 0.075, 0.65, 8, 0.03
ER_HI, ER_LO, RSI_BUY, MAP_MULT = 0.40, 0.35, 15.0, 2000.0


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


def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L=20):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); d = np.asarray(r)[m]; d = d[np.isfinite(d)]
    if len(d) < 30 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(252)     # DAILY returns
    eq = np.cumprod(1 + d); return sh, eq[-1] ** (252 / len(d)) - 1, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    raw = pd.read_parquet(CACHE)
    close = raw["Close"]; high = raw["High"]; low = raw["Low"]
    spy = close["SPY"]; spy200 = spy.rolling(200).mean(); idx = close.index; n = len(idx); fin = np.isfinite
    names = [s for s in TREND_ASSETS if s in close.columns and close[s].notna().sum() > 400]

    off = False; gate = np.zeros(n, bool); sv, s2 = spy.values, spy200.values
    for i in range(n):
        if fin(s2[i]):
            if off and sv[i] > s2[i] * (1 + BUF):
                off = False
            elif (not off) and sv[i] < s2[i] * (1 - BUF):
                off = True
        gate[i] = off

    pv = {}; m50 = {}; m200 = {}; SL63 = {}; MSL = {}; MR2 = {}; HI = {}; RS2 = {}; RS14 = {}; ER = {}; VOL = {}; ATRx = {}
    for c in names:
        px = close[c]; hi = high[c]; lo = low[c]; pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s63, r63 = roll_sl_r2(px.values, 63); SL63[c] = s63 * r63
        s20, r20 = roll_sl_r2(px.values, 20); MSL[c] = s20 * MAP_MULT; MR2[c] = r20
        HI[c] = px.values / px.rolling(252).max().values
        RS2[c] = rsi(px.values, 2); RS14[c] = rsi(px.values, 14); ER[c] = er(px.values)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)
        tr = pd.concat([hi - lo, (hi - px.shift()).abs(), (lo - px.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        ATRx[c] = (atr / atr.rolling(126).mean()).values      # ATR vs its 126d avg
    XABS = np.maximum(pd.DataFrame({c: np.abs(MSL[c]) for c in names}).max(axis=1).values, 3.0)
    slxr = pd.DataFrame({c: SL63[c] for c in names}).rank(axis=1, pct=True)
    hir = pd.DataFrame({c: HI[c] for c in names}).rank(axis=1, pct=True)
    BL = {c: ((slxr[c] + hir[c]) / 2).values for c in names}
    si = int(np.searchsorted(idx.values, np.datetime64("2006-01-01")))
    ds = pd.Series(idx); rd = {g.iloc[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.iloc[-1] >= idx[si]}
    rd = {idx.get_loc(d) for d in rd}

    def conv(c, j):
        v = abs(MSL[c][j]) * MR2[c][j]
        return v if fin(v) else 0.0

    def route(c, j, guard):
        e = ER[c][j]
        if not (fin(e) and fin(pv[c][j]) and fin(m200[c][j])) or pv[c][j] <= m200[c][j]:
            return None
        if e >= ER_HI and fin(m50[c][j]) and pv[c][j] > m50[c][j] and fin(SL63[c][j]) and SL63[c][j] > 0:
            if guard in ("rsi", "both") and (not fin(RS14[c][j]) or RS14[c][j] > 70):
                return None
            if guard in ("atr", "both") and (not fin(ATRx[c][j]) or ATRx[c][j] > 2.0):
                return None
            return "MOM"
        if e <= ER_LO and fin(RS2[c][j]) and RS2[c][j] < RSI_BUY and conv(c, j) < 0.18 * XABS[j]:
            return "REV"
        return None

    def sim(guard):
        w = {c: 0. for c in names}; ret = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                goff = gate[j]
                mom = [] if goff else [c for c in names if route(c, j, guard) == "MOM" and fin(VOL[c][j]) and VOL[c][j] > 0]
                rev = [] if goff else [c for c in names if route(c, j, guard) == "REV" and fin(VOL[c][j]) and VOL[c][j] > 0]
                if len(mom) > TOPK:
                    mom = sorted(mom, key=lambda c: (BL[c][j] if fin(BL[c][j]) else -1), reverse=True)[:TOPK]
                sel = mom + rev; cv = {c: 1.0 / VOL[c][j] for c in sel}; tot = sum(cv.values())
                if tot > 0:
                    rv = {c: min(cv[c] / tot, CAP) for c in sel}; g = sum(rv.values())
                    if g > DEPLOY and g > 0:
                        rv = {c: v * (DEPLOY / g) for c, v in rv.items()}
                    w = {c: rv.get(c, 0.0) for c in names}
                else:
                    w = {c: 0. for c in names}
            s = sum(w[c] * (pv[c][j + 1] / pv[c][j] - 1.) for c in names
                    if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]))
            ret[j + 1] = s   # idle cash earns 0 (conservative; same across variants)
        sc = slice(si + 1, n)
        return ret[sc], idx[sc].values

    wins = [("2006-2012", "2006-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"), ("FULL", "2006-01-01", "2026-08-21")]
    print(f"\nExtension guard on live-proxy MOM entry, net@{COST:.0f}bps\n")
    R = {g: sim(g) for g in [None, "rsi", "atr", "both"]}
    lbl = {None: "base", "rsi": "guard_rsi", "atr": "guard_atr", "both": "guard_both"}
    for lab, lo, hi in wins:
        print(f"== {lab} ==          {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for g in [None, "rsi", "atr", "both"]:
            r, d = R[g]; sh, cg, dd = perf(r, d, lo, hi)
            star = " *" if g is None else "  "
            print(f"  {lbl[g]:11s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
