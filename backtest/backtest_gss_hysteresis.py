"""
Does adding HYSTERESIS to the ER router quiet the day-to-day flicker WITHOUT
hurting returns? The flicker source is the memoryless router: a name is REV only
while RSI2<15 (a 2-period RSI is a fast oscillator - dips under 15 and pops back
in days), and MOM flips at the ER=0.40 boundary. Hysteresis = a state machine
that ENTERS on the tight threshold but STAYS until a wider EXIT is crossed.

State machine (walked daily; monthly rebalance reads state at month-end):
  base   memoryless (current): MOM iff ER>=.40 & >200 & >50 & slope>0;
                               REV iff ER<=.35 & >200 & RSI2<15
  rsi    REV hold-until-recovered: enter RSI2<15, exit RSI2>50 (classic bounce
                               hold) or <200DMA; MOM unchanged
  both   rsi + MOM band: enter MOM ER>=.40, stay until ER<.30 / slope<=0 / <200

Everything else identical (inverse-vol, top-8 blend cut on MOM, 7.5% cap, 65%
deploy, ToM rebal, net@5bps). A variant ships only if it cuts daily churn AND
holds Sharpe/CAGR/maxDD (and ideally lowers monthly turnover too).
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
COST, SL_WIN, CAP, DEPLOY, TOPK = 5.0, 63, 0.075, 0.65, 8
ER_HI, ER_LO, ER_MOM_EXIT = 0.40, 0.35, 0.30
RSI_BUY, RSI_EXIT = 15.0, 50.0


def slr2(p, w=SL_WIN):
    n = len(p); s = np.full(n, np.nan); r = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(p > 0, p, np.nan))
    if n < w or sliding_window_view is None:
        return s, r
    W = sliding_window_view(lp, w); x = np.arange(w); xc = x - x.mean(); dn = float(xc @ xc)
    ym = W.mean(1); sl = (W - ym[:, None]) @ xc / dn
    pr = sl[:, None] * x[None, :] + (ym - sl * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pr) ** 2).sum(1)
    s[w - 1:] = sl; r[w - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return s, r


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
    pv = {}; m50 = {}; m200 = {}; SL = {}; HI = {}; RS = {}; ER = {}; VOL = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s, r = slr2(px.values); SL[c] = s; HI[c] = pv[c] / px.rolling(252).max().values
        RS[c] = rsi(px.values); ER[c] = er(px.values)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)
    # blend rank uses slope*r2 and hi52
    slxr = pd.DataFrame({c: np.where(fin(SL[c]), SL[c], np.nan) for c in present}).rank(axis=1, pct=True)
    hir = pd.DataFrame({c: HI[c] for c in present}).rank(axis=1, pct=True)
    BL = {c: ((slxr[c] + hir[c]) / 2).values for c in present}
    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def build_state(mode):
        # returns STATE[c] = array of "MOM"/"REV"/"FLAT" per day, with hysteresis
        ST = {}
        for c in present:
            p = pv[c]; m2 = m200[c]; m5 = m50[c]; e = ER[c]; sl = SL[c]; rs = RS[c]
            st = "FLAT"; out = np.empty(n, dtype=object)
            for j in range(n):
                ok = fin(p[j]) and fin(m2[j]) and fin(e[j])
                if not ok:
                    st = "FLAT"; out[j] = st; continue
                above = p[j] > m2[j]
                if mode == "base":
                    if above and e[j] >= ER_HI and fin(m5[j]) and p[j] > m5[j] and fin(sl[j]) and sl[j] > 0:
                        st = "MOM"
                    elif above and e[j] <= ER_LO and fin(rs[j]) and rs[j] < RSI_BUY:
                        st = "REV"
                    else:
                        st = "FLAT"
                    out[j] = st; continue
                # hysteresis modes
                if st == "MOM":
                    if mode == "both":
                        if not above or (fin(sl[j]) and sl[j] <= 0) or e[j] < ER_MOM_EXIT:
                            st = "FLAT"
                    else:  # rsi mode: MOM behaves like base
                        if not (above and e[j] >= ER_HI and fin(m5[j]) and p[j] > m5[j] and fin(sl[j]) and sl[j] > 0):
                            st = "FLAT"
                elif st == "REV":
                    if not above or (fin(rs[j]) and rs[j] > RSI_EXIT):
                        st = "FLAT"
                if st == "FLAT":
                    if above and e[j] >= ER_HI and fin(m5[j]) and p[j] > m5[j] and fin(sl[j]) and sl[j] > 0:
                        st = "MOM"
                    elif above and e[j] <= ER_LO and fin(rs[j]) and rs[j] < RSI_BUY:
                        st = "REV"
                out[j] = st
            ST[c] = out
        return ST

    def daily_churn(ST):
        prev = None; tot = 0; days = 0
        for j in range(si, n):
            cur = {c for c in present if ST[c][j] in ("MOM", "REV")}
            if prev is not None:
                tot += len(cur ^ prev); days += 1
            prev = cur
        return tot / days

    def sim(ST):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                mom = [c for c in present if ST[c][j] == "MOM" and fin(VOL[c][j]) and VOL[c][j] > 0]
                rev = [c for c in present if ST[c][j] == "REV" and fin(VOL[c][j]) and VOL[c][j] > 0]
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
        return ret[sc], idx[sc].values, turn[sc].sum() / 2 / yrs

    print(f"\nDaily churn (names in/out per day) | Sharpe 11-19/20-26 | full CAGR/maxDD | 2Y CAGR | turnover")
    for mode in ("base", "rsi", "both"):
        ST = build_state(mode); ch = daily_churn(ST); r, d, tn = sim(ST)
        s1, _, _ = perf(r, d, "2011-01-01", "2019-12-31")
        s2, _, _ = perf(r, d, "2020-01-01", END)
        _, cgf, ddf = perf(r, d, "2011-01-01", END)
        _, cg2, _ = perf(r, d, "2024-01-01", END)
        print(f"  {mode:5s}  churn {ch:>5.2f}   {s1:>4.2f}/{s2:>4.2f}   {cgf:>5.1%}/{ddf:>6.1%}   2Y {cg2:>5.1%}   turn {tn:>4.1f}x")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
