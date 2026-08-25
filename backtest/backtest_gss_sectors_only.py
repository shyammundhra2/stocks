"""
Apply the SHIPPED GSS adaptive strategy to sectors ONLY - as if the 11 sector
ETFs were the entire universe. This is NOT the cross-sectional ranking test
(that failed: sectors share beta, no dispersion). This is the full adaptive
LONG-ONLY book with trend/mean-reversion tension per sector:

  - ER router per sector: TREND(ER>=0.40)->momentum(slope*r2), CHOP(ER<=0.35)
    ->RSI2 oversold reversion, both gated >200DMA; else FLAT (to cash)
  - inverse-vol sizing, MOM cap 15% / REV cap 5%
  - buffered SPY-200DMA +/-3% risk gate (no defensives exist here -> risk-off
    goes fully to cash, the honest sector-book behavior)
  - vol-target regime scalar on MOM, ToM rebalance, BIL on idle cash, net@5bps

Compared to SPY and (context) the full 36-asset book's known ~0.75 Sharpe.
The question: does per-sector trend/revert timing + cash beat buy&hold, or
does the shared equity beta make it just an expensive SPY? 2007-26 + crises,
with Sortino.
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

SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE']
DATA_START, TRADE_START, END = "2004-01-01", "2007-01-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY = 20, 0.40, 0.35, 15.0
CAP, REV_CAP, COST, BUF = 0.15, 0.05, 5.0, 0.03
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("2020 COVID", "2020-02-01", "2020-04-30"),
           ("2022 bear", "2022-01-01", "2022-12-31"), ("FULL 2007-26", TRADE_START, END),
           ("DEV 2020-26", "2020-01-01", END)]


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


def stats(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return np.nan, np.nan, np.nan, np.nan
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    dn_ = d[d < 0]
    so = (d.mean() * 252) / (np.sqrt((dn_ ** 2).mean()) * np.sqrt(252)) if len(dn_) > 4 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, so, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = SECTORS + ["SPY", "BIL"]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in SECTORS if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).values
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Sectors available: {len(present)}  ({', '.join(present)})\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    # vol-target regime scalar from SPY (shipped Lever A)
    spy_rv = close["SPY"].pct_change().rolling(21).std().values * np.sqrt(252)
    spy_rv_med = pd.Series(spy_rv).rolling(252, min_periods=60).median().values

    def scalar(j):
        rv, md = spy_rv[j], spy_rv_med[j]
        if not fin(rv) or not fin(md) or rv <= 0:
            return 1.0
        return float(np.clip(md / rv, 0.25, 1.0))

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

    def sim(use_gate=True, use_scalar=True):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        deployed = []
        for j in range(si, n - 1):
            if j in rd:
                if use_gate and fin(spy200[j]):
                    th = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < th
                rs = scalar(j) if use_scalar else 1.0
                sel = {}; kind = {}
                if not (use_gate and riskoff):
                    for c in present:
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
                    tg = {c: (v * rs if kind[c] == "MOM" else v) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            deployed.append(sum(w.values()))
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, np.mean(deployed)

    configs = [("sectors adaptive", True, True), ("sectors no-gate", False, True),
               ("sectors no-scalar", True, False)]
    results = {nm: sim(g, s) for nm, g, s in configs}
    spyr = close["SPY"].pct_change().values[si + 1:]

    for nm, _, _ in configs:
        print(f"  {nm}: avg deployment {results[nm][2]:.0%}")
    print()
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'maxDD':>7s}")
        for nm, _, _ in configs:
            r, dts, _ = results[nm]; sh, so, cg, dd = stats(r, dts, lo, hi)
            print(f"  {nm:18s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}")
        sh, so, cg, dd = stats(spyr, results["sectors adaptive"][1], lo, hi)
        print(f"  {'SPY buy&hold':18s} {sh:>7.2f} {so:>8.2f} {cg:>7.1%} {dd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
