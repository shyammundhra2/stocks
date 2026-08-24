"""
Is a tighter per-name cap on REV (mean-reversion) trades defensible? REV is the
junior sleeve (standalone Sharpe 0.78; its value = small scarce doses), yet
under inverse-vol conviction low-vol oversold names can take the biggest slots
(live book 2026-08: XLI+ITA both REV at 15% = 30% of the book on the weak
sleeve). Sweep a REV-specific cap on top of the shipped gate_eq_iv config:

  rev cap in {15% (=current), 10%, 5%, 3%}   (MOM cap stays 15%)

Full 2007-26, ToM rebalance, equity-only buffered gate, inverse-vol sizing,
net@5bps, BIL on idle cash, chop + crisis windows.
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
WIN, ER_HI, ER_LO, RSI_BUY, CAP, COST, BUF = 20, 0.40, 0.35, 15.0, 0.15, 5.0, 0.03
DEFENSIVE = {"GLD", "SLV", "DBC", "TLT"}
REV_CAPS = [0.15, 0.10, 0.05, 0.03]
WINDOWS = [("2008 GFC", "2007-10-01", "2009-06-30"), ("2015-16", "2015-06-01", "2016-06-30"),
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
    spy = close["SPY"].values; spy200 = close["SPY"].rolling(200).mean().values
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
        VOL[c] = close[c].pct_change().rolling(63).std().values
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
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

    def sim(rev_cap):
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n); riskoff = False
        rev_share = []
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    thresh = spy200[j] * (1 - BUF) if riskoff else spy200[j] * (1 + BUF)
                    riskoff = spy[j] < thresh
                sel = {}; kind = {}
                for c in present:
                    if riskoff and c not in DEFENSIVE:
                        continue
                    sg = route(c, j)
                    if sg and fin(VOL[c][j]) and VOL[c][j] > 0 and fin(pv[c][j]):
                        sel[c] = 1.0 / VOL[c][j]; kind[c] = sg
                tot = sum(sel.values())
                tg = {}
                if tot > 0:
                    for c in sel:
                        cap_c = rev_cap if kind[c] == "REV" else CAP
                        tg[c] = min(sel[c] / tot, cap_c)
                    s2 = sum(tg.values())
                    # renormalize but respect per-name caps (2 passes is enough)
                    if s2 > 0:
                        tg = {c: v / s2 for c, v in tg.items()}
                        tg = {c: min(v, rev_cap if kind[c] == "REV" else CAP) for c, v in tg.items()}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
                tot_w = sum(w.values())
                if tot_w > 0:
                    rev_share.append(sum(w[c] for c in w if kind.get(c) == "REV") / tot_w)
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            cash = 1.0 - sum(w.values())
            ret[j + 1] = s + max(cash, 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values, np.mean(rev_share) if rev_share else 0.0

    results = {rc: sim(rc) for rc in REV_CAPS}
    print("gate_eq_iv base; REV per-name cap sweep (MOM cap fixed 15%)")
    print(f"avg REV share of book: " + "  ".join(f"cap{int(rc*100)}%={results[rc][2]:.0%}" for rc in REV_CAPS) + "\n")
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for rc in REV_CAPS:
            r, dts, _ = results[rc]; sh, cg, dd = stats(r, dts, lo, hi)
            tag = " (current)" if rc == 0.15 else ""
            print(f"  rev_cap {int(rc*100):>2d}%{tag:10s} {sh:>6.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
