"""
Which of the 15 breadth ETFs IMPROVE CAGR at full deployment? Start from the
36-name base, add each candidate one at a time, measure 2020-2026 full-
deployment CAGR delta. Keep the ones that improve CAGR, drop the rest.

CAVEAT (honest): selecting names on ex-post 2020-26 CAGR is data-mining - the
names that helped in this window may not going forward. Reported alongside the
2011-19 delta so you can see which improvements are regime-specific vs robust.
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

CAND_NAMES = {"XBI":"Biotech","XHB":"Homebuilders","XRT":"Retail","KRE":"Regional Banks",
              "XOP":"Oil & Gas E&P","OIH":"Oil Services","IYT":"Transports","GDX":"Gold Miners",
              "KIE":"Insurance","FDN":"Internet","IHI":"Medical Devices","SIL":"Silver Miners",
              "REMX":"Rare Earth/Strategic Metals","LIT":"Lithium/Battery","GNR":"Natural Resources"}
CANDIDATES = list(CAND_NAMES)
BASE36 = [t for t in TREND_ASSETS if t not in CANDIDATES]
DATA_START, END = "2010-06-01", "2026-08-21"
WIN, ER_HI, ER_LO, RSI_BUY, COST = 20, 0.40, 0.35, 15.0, 5.0


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


def cagr(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15:
        return np.nan
    eq = np.cumprod(1 + np.nan_to_num(d))
    return eq[-1] ** (252 / len(d)) - 1


def main():
    t0 = time.time()
    dl = sorted(set(BASE36 + CANDIDATES + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    allnames = [t for t in BASE36 + CANDIDATES if t in close.columns and close[t].notna().sum() > 200]
    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in allnames:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].reindex(idx).values
        m50[c] = close[c].rolling(50).mean().reindex(idx).values
        m200[c] = close[c].rolling(200).mean().reindex(idx).values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
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

    def sim(universe):
        present = [c for c in universe if c in allnames]
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                sel = [c for c in present if route(c, j) and fin(pv[c][j])]
                neww = {c: (1.0 / len(sel) if c in sel else 0.0) for c in present} if sel else {c: 0. for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    base_r, dts = sim(BASE36)
    base20 = cagr(base_r, dts, "2020-01-01", END)
    base11 = cagr(base_r, dts, "2011-01-01", "2019-12-31")
    print(f"\nBase 36-name full-deploy CAGR: 2020-26 {base20:.1%}   2011-19 {base11:.1%}\n")
    print(f"{'add':>6s} {'d2020-26':>9s} {'d2011-19':>9s}   name")
    print("-" * 48)
    rows = []
    for c in CANDIDATES:
        r, d = sim(BASE36 + [c])
        d20 = cagr(r, d, "2020-01-01", END) - base20
        d11 = cagr(r, d, "2011-01-01", "2019-12-31") - base11
        rows.append((c, d20, d11))
    for c, d20, d11 in sorted(rows, key=lambda x: -x[1]):
        tag = "  <- improves 2020-26" if d20 > 0 else ""
        print(f"{c:>6s} {d20:>+8.2%} {d11:>+8.2%}   {CAND_NAMES[c]}{tag}")
    keep = [c for c, d20, _ in rows if d20 > 0]
    print(f"\nKeep (improve 2020-26 CAGR): {keep}")
    print(f"Drop: {[c for c,d20,_ in rows if d20<=0]}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
