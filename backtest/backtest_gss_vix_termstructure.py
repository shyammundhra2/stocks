"""
Is the VIX term structure (VIX/VIX3M) a BETTER regime signal than spot VIX? The
HMM's VIX feature is a spot-VIX z-score (lagging fear proxy); term structure is
forward-looking (backwardation VIX>VIX3M flags stress before spot VIX rises).
If the term-structure throttle protects the book with LESS CAGR give-up than the
spot-VIX throttle, then feeding VIX3M to the HMM (or the deploy throttle) is a
real upgrade.

VIX3M sourced from CBOE (yfinance lacks it). Overlay = scale risky exposure
daily by f in [0,1], rest to BIL; f known at close j, applied j->j+1.

  base        f = 1 (current)
  spot_level  f = clip(1.3 - VIX/40, 0.3, 1.0)         (best spot-VIX from before)
  ts_gate     hysteresis on VIX/VIX3M: f=0.5 when ratio>1.00 (backwardation),
              back to 1.0 when ratio<0.95
  ts_cont     f = clip(1.0 - 2.5*(ratio - 0.90), 0.3, 1.0)  (throttle as ratio->1)

Bar: better drawdown protection per unit CAGR given up, in BOTH 2020-26 and
2011-26 (term structure should be more robust across regimes than spot level).
"""
import ssl
import sys
import time
import io
import urllib.request

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
COST, CAP, DEPLOY, TOPK, MAP_MULT = 5.0, 0.075, 0.65, 8, 2000.0
ER_HI, ER_LO, RSI_BUY = 0.40, 0.35, 15.0


def _cboe(sym):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=ctx, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw)); df.columns = [c.strip().upper() for c in df.columns]
    dcol = [c for c in df.columns if "DATE" in c][0]; ccol = [c for c in df.columns if "CLOSE" in c][0]
    df[dcol] = pd.to_datetime(df[dcol])
    return df.set_index(dcol)[ccol].sort_index()


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
    dl = sorted(set(UNIV + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers + CBOE VIX/VIX3M ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values
    vix = _cboe("VIX").reindex(idx).ffill(); vix3m = _cboe("VIX3M").reindex(idx).ffill()
    vixv = vix.values; ts = (vix / vix3m).values                      # term structure ratio
    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]

    pv = {}; m50 = {}; m200 = {}; SL63 = {}; MSL = {}; MR2 = {}; HI = {}; RS = {}; ER = {}; VOL = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s63, r63 = roll_sl_r2(px.values, 63); SL63[c] = s63 * r63
        s20, r20 = roll_sl_r2(px.values, 20); MSL[c] = s20 * MAP_MULT; MR2[c] = r20
        HI[c] = pv[c] / px.rolling(252).max().values
        RS[c] = rsi(px.values); ER[c] = er(px.values)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)
    XABS = np.maximum(pd.DataFrame({c: np.abs(MSL[c]) for c in present}).max(axis=1).values, 3.0)
    slxr = pd.DataFrame({c: SL63[c] for c in present}).rank(axis=1, pct=True)
    hir = pd.DataFrame({c: HI[c] for c in present}).rank(axis=1, pct=True)
    BL = {c: ((slxr[c] + hir[c]) / 2).values for c in present}
    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def conv(c, j):
        v = abs(MSL[c][j]) * MR2[c][j]
        return v if fin(v) else 0.0

    def route(c, j):
        e = ER[c][j]
        if not (fin(e) and fin(pv[c][j]) and fin(m200[c][j])) or pv[c][j] <= m200[c][j]:
            return None
        if e >= ER_HI and fin(m50[c][j]) and pv[c][j] > m50[c][j] and fin(SL63[c][j]) and SL63[c][j] > 0:
            return "MOM"
        if e <= ER_LO and fin(RS[c][j]) and RS[c][j] < RSI_BUY and conv(c, j) < 0.18 * XABS[j]:
            return "REV"
        return None

    def f_base():
        return np.ones(n)

    def f_spot_level():
        return np.clip(1.3 - vixv / 40.0, 0.3, 1.0)

    def f_ts_gate():
        f = np.ones(n); off = False
        for j in range(n):
            if not fin(ts[j]):
                f[j] = 1.0; continue
            if off:
                if ts[j] < 0.95:
                    off = False
            elif ts[j] > 1.00:
                off = True
            f[j] = 0.5 if off else 1.0
        return f

    def f_ts_cont():
        return np.where(fin(ts), np.clip(1.0 - 2.5 * (ts - 0.90), 0.3, 1.0), 1.0)

    def sim(fac):
        w = {c: 0. for c in present}; ret = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                mom = [c for c in present if route(c, j) == "MOM" and fin(VOL[c][j]) and VOL[c][j] > 0]
                rev = [c for c in present if route(c, j) == "REV" and fin(VOL[c][j]) and VOL[c][j] > 0]
                if len(mom) > TOPK:
                    mom = sorted(mom, key=lambda c: (BL[c][j] if fin(BL[c][j]) else -1), reverse=True)[:TOPK]
                sel = mom + rev; cv = {c: 1.0 / VOL[c][j] for c in sel}; tot = sum(cv.values())
                if tot > 0:
                    rv = {c: min(cv[c] / tot, CAP) for c in sel}; g = sum(rv.values())
                    if g > DEPLOY and g > 0:
                        rv = {c: v * (DEPLOY / g) for c, v in rv.items()}
                    nw = {c: rv.get(c, 0.0) for c in present}
                else:
                    nw = {c: 0. for c in present}
                w = nw
            fj = fac[j]
            s = sum(fj * w[c] * (pv[c][j + 1] / pv[c][j] - 1.) for c in present
                    if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]))
            ret[j + 1] = s + max(1.0 - fj * sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc], idx[sc].values

    print(f"\nSpot-VIX vs term-structure throttle. net@{COST:.0f}bps")
    print("              Sh 11-19/20-26  full CAGR/maxDD  |  2020-26 CAGR/maxDD")
    for lab, fac in [("base", f_base()), ("spot_level", f_spot_level()),
                     ("ts_gate", f_ts_gate()), ("ts_cont", f_ts_cont())]:
        r, d = sim(fac)
        s1, _, _ = perf(r, d, "2011-01-01", "2019-12-31")
        s2, _, _ = perf(r, d, "2020-01-01", END)
        _, cgf, ddf = perf(r, d, "2011-01-01", END)
        _, c2, dd2 = perf(r, d, "2020-01-01", END)
        print(f"  {lab:11s}  {s1:>4.2f}/{s2:>4.2f}   {cgf:>5.1%}/{ddf:>6.1%}  |  {c2:>5.1%}/{dd2:>6.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
