"""
backtest_gss_whipsaw.py validated A 3% hysteresis buffer around the SPY-
200DMA gate vs. no buffer - but only ever tested buf=0.03, never swept
other widths. The premise worth testing directly: everyone watches the
200DMA, so stop-losses and algo triggers cluster exactly there, making
the raw crossing noisy - find the buffer width that actually performs
best, not just confirm that SOME buffer beats none.

Same exact simulation as backtest_gss_whipsaw.py (ER router, inverse-vol-
free equal-weight-capped sizing, net@5bps, monthly rebalance), varying
ONLY the hysteresis buffer width: 0% (no gate), 1%, 2%, 3% (current),
4%, 5%, 7%, 10%. Same windows: the whipsaw years (2015-16, 2018 Q4), 2022
bear, full 2007-26, and dev 2020-26 (must not break the good regime).
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
WIN, ER_L, ER_HI, ER_LO, RSI_BUY, CAP, COST = 20, 20, 0.40, 0.35, 15.0, 0.15, 5.0
BUFS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
WINDOWS = [("2015-16", "2015-06-01", "2016-06-30"), ("2018 Q4", "2018-09-01", "2019-01-31"),
           ("2022 bear", "2022-01-01", "2022-10-31"), ("FULL 2007-26", TRADE_START, END),
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
    if len(d) < 20 or d.std() == 0:
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

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, ER_L))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)

    def route(c, i):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return 0.
        a200 = pv[c][i] > m200[c][i]
        if e >= ER_HI and a200 and pv[c][i] > m50[c][i] and SL[c][i] > 0:
            return max(SL[c][i] * R2[c][i], 0.)
        if e <= ER_LO and a200 and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
            return np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return 0.

    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def sim(buf):
        w = {c: 0. for c in present}; eqret = np.zeros(n); turn = np.zeros(n); inv = np.zeros(n)
        riskoff = False; flips = 0
        for j in range(si, n - 1):
            if j in rd:
                if fin(spy200[j]):
                    # Proper sticky hysteresis, matching the LIVE _spy_risk_gate
                    # exactly (macro/indicators/portfolio.py): exit risk-off only
                    # above the UPPER band, enter risk-off only below the LOWER
                    # band; in between, state is UNCHANGED. backtest_gss_whipsaw.py's
                    # version re-picked a single threshold off the CURRENT state
                    # and compared with the same operator both ways - confirmed by
                    # direct test to flip on every step even for prices oscillating
                    # gently inside the band, regardless of buffer width (explains
                    # why that run's flip-count rose WITH buffer width instead of
                    # falling). That bug pre-dates this script; fixed here, not
                    # touching the live gate since _spy_risk_gate itself is correct.
                    new_riskoff = riskoff
                    if riskoff:
                        if spy[j] > spy200[j] * (1.0 + buf):
                            new_riskoff = False
                    else:
                        if spy[j] < spy200[j] * (1.0 - buf):
                            new_riskoff = True
                    if new_riskoff != riskoff:
                        flips += 1
                    riskoff = new_riskoff
                if riskoff:
                    tg = {}
                else:
                    sel = {c: route(c, j) for c in present}
                    sel = {c: v for c, v in sel.items() if v > 0 and fin(pv[c][j])}
                    tot = sum(sel.values()); tg = {c: min(sel[c] / tot, CAP) for c in sel} if tot > 0 else {}
                    s2 = sum(tg.values()); tg = {c: v / s2 for c, v in tg.items()} if s2 > 0 else {}
                full = {c: tg.get(c, 0.) for c in present}
                turn[j + 1] += sum(abs(full[c] - w[c]) for c in present); w = full
            s = 0.
            for c in present:
                if w[c] > 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            eqret[j + 1] = s; inv[j + 1] = sum(w.values())
        sc = slice(si + 1, n); net = eqret[sc] - turn[sc] * (COST / 1e4)
        return net + bil[sc] * (1 - inv[sc]), idx[sc].values, flips

    results = {b: sim(b) for b in BUFS}
    spy_ret = close["SPY"].pct_change().values[si + 1:]; dts0 = idx[si + 1:].values

    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'flips(full)':>12s}")
        for b in BUFS:
            r, dts, flips = results[b]; sh, cg, dd = stats(r, dts, lo, hi)
            tag = f"buf={b:.0%}" + ("  <- current" if b == 0.03 else "")
            print(f"  {tag:18s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {flips:>12d}")
        sh, cg, dd = stats(spy_ret, dts0, lo, hi)
        print(f"  {'SPY':18s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}\n")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
