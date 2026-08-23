"""
Does conviction-weighting (regime-aware) and a u_fit tilt add anything ON TOP
of the ER-router selection - or is equal-weight already the edge?

Router (ER20): TREND->momentum long (price>50/200SMA & slope>0);
               CHOP ->reversion long (RSI2<15 & price>200SMA); else FLAT.
Regime-aware base: MOM = max(slope*r2,0); REV = clip((15-rsi2)/15,0,1)*r2.
Schemes (weekly hold 5, net@5bps, 2020-26, train/test):
  buys_ew      equal-weight all trend BUYs (no router)   [baseline]
  adapt_ew     router-select, equal-weight               [the 1.15 edge]
  adapt_conv   router-select, weight ~ regime base
  adapt_ufit   router-select, weight ~ base*(0.3+0.7*u_fit)
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

DATA_START, TRADE_START, END = "2019-01-01", "2020-01-01", "2026-08-21"
SPLIT = pd.Timestamp("2023-06-01")
WIN, FWD, ER_L = 20, 5, 20
ER_HI, ER_LO, RSI_BUY, COST, PATHN = 0.50, 0.35, 15.0, 5.0, 22


def roll_sr(prices, win):
    n = len(prices); slope = np.full(n, np.nan); r2 = np.full(n, np.nan)
    lp = np.log(prices)
    if n < win or sliding_window_view is None:
        return slope, r2
    W = sliding_window_view(lp, win); x = np.arange(win); xc = x - x.mean(); denom = float(xc @ xc)
    ym = W.mean(1); slc = (W - ym[:, None]) @ xc / denom
    pred = slc[:, None] * x[None, :] + (ym - slc * x.mean())[:, None]
    sst = ((W - ym[:, None]) ** 2).sum(1); ssr = ((W - pred) ** 2).sum(1)
    slope[win - 1:] = slc * 1000.0
    r2[win - 1:] = np.clip(np.where(sst > 0, 1 - ssr / sst, 0), 0, 1)
    return slope, r2


def rsi(prices, p):
    d = np.diff(prices, prepend=prices[0])
    up = pd.Series(np.where(d > 0, d, 0.0)); dn = pd.Series(np.where(d < 0, -d, 0.0))
    return (100 - 100 / (1 + up.rolling(p).mean() / dn.rolling(p).mean().replace(0, np.nan))).values


def eff_ratio(prices, L):
    s = pd.Series(prices)
    return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def ufit(sp, rp):
    if len(sp) < 15 or not (np.isfinite(sp).all() and np.isfinite(rp).all()):
        return 0.0
    L = len(sp); a, b = L // 5, (4 * L) // 5
    down = np.clip(-sp[0] / 3.0, 0, 1); brk = np.clip((0.30 - rp[a:b].min()) / 0.30, 0, 1)
    return float(down * brk * (1.0 if sp[-1] > 0 else 0.0))


def perf(r, dates):
    r = np.asarray(r, float)
    def sh(m):
        x = r[m]; x = x[np.isfinite(x)]
        return (x.mean() * (252 / FWD)) / (x.std() * np.sqrt(252 / FWD)) if len(x) > 10 and x.std() > 0 else np.nan
    eq = np.cumprod(1 + np.nan_to_num(r)); cagr = eq[-1] ** ((252 / FWD) / len(r)) - 1
    return sh(np.ones(len(r), bool)), sh(dates < SPLIT), sh(dates >= SPLIT), cagr, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx)
    print(f"Universe: {len(present)}\n")

    pv = {}; ma50 = {}; ma200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); reidx = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN)
        pv[c] = close[c].values
        ma50[c] = close[c].rolling(50).mean().values
        ma200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = reidx(s), reidx(r)
        RS2[c] = reidx(rsi(v.values, 2)); ER[c] = reidx(eff_ratio(v.values, ER_L))

    start_i = int(np.searchsorted(idx.values, np.datetime64(TRADE_START)))
    fin = np.isfinite

    def route(c, i, er_hi):
        e = ER[c][i]
        if not fin(e) or not fin(pv[c][i]):
            return "FLAT", 0.0
        a200 = pv[c][i] > ma200[c][i]
        if e >= er_hi:                              # TREND gate (widened here)
            if a200 and pv[c][i] > ma50[c][i] and SL[c][i] > 0:
                return "MOM", max(SL[c][i] * R2[c][i], 0.0)
        elif e <= ER_LO:                            # CHOP gate (kept strict at 0.35)
            if a200 and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY:
                return "REV", np.clip((RSI_BUY - RS2[c][i]) / RSI_BUY, 0, 1) * R2[c][i]
        return "FLAT", 0.0

    def uf(c, i):
        return ufit(SL[c][i - PATHN + 1:i + 1], R2[c][i - PATHN + 1:i + 1])

    def run(scheme, er_hi=ER_HI):
        prev = {c: 0.0 for c in present}; rr = []; dts = []; nm = []
        for i in range(max(start_i, 200), n - FWD, FWD):
            sel = {}
            if scheme == "buys_ew":
                sel = {c: 1.0 for c in present
                       if pv[c][i] > ma200[c][i] and pv[c][i] > ma50[c][i] and SL[c][i] > 0 and R2[c][i] > 0.6
                       and fin(pv[c][i + FWD])}
            else:
                for c in present:
                    if not fin(pv[c][i + FWD]):
                        continue
                    sig, base = route(c, i, er_hi)
                    if sig in ("MOM", "REV") and base > 0:
                        sel[c] = base
            tot = sum(sel.values())
            w = {c: sel[c] / tot for c in sel} if tot > 0 else {}
            gross = sum(w[c] * (pv[c][i + FWD] / pv[c][i] - 1.0) for c in w)
            turn = sum(abs(w.get(c, 0.0) - prev.get(c, 0.0)) for c in set(w) | set(prev))
            rr.append(gross - turn * (COST / 1e4)); dts.append(idx[i]); prev = w; nm.append(len(w))
        return np.array(rr), pd.DatetimeIndex(dts), np.mean(nm)

    print(f"router ER{ER_L}, hold {FWD}d, net@{COST}bps, SPLIT {SPLIT.date()}\n")
    print(f"{'scheme':16s} {'FULL':>6s} {'TRAIN':>6s} {'TEST':>6s} {'CAGR':>7s} {'maxDD':>7s} {'names':>6s}")
    print("-" * 64)
    spy = close["SPY"].values
    sp = [spy[i + FWD] / spy[i] - 1 for i in range(max(start_i, 200), n - FWD, FWD)]
    dd = pd.DatetimeIndex([idx[i] for i in range(max(start_i, 200), n - FWD, FWD)])
    f, tr, te, cg, mdd = perf(np.array(sp), dd)
    print(f"{'SPY B&H':16s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {mdd:>7.1%} {'-':>6s}")
    r, d, avgn = run("buys_ew")
    f, tr, te, cg, mdd = perf(r, d)
    print(f"{'buys_ew':16s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {mdd:>7.1%} {avgn:>6.1f}")
    print("-- adapt_conv, TREND gate widened (CHOP fixed at 0.35) --")
    for er_hi in [0.50, 0.45, 0.40, 0.36]:
        r, d, avgn = run("adapt_conv", er_hi)
        f, tr, te, cg, mdd = perf(r, d)
        print(f"{'TREND>='+format(er_hi,'.2f'):16s} {f:>6.2f} {tr:>6.2f} {te:>6.2f} {cg:>7.1%} {mdd:>7.1%} {avgn:>6.1f}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
