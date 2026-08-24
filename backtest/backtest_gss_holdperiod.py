"""
How long to hold MOM vs REV positions? Event study of forward-return decay:
for every day a name fires a MOM signal (or a REV signal), record its forward
return at horizons H, then find where the edge lives.

Decision metric = NET return-per-day = (mean_fwd_H - roundtrip_cost) / H. A
round trip costs the same (~10bps) whatever H, so short holds pay more per day;
this folds that in. The H that maximizes net-per-day is the optimal hold.

  MOM signal: ER>=0.40 & >200/50DMA & slope>0   (trend - expected slow)
  REV signal: ER<=0.35 & >200DMA & RSI2<15       (oversold - expected fast)
Full 2007-2026, per-signal, gross + net@5bps/side.
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
WIN, ER_HI, ER_LO, RSI_BUY, RT_COST = 20, 0.40, 0.35, 15.0, 0.0010  # 2x5bps round trip
HZ = [1, 2, 3, 5, 10, 15, 21, 42, 63]


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


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    print(f"Universe: {len(present)}\n")

    pv = {}; m50 = {}; m200 = {}; SL = {}; R2 = {}; RS2 = {}; ER = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        m50[c] = close[c].rolling(50).mean().values; m200[c] = close[c].rolling(200).mean().values
        SL[c], R2[c] = ri(s), ri(r); RS2[c] = ri(rsi(v.values, 2)); ER[c] = ri(er(v.values, 20))
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)

    def is_mom(c, i):
        return (fin(ER[c][i]) and ER[c][i] >= ER_HI and fin(pv[c][i]) and pv[c][i] > m200[c][i]
                and pv[c][i] > m50[c][i] and fin(SL[c][i]) and SL[c][i] > 0)

    def is_rev(c, i):
        return (fin(ER[c][i]) and ER[c][i] <= ER_LO and fin(pv[c][i]) and pv[c][i] > m200[c][i]
                and fin(RS2[c][i]) and RS2[c][i] < RSI_BUY)

    def collect(sig):
        fwd = {h: [] for h in HZ}
        for c in present:
            for i in range(si, n - max(HZ) - 1):
                if sig(c, i) and fin(pv[c][i]):
                    for h in HZ:
                        p2 = pv[c][i + h]
                        if fin(p2):
                            fwd[h].append(p2 / pv[c][i] - 1.0)
        return fwd

    def report(name, fwd):
        n0 = len(fwd[HZ[0]])
        print(f"== {name} sleeve ==  ({n0} signals)")
        print(f"  {'hold(d)':>8s} {'meanFwd%':>9s} {'hit%':>6s} {'net/day(bps)':>13s} {'annNet%':>8s}")
        best_h, best_pd = None, -1e9
        for h in HZ:
            a = np.array(fwd[h]); a = a[np.isfinite(a)]
            if len(a) < 20:
                continue
            m = a.mean(); hit = (a > 0).mean(); net = m - RT_COST
            perday = net / h * 1e4; ann = (1 + net) ** (252 / h) - 1
            star = ""
            if perday > best_pd:
                best_pd, best_h = perday, h
            print(f"  {h:>8d} {m * 100:>8.2f} {hit * 100:>5.0f}% {perday:>12.2f} {ann:>7.1%}")
        print(f"  -> best net-per-day at hold = {best_h} days\n")

    report("MOM (trend)", collect(is_mom))
    report("REV (mean-revert)", collect(is_rev))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
