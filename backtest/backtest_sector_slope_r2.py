"""
Do SECTORS have a slope*r2 edge - tested at the RIGHT horizon and structure?
The prior ranker test used 63d forward + top-1-pick, but the session's
validated slope*r2 edge was ~2-week horizon as a LONG-TOP/SHORT-BOTTOM
cross-sectional spread. This tests slope*r2 (and momentum, for contrast)
on the 11 sector ETFs at horizons 5/10/21/42/63d, two structures:

  IC          - cross-sectional Spearman(signal rank, fwd return rank)
  LS spread   - mean fwd return of top-third MINUS bottom-third (the actual
                dollar-neutral book), annualized, net of a rough 10bps/turn

If slope*r2 IC and the LS spread are positive at 10-21d and decay by 63d,
that reproduces the known edge inside sectors. If flat/negative everywhere,
sectors genuinely lack it. 2007-26, strided by horizon.
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

SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
DATA_START, EVAL_START, END = "2004-01-01", "2007-01-01", "2026-08-21"
HORIZONS = [5, 10, 21, 42, 63]
COST = 10.0  # bps per full turn of the LS book, charged per rebalance


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


def main():
    t0 = time.time()
    print("Downloading sectors ...")
    close = yf.download(SECTORS, start=DATA_START, end=END, progress=False,
                        auto_adjust=True)["Close"][SECTORS].dropna(how="all")
    n = len(close)

    # slope*r2 (20d) and momentum(21d) signal frames
    sr20 = {}; sr63 = {}
    for c in SECTORS:
        s, r = roll_sr(close[c].values, 20); sr20[c] = s * r
        s6, r6 = roll_sr(close[c].values, 63); sr63[c] = s6 * r6
    SR20 = pd.DataFrame(sr20, index=close.index)
    SR63 = pd.DataFrame(sr63, index=close.index)
    MOM = close.pct_change(21)

    signals = {"slope_r2_20": SR20, "slope_r2_63": SR63, "mom21": MOM}
    idx = close.index
    eval_mask = idx >= EVAL_START

    print(f"\n{'signal':>13s} {'H':>4s} {'IC':>7s} {'LS ann%':>8s} {'LSnet ann%':>11s} {'LS Sharpe':>9s} {'n':>5s}")
    print("-" * 68)
    for sname, S in signals.items():
        for H in HORIZONS:
            fwd = close.pct_change(H).shift(-H)
            dates = idx[eval_mask][::H]            # non-overlapping at this horizon
            ics = []; ls = []
            for d in dates:
                if d not in S.index or d not in fwd.index:
                    continue
                sv = S.loc[d]; fv = fwd.loc[d]
                m = sv.notna() & fv.notna()
                if m.sum() < 6:
                    continue
                sv = sv[m]; fv = fv[m]
                ics.append(sv.rank().corr(fv.rank()))
                k = max(1, m.sum() // 3)
                order = sv.sort_values()
                longs = order.index[-k:]; shorts = order.index[:k]
                ls.append(fv[longs].mean() - fv[shorts].mean())
            if not ics:
                continue
            ppy = 252 / H
            ls = np.array(ls)
            ann = ls.mean() * ppy
            ann_net = (ls.mean() - 2 * COST / 1e4) * ppy    # ~2 legs turn each rebal
            shp = (ls.mean() * ppy) / (ls.std() * np.sqrt(ppy)) if ls.std() > 0 else np.nan
            print(f"{sname:>13s} {H:>3d}d {np.mean(ics):>7.3f} {ann*100:>7.1f}% "
                  f"{ann_net*100:>10.1f}% {shp:>9.2f} {len(ics):>5d}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
