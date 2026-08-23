"""
How can ETFs work for relative value? The dollar-neutral slope*r2 L/S failed
because on 36 macro ETFs it's a DISGUISED net-long-beta bet (momentum longs
carry more market beta than the shorts). Test the fix: BETA-neutral - hedge the
book's residual beta with SPY so you trade the relative signal, not beta.

  dollar-neutral : long top third / short bottom third, equal notional  [failed]
  beta-neutral   : same legs + SPY hedge sized to zero net rolling beta

2-week rebalance (native horizon), net@5bps, 2007-2026, chop windows + full.
If beta-neutral compounds where dollar-neutral didn't, ETF market-neutral is
viable; if it still doesn't, ETFs only work long-only cross-asset + carry.
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
WIN, BETA_W, HOLD, COST = 20, 60, 10, 5.0
WINDOWS = [("2011 chop", "2011-05-01", "2011-12-31"), ("2015-16", "2015-06-01", "2016-06-30"),
           ("2018 Q4", "2018-09-01", "2019-01-31"), ("FULL 2007-26", TRADE_START, END),
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
    tickers = list(TREND_ASSETS.keys())
    print(f"Downloading {len(tickers)} tickers ...")
    raw = yf.download(tickers, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    present = [t for t in tickers if t in close.columns and close[t].notna().sum() > 260]
    idx = close.index; n = len(idx); fin = np.isfinite
    spy_r = close["SPY"].pct_change().values
    var_spy = pd.Series(spy_r).rolling(BETA_W).var().values
    print(f"Universe: {len(present)}\n")

    pv = {}; SL = {}; R2 = {}; BETA = {}
    for c in present:
        v = close[c].dropna(); ri = lambda a: pd.Series(a, index=v.index).reindex(idx).values
        s, r = roll_sr(v.values, WIN); pv[c] = close[c].values
        SL[c], R2[c] = ri(s), ri(r)
        rc = close[c].pct_change()
        cov = rc.rolling(BETA_W).cov(pd.Series(spy_r, index=idx))
        BETA[c] = (cov.values / var_spy)
    si = max(int(np.searchsorted(idx.values, np.datetime64(TRADE_START))), 220)
    rebal = set(range(si, n - 1, HOLD))

    def sim(beta_neutral):
        w = {c: 0. for c in present}; hspy = 0.0; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rebal:
                sc_ = {c: SL[c][j] * R2[c][j] for c in present
                       if fin(SL[c][j]) and fin(R2[c][j]) and fin(pv[c][j]) and fin(pv[c][j + 1])}
                names = sorted(sc_, key=sc_.get); k = max(1, len(names) // 3)
                new = {c: 0. for c in present}
                for c in names[-k:]:
                    new[c] = 0.5 / k
                for c in names[:k]:
                    new[c] = -0.5 / k
                nb = 0.0
                if beta_neutral:
                    nb = sum(new[c] * BETA[c][j] for c in present if fin(BETA[c][j]))
                new_h = -nb
                turn[j + 1] += sum(abs(new[c] - w[c]) for c in present) + abs(new_h - hspy)
                w = new; hspy = new_h
            s = 0.0
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            if hspy != 0 and fin(spy_r[j + 1]):
                s += hspy * spy_r[j + 1]
            ret[j + 1] = s
        sc = slice(si + 1, n); return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    dn, dts = sim(False); bn, _ = sim(True); spy = spy_r[si + 1:]
    series = [("dollar-neutral", dn), ("beta-neutral (SPY-hedged)", bn), ("SPY", spy)]
    for lab, lo, hi in WINDOWS:
        print(f"== {lab} ==   {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for nm, r in series:
            sh, cg, dd = stats(r, dts, lo, hi)
            print(f"  {nm:26s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
