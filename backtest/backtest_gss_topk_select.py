"""
Should a top-K SELECTION cut (rank routed names by the slope*r2+hi52 blend, keep
best K) be added BEFORE the live inverse-vol sizing? This is the missing test:
the horse-race edge came from concentration, and convblend showed a soft
conviction tilt is a no-op. A HARD top-K cut is the actual concentration lever -
but paired with inverse-vol sizing + caps + deploy cap (so drawdown stays
controlled, unlike top-K + equal-weight which ran -30% DD).

Under live-proxy sizing (inverse-vol, 7.5% cap, 65% deploy, net@5bps):
  base(all)    select ALL routed names           (current live)
  slr2 K=..    keep top-K routed by slope*r2 rank
  blend K=..   keep top-K routed by (slope*r2 + hi52) rank-blend

Answers two things: (1) does concentrating via top-K help under live sizing at
all, and (2) if so, does the hi52 blend pick better than slope*r2 alone.
A change ships only if it beats base in BOTH windows without wrecking maxDD.
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

UNIV = list(TREND_ASSETS.keys())
DATA_START, END = "2010-06-01", "2026-08-21"
COST, SL_WIN = 5.0, 63
CAP, DEPLOY = 0.075, 0.65
ER_HI, ER_LO, RSI_BUY = 0.40, 0.35, 15.0


def roll_slope_r2(p, win):
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


def rsi(p, k):
    d = np.diff(p, prepend=p[0]); up = pd.Series(np.where(d > 0, d, 0.)); dn = pd.Series(np.where(d < 0, -d, 0.))
    return (100 - 100 / (1 + up.rolling(k).mean() / dn.rolling(k).mean().replace(0, np.nan))).values


def er(p, L):
    s = pd.Series(p); return ((s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)).values


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); x = np.asarray(r, float)[m]
    d = x[np.isfinite(x)]
    if len(d) < 15 or d.std() == 0:
        return (np.nan, np.nan, np.nan)
    sh = (d.mean() * 252) / (d.std() * np.sqrt(252))
    eq = np.cumprod(1 + np.nan_to_num(d)); cg = eq[-1] ** (252 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    dl = sorted(set(UNIV + ["BIL", "SPY"]))
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    idx = close.index; n = len(idx); fin = np.isfinite
    bil = close["BIL"].pct_change().fillna(0).reindex(idx).values

    present = [t for t in UNIV if t in close.columns and close[t].notna().sum() > 300]
    pv = {}; m50 = {}; m200 = {}; SLR2 = {}; HI52 = {}; RS2 = {}; ER = {}; VOL = {}
    for c in present:
        px = close[c].reindex(idx); pv[c] = px.values
        m50[c] = px.rolling(50).mean().values; m200[c] = px.rolling(200).mean().values
        s, r = roll_slope_r2(px.values, SL_WIN); SLR2[c] = s * r
        HI52[c] = pv[c] / px.rolling(252).max().values
        RS2[c] = rsi(px.values, 2); ER[c] = er(px.values, 20)
        VOL[c] = pd.Series(px.pct_change()).rolling(63).std().values * np.sqrt(252)

    slr2_rank = pd.DataFrame({c: SLR2[c] for c in present}).rank(axis=1, pct=True)
    hi52_rank = pd.DataFrame({c: HI52[c] for c in present}).rank(axis=1, pct=True)
    RANK_SLR2 = {c: slr2_rank[c].values for c in present}
    RANK_BLEND = {c: ((slr2_rank[c] + hi52_rank[c]) / 2.0).values for c in present}

    si = int(np.searchsorted(idx.values, np.datetime64("2011-01-01")))
    ds = pd.Series(idx); rd = {g.index[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.index[-1] >= si}

    def route(c, j):
        e = ER[c][j]
        if not (fin(e) and fin(pv[c][j]) and fin(m200[c][j])) or pv[c][j] <= m200[c][j]:
            return None
        if e >= ER_HI and fin(m50[c][j]) and pv[c][j] > m50[c][j] and fin(SLR2[c][j]) and SLR2[c][j] > 0:
            return "MOM"
        if e <= ER_LO and fin(RS2[c][j]) and RS2[c][j] < RSI_BUY:
            return "REV"
        return None

    def sim(rankkey, topk):
        # rankkey: None = keep all; else dict of per-day ranks used to cut MOM
        # names to top-K. REV names always kept (reversion sleeve untouched).
        w = {c: 0. for c in present}; ret = np.zeros(n); turn = np.zeros(n)
        for j in range(si, n - 1):
            if j in rd:
                routed = {c: route(c, j) for c in present}
                routed = {c: s for c, s in routed.items() if s and fin(VOL[c][j]) and VOL[c][j] > 0}
                mom = [c for c, s in routed.items() if s == "MOM"]
                rev = [c for c, s in routed.items() if s == "REV"]
                if rankkey is not None and topk and len(mom) > topk:
                    mom = sorted(mom, key=lambda c: (rankkey[c][j] if fin(rankkey[c][j]) else -1),
                                 reverse=True)[:topk]
                sel = mom + rev
                conv = {c: 1.0 / VOL[c][j] for c in sel}
                tot = sum(conv.values())
                if tot > 0:
                    rawv = {c: min(conv[c] / tot, CAP) for c in sel}
                    g = sum(rawv.values())
                    if g > DEPLOY and g > 0:
                        rawv = {c: v * (DEPLOY / g) for c, v in rawv.items()}
                    neww = {c: rawv.get(c, 0.0) for c in present}
                else:
                    neww = {c: 0. for c in present}
                turn[j + 1] += sum(abs(neww[c] - w[c]) for c in present); w = neww
            s = 0.
            for c in present:
                if w[c] != 0 and fin(pv[c][j + 1]) and fin(pv[c][j]):
                    s += w[c] * (pv[c][j + 1] / pv[c][j] - 1.)
            ret[j + 1] = s + max(1.0 - sum(w.values()), 0.) * bil[j + 1]
        sc = slice(si + 1, n)
        return ret[sc] - turn[sc] * (COST / 1e4), idx[sc].values

    print(f"\nLIVE-proxy: ER router + inverse-vol + {CAP:.1%} cap + {DEPLOY:.0%} deploy, net@{COST:.0f}bps")
    print("(top-K cut applies to MOM sleeve only; REV sleeve always kept)")
    print("        LAST 2Y (2024-26)      | full    ")
    print("        Sharpe   CAGR   maxDD   | CAGR    maxDD\n")
    runs = [("base (all routed)", None, 0)]
    for k in (6, 8, 10):
        runs.append((f"slr2  K={k}", RANK_SLR2, k))
    for k in (6, 8, 10):
        runs.append((f"blend K={k}", RANK_BLEND, k))
    for lab, rk, k in runs:
        r, d = sim(rk, k)
        s24, cg24, dd24 = perf(r, d, "2024-01-01", END)
        _, cgf, ddf = perf(r, d, "2011-01-01", END)
        print(f"  {lab:18s} {s24:>5.2f} {cg24:>6.1%} {dd24:>7.1%}   | {cgf:>6.1%} {ddf:>7.1%}")
    spy = close["SPY"].pct_change().reindex(idx).values[si + 1:]
    dsp = idx[si + 1:].values
    s24, cg24, dd24 = perf(spy, dsp, "2024-01-01", END)
    _, cgf, ddf = perf(spy, dsp, "2011-01-01", END)
    print(f"  {'SPY':18s} {s24:>5.2f} {cg24:>6.1%} {dd24:>7.1%}   | {cgf:>6.1%} {ddf:>7.1%}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
