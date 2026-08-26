"""
Does ChatGPT's Trendguard ENTRY (rules only, no ML - the one part that showed
value: +0.81%/21d) beat or complement the live ER-router MOM entry? Same GSS ETF
universe, monthly rebalance, equal-weight selected names, buffered SPY-200DMA gate
(+/-3%), net@5bps. Isolates the entry rule (no sizing/REV/throttle machinery).

  ER_router   ER20>=0.40 & Close>SMA200 & Close>SMA50 & 20d-return>0   (current)
  Trendguard  SMA50>SMA200 & Close>prior-20d-high & ATR14<2*mean(ATR14,126) & RSI<70
  Both        intersection (must satisfy both)
  Either      union
vs EQ-hold universe and SPY.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/tg_ohlc.parquet"
COST, BUF = 5.0, 0.03


def rsi(s, k=14):
    d = s.diff(); up = d.clip(lower=0).rolling(k).mean(); dn = (-d.clip(upper=0)).rolling(k).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def er(s, L=20):
    return (s - s.shift(L)).abs() / s.diff().abs().rolling(L).sum().replace(0, np.nan)


def perf(r, dts, lo, hi):
    m = (dts >= np.datetime64(lo)) & (dts <= np.datetime64(hi)); d = np.asarray(r)[m]
    d = d[np.isfinite(d)]
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); return sh, eq[-1] ** (12 / len(d)) - 1, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    raw = pd.read_parquet(CACHE)
    close = raw["Close"]; high = raw["High"]; low = raw["Low"]
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    idx = close.index
    names = [s for s in TREND_ASSETS if s in close.columns and close[s].notna().sum() > 400]

    # buffered SPY gate state
    off = False; gate = pd.Series(False, index=idx); sv, s2 = spy.values, spy200.values
    for i in range(len(idx)):
        if np.isfinite(s2[i]):
            if off and sv[i] > s2[i] * (1 + BUF):
                off = False
            elif (not off) and sv[i] < s2[i] * (1 - BUF):
                off = True
        gate.iloc[i] = off

    ER = {}; TG = {}; pv = {}
    for c in names:
        px = close[c]; hi = high[c]; lo = low[c]; pv[c] = px.values
        sma50 = px.rolling(50).mean(); sma200 = px.rolling(200).mean()
        high20 = hi.rolling(20).max().shift(1)
        tr = pd.concat([hi - lo, (hi - px.shift()).abs(), (lo - px.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        er_ok = (er(px, 20) >= 0.40) & (px > sma200) & (px > sma50) & (px / px.shift(20) - 1 > 0)
        tg_ok = (sma50 > sma200) & (px > high20) & (atr < 2 * atr.rolling(126).mean()) & (rsi(px, 14) < 70)
        ER[c] = er_ok.reindex(idx).fillna(False).values
        TG[c] = tg_ok.reindex(idx).fillna(False).values

    si = int(np.searchsorted(idx.values, np.datetime64("2006-01-01")))
    ds = pd.Series(idx); reb = sorted({g.iloc[-1] for _, g in ds.groupby(ds.dt.to_period("M")) if g.iloc[-1] >= idx[si]})
    reb_i = [idx.get_loc(d) for d in reb]

    rules = {"ER_router": ER, "Trendguard": TG,
             "Both": {c: ER[c] & TG[c] for c in names},
             "Either": {c: ER[c] | TG[c] for c in names},
             "EQ-hold": {c: np.ones(len(idx), bool) for c in names}}
    res = {}
    for name, sel in rules.items():
        ret = []; dts = []; prev = set()
        for k in range(len(reb_i) - 1):
            j0, j1 = reb_i[k], reb_i[k + 1]
            goff = bool(gate.iloc[j0]) and name != "EQ-hold"
            chosen = [] if goff else [c for c in names if sel[c][j0] and np.isfinite(pv[c][j1]) and np.isfinite(pv[c][j0])]
            if chosen:
                r = np.mean([pv[c][j1] / pv[c][j0] - 1 for c in chosen])
            else:
                r = 0.0
            to = len(set(chosen) ^ prev) / max(len(chosen) + len(prev), 1) if chosen or prev else 0.0
            ret.append(r - to * COST / 1e4); dts.append(idx[j1]); prev = set(chosen)
        res[name] = (np.array(ret), np.array(dts, dtype="datetime64[ns]"))
    spm = np.array([spy.iloc[reb_i[k + 1]] / spy.iloc[reb_i[k]] - 1 for k in range(len(reb_i) - 1)])
    res["SPY"] = (spm, res["ER_router"][1])

    wins = [("2006-2012", "2006-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"), ("FULL", "2006-01-01", "2026-08-21")]
    print(f"\nTrendguard entry vs ER-router (rules only, gated, eq-wt), net@{COST:.0f}bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==          {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for name in ["ER_router", "Trendguard", "Both", "Either", "EQ-hold", "SPY"]:
            r, d = res[name]; sh, cg, dd = perf(r, d, lo, hi)
            star = " *" if name == "ER_router" else "  "
            print(f"  {name:11s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
