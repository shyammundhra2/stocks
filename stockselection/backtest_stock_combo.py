"""
Can momentum + a crash filter beat equal-weight ROBUSTLY (all regimes), fixing
momentum's 2008 crash (long-only mom12_1 Sharpe 0.61, -51% DD in 2005-12)?
Same cross-sectional S&P 500 engine (cached), monthly, net@10bps. Variants on the
mom12_1 top quintile:

  mom_base     top-quintile momentum, equal weight            (the base)
  mom_ivol     top-quintile momentum, INVERSE-VOL weighted    (tilt to calmer names)
  mom_x_lowvol double sort: top-40% momentum AND bottom-40% 126d vol, eq-wt
  mom_spygate  top-quintile momentum WHEN SPY>200DMA, else CASH (0%) - the
               classic momentum-crash market-timing overlay
  mom_owntrend top-quintile momentum among names above their OWN 200DMA

Benchmarks: EW-universe (fair, absorbs survivorship), SPY. A variant WINS only if
it beats EW-universe Sharpe in ALL THREE sub-periods AND cuts the 2008 drawdown.
CASH credited 0% (conservative - real T-bill carry would only help the gate).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
COST, QUINTILE = 10.0, 0.20


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    t0 = time.time()
    close = pd.read_parquet(CACHE)
    spy = close["SPY"]; spy_200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]
    idx = cs.index
    logret = np.log(cs).diff()

    mom = cs.shift(21) / cs.shift(252) - 1.0
    vol = logret.rolling(126).std()
    own200 = cs.rolling(200).mean()

    me = cs.resample("ME").last().index
    reb = []
    for d in me:
        pos = idx.searchsorted(d, side="right") - 1
        if pos > 260:
            reb.append(idx[pos])
    reb = sorted(set(reb))

    variants = ["mom_base", "mom_ivol", "mom_x_lowvol", "mom_spygate", "mom_owntrend",
                "EW-universe", "SPY"]
    ser = {v: [] for v in variants}
    dts = []
    prev = {v: set() for v in variants}

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m = mom.loc[d0]; v = vol.loc[d0]
        fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwdv, vv = m[ok], fwd[ok], v[ok]
        if len(m) < 50:
            continue
        n_q = max(int(len(m) * QUINTILE), 5)
        order = m.sort_values()
        longs = list(order.index[-n_q:])
        spy_on = bool(spy.loc[d0] > spy_200.loc[d0]) if np.isfinite(spy_200.loc[d0]) else True

        def book(sel, weights=None, cash=False):
            if cash or len(sel) == 0:
                return 0.0, set()
            if weights is None:
                r = fwd[sel].mean()
            else:
                w = weights / weights.sum()
                r = float((fwd[sel] * w).sum())
            return r, set(sel)

        picks = {}
        picks["mom_base"] = book(longs)
        iv = (1.0 / vv[longs]).replace([np.inf, -np.inf], np.nan).dropna()
        picks["mom_ivol"] = book(list(iv.index), weights=iv)
        # double sort: top-40% mom AND bottom-40% vol
        top_m = set(order.index[-int(len(m) * 0.4):])
        low_v = set(vv.sort_values().index[:int(len(vv) * 0.4)])
        ds = list(top_m & low_v)
        picks["mom_x_lowvol"] = book(ds if ds else longs)
        picks["mom_spygate"] = book(longs, cash=not spy_on)
        # own-trend: longs among names above own 200DMA
        above = own200.loc[d0]
        ot = [c for c in longs if np.isfinite(above.get(c, np.nan)) and cs.loc[d0, c] > above[c]]
        picks["mom_owntrend"] = book(ot if ot else longs)
        picks["EW-universe"] = book(list(fwdv.index))
        picks["SPY"] = (float(spy.loc[d1] / spy.loc[d0] - 1.0), set())

        for vname, (r, held) in picks.items():
            to = len(held ^ prev[vname]) / max(len(held) + len(prev[vname]), 1) if held or prev[vname] else 0.0
            ser[vname].append(r - to * COST / 1e4)
            prev[vname] = held
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print("\nMomentum + crash-filter variants, S&P 500 cross-section, monthly, net@10bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==        {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for v in variants:
            sh, cg, dd = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "EW-universe" else "  "
            print(f"  {v:14s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
