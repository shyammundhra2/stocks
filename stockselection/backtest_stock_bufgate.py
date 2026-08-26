"""
Refine the winner (momentum + SPY-200DMA gate): does a BUFFERED +/-3% gate
(GSS's anti-whipsaw fix) recover the calm-regime return the exact gate gave up,
and does crediting real T-bill carry (^IRX) during cash months help? Same cached
S&P 500 cross-section, monthly, net@10bps.

  mom_base    top-quintile momentum, no gate
  mom_exact   gate to T-bill when SPY < 200DMA (exact threshold)
  mom_buf3    gate to T-bill with +/-3% hysteresis (risk-off below 200DMA*0.97,
              risk-on above 200DMA*1.03) - state walked daily, read at rebalance
  EW-universe, SPY

Cash now earns ^IRX (13-wk T-bill), not 0%. A buffered gate should beat the exact
gate in 2013-19/2020-26 (fewer whipsaw round-trips) while keeping 2008 protection.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
COST, QUINTILE, BUF = 10.0, 0.20, 0.03


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
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]; idx = cs.index
    mom = cs.shift(21) / cs.shift(252) - 1.0

    # T-bill (^IRX = annualized %); monthly cash return
    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100.0 / 12.0).fillna(0.0)      # ~monthly T-bill accrual

    # daily buffered-gate state (True = risk-off / cash)
    off = False; gate_off = pd.Series(False, index=idx)
    sv, s2 = spy.values, spy200.values
    for i in range(len(idx)):
        if np.isfinite(s2[i]):
            if off and sv[i] > s2[i] * (1 + BUF):
                off = False
            elif (not off) and sv[i] < s2[i] * (1 - BUF):
                off = True
        gate_off.iloc[i] = off
    exact_off = (spy < spy200)

    me = cs.resample("ME").last().index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me
           if idx.searchsorted(d, side="right") - 1 > 260]
    reb = sorted(set(reb))

    variants = ["mom_base", "mom_exact", "mom_buf3", "EW-universe", "SPY"]
    ser = {v: [] for v in variants}; dts = []; prev = {v: set() for v in variants}
    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m = mom.loc[d0]; fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 50:
            continue
        n_q = max(int(len(m) * QUINTILE), 5)
        longs = list(m.sort_values().index[-n_q:])
        r_long = fwd[longs].mean()
        cashret = float(cash_mo.loc[d1]) if np.isfinite(cash_mo.loc[d1]) else 0.0

        def rec(name, r, held):
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1) if held or prev[name] else 0.0
            ser[name].append(r - to * COST / 1e4); prev[name] = held

        rec("mom_base", r_long, set(longs))
        rec("mom_exact", cashret if bool(exact_off.loc[d0]) else r_long,
            set() if bool(exact_off.loc[d0]) else set(longs))
        rec("mom_buf3", cashret if bool(gate_off.loc[d0]) else r_long,
            set() if bool(gate_off.loc[d0]) else set(longs))
        rec("EW-universe", fwd.mean(), set(fwd.index))
        rec("SPY", float(spy.loc[d1] / spy.loc[d0] - 1.0), set())
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nMomentum + gate (T-bill carry on cash), net@10bps, buffer +/-{BUF:.0%}\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==        {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for v in variants:
            sh, cg, dd = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "EW-universe" else "  "
            print(f"  {v:12s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
