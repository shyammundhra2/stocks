"""
Does the correlation-constrained DIVERSIFIED selection beat (or cost vs) the
concentrated top-10 momentum? Each month: rank by 12-1 momentum, then EITHER take
the top-10 (naive) OR greedily pick 10 with the correlation cap (no >0.60 cluster
> 2 names), equal-weight, gated to cash on SPY<200DMA, net@10bps.

  top10_naive   top-10 momentum, equal weight
  top10_div     10 momentum names, correlation-diversified (the picker's rule)
  top20_naive   top-20 momentum (the shipped sleeve, for reference)
  EW-universe / SPY

Also reports each book's realized volatility and average trailing basket
correlation, to see whether diversification lowered vol/drawdown and at what
CAGR cost. Same survivorship caveat (relative read is the trustworthy one).
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
COST, CORR_HI, CAND_K = 10.0, 0.60, 80


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 4
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    vol = d.std() * np.sqrt(12)
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min()), vol


def diversified_pick(cand, Cmat, n=10):
    """Greedy: admit a momentum-ranked name if highly-correlated (>CORR_HI) with
    <=1 already-held name; fallback-fill with least-correlated leftovers."""
    picked = []
    for t in cand:
        hp = [p for p in picked if abs(Cmat.loc[t, p]) > CORR_HI]
        if len(hp) <= 1:
            picked.append(t)
        if len(picked) == n:
            return picked
    rem = [t for t in cand if t not in picked]
    while len(picked) < n and rem:
        best = min(rem, key=lambda t: Cmat.loc[t, picked].abs().max())
        picked.append(best); rem.remove(best)
    return picked


def main():
    t0 = time.time()
    close = pd.read_parquet(CACHE)
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]; idx = cs.index
    rets = cs.pct_change()
    mom = cs.shift(21) / cs.shift(252) - 1.0
    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200)

    me = cs.resample("ME").last().index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me
           if idx.searchsorted(d, side="right") - 1 > 300]
    reb = sorted(set(reb))

    variants = ["top10_naive", "top10_div", "top20_naive", "EW-universe", "SPY"]
    ser = {v: [] for v in variants}; dts = []; prev = {v: set() for v in variants}
    div_corr = []           # avg pairwise corr of the diversified basket
    naive_corr = []

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m = mom.loc[d0]; fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 60:
            continue
        off = bool(gate_off.loc[d0]); c = float(cash_mo.loc[d1])
        order = m.sort_values(ascending=False)

        # trailing 252d returns for the top candidates -> corr, drop artifacts
        cand = list(order.index[:CAND_K])
        win = rets.loc[:d0, cand].tail(252)
        cand = [x for x in cand if win[x].abs().max() < 0.50 and win[x].notna().sum() > 200]
        Cm = win[cand].corr()

        naive10 = cand[:10] if len(cand) >= 10 else cand
        div10 = diversified_pick(cand, Cm, 10)
        naive20 = list(order.index[:20])

        def rec(name, sel):
            if off:
                r, held = c, set()
            else:
                sel = [x for x in sel if x in fwd.index]
                r, held = (fwd[sel].mean(), set(sel)) if sel else (c, set())
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1) if held or prev[name] else 0.0
            ser[name].append(r - to * COST / 1e4); prev[name] = held

        rec("top10_naive", naive10)
        rec("top10_div", div10)
        rec("top20_naive", naive20)
        # benchmarks (ungated)
        held = set(fwd.index)
        to = len(held ^ prev["EW-universe"]) / max(len(held) + len(prev["EW-universe"]), 1)
        ser["EW-universe"].append(fwd.mean() - to * COST / 1e4); prev["EW-universe"] = held
        ser["SPY"].append(float(spy.loc[d1] / spy.loc[d0] - 1.0))
        dts.append(d1)

        if not off:
            for basket, store in [(div10, div_corr), (naive10, naive_corr)]:
                sub = Cm.loc[[x for x in basket if x in Cm.index],
                             [x for x in basket if x in Cm.index]].copy()
                np.fill_diagonal(sub.values, np.nan)
                store.append(np.nanmean(sub.abs().values))

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nDiversified vs concentrated momentum (gated), net@{COST:.0f}bps")
    print(f"avg basket |corr|: diversified {np.mean(div_corr):.2f}  vs  naive top-10 {np.mean(naive_corr):.2f}\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==          {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'vol':>6s}")
        for v in variants:
            sh, cg, dd, vol = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "top10_div" else "  "
            print(f"  {v:12s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {vol:>6.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
