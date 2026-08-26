"""
Should the gated momentum book use FEWER buys (concentration) and/or Kelly/
conviction sizing instead of equal-weight top quintile (~100 names)? All gated
(SPY-200DMA, T-bill cash), 12-1 momentum, monthly, net@10bps.

Concentration (equal-weight top-N by momentum):  N = 100, 50, 30, 20, 10
Sizing on the top quintile:
  ew        equal weight (the shipped rule)
  rankwt    weight by momentum PERCENTILE rank (Kelly-flavored: bigger edge ->
            bigger bet)
  ivol      inverse-vol weight (fractional-Kelly / risk parity)

Honest prior: fewer names -> more return in calm regimes but MORE estimation
noise and WORSE crash risk (the hottest momentum crashes hardest); Kelly needs
edge/odds estimates we don't reliably have. Let the drawdown column decide.
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
COST = 10.0


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
    vol = np.log(cs).diff().rolling(126).std()
    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200)

    me = cs.resample("ME").last().index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me
           if idx.searchsorted(d, side="right") - 1 > 260]
    reb = sorted(set(reb))

    variants = ["N100_ew", "N50_ew", "N30_ew", "N20_ew", "N10_ew",
                "Q_rankwt", "Q_ivol", "EW-universe"]
    ser = {v: [] for v in variants}; dts = []; prev = {v: set() for v in variants}

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m = mom.loc[d0]; fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 50:
            continue
        off = bool(gate_off.loc[d0]); c = float(cash_mo.loc[d1])
        order = m.sort_values()

        def rec(name, r, held):
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1) if held or prev[name] else 0.0
            ser[name].append(r - to * COST / 1e4); prev[name] = held

        def eq_topN(N):
            if off:
                return c, set()
            sel = list(order.index[-N:])
            return fwd[sel].mean(), set(sel)

        for N, nm in [(100, "N100_ew"), (50, "N50_ew"), (30, "N30_ew"),
                      (20, "N20_ew"), (10, "N10_ew")]:
            rec(nm, *eq_topN(N))

        # top quintile with weighting
        nq = max(int(len(m) * 0.20), 5)
        longs = list(order.index[-nq:])
        if off:
            rec("Q_rankwt", c, set()); rec("Q_ivol", c, set())
        else:
            # rank weight: momentum percentile within the selected set
            rk = m[longs].rank(); w = rk / rk.sum()
            rec("Q_rankwt", float((fwd[longs] * w).sum()), set(longs))
            iv = (1.0 / vol.loc[d0][longs]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(iv):
                w2 = iv / iv.sum()
                rec("Q_ivol", float((fwd[iv.index] * w2).sum()), set(iv.index))
            else:
                rec("Q_ivol", fwd[longs].mean(), set(longs))

        held = set(fwd.index)
        rec("EW-universe", fwd.mean(), held)   # ungated benchmark (gate n/a)
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nConcentration & Kelly sizing on the gated momentum book, net@10bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==         {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for v in variants:
            sh, cg, dd = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "N100_ew" else "  "
            print(f"  {v:12s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
