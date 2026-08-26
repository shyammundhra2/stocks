"""
Diversify the momentum book by a SECTOR CAP (<=3 names per GICS sector) instead
of correlation-picking. Cleaner and more interpretable. Shows (1) the current
10-name pick and (2) a backtest of the sector-capped rule vs naive top-10.

Each month: rank by 12-1 momentum, greedily admit a name if its GICS sector has
< MAX_PER_SEC already picked, until N. Equal-weight, gated to cash on SPY<200DMA,
net@10bps. Sectors from the cached Wikipedia GICS table (roughly static - minor
lookahead, sectors rarely change).
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
SECMAP = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_sectors.parquet"
NAMES = "/Users/riddhisiddhi/stocks/stockselection/sp500.csv"
COST, MAX_PER_SEC, N = 10.0, 3, 10


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 4
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min()), d.std() * np.sqrt(12)


def sector_pick(order, sec, n=N, cap=MAX_PER_SEC):
    """Momentum-ranked walk; admit if the name's sector has < cap already picked."""
    picked, cnt = [], {}
    for t in order:
        s = sec.get(t, t)
        if cnt.get(s, 0) < cap:
            picked.append(t); cnt[s] = cnt.get(s, 0) + 1
        if len(picked) == n:
            break
    return picked


def main():
    t0 = time.time()
    close = pd.read_parquet(CACHE)
    sec = pd.read_parquet(SECMAP).set_index("Symbol")["GICS Sector"].to_dict()
    names = dict(zip(*pd.read_csv(NAMES).values.T))
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]; idx = cs.index
    rets = cs.pct_change()
    mom = cs.shift(21) / cs.shift(252) - 1.0

    # ---- current pick ----
    dclean = rets.tail(252)
    keep = [c for c in stocks if dclean[c].abs().max() < 0.50]           # drop artifacts
    m_now = (cs[keep].shift(21).iloc[-1] / cs[keep].shift(252).iloc[-1] - 1.0)
    m_now = m_now[(m_now > 0) & (m_now < 4.0)].sort_values(ascending=False)
    pick = sector_pick(list(m_now.index), sec)
    print(f"\nCURRENT 10-name pick (<= {MAX_PER_SEC}/sector), as of {idx[-1].date()}\n")
    print(f"{'ticker':6s} {'12-1 mom':>9s}  {'sector':22s} name")
    print("-" * 70)
    for t in pick:
        print(f"{t:6s} {m_now[t]:>+8.1%}  {str(sec.get(t,'?')):22s} {str(names.get(t,t))[:26]}")
    from collections import Counter
    print("sector mix:", dict(Counter(sec.get(t, '?') for t in pick)))

    # ---- backtest ----
    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=idx if False else irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200)
    me = cs.resample("ME").last().index
    reb = [idx[idx.searchsorted(d, side="right") - 1] for d in me
           if idx.searchsorted(d, side="right") - 1 > 300]
    reb = sorted(set(reb))

    variants = ["sectorcap10", "naive10", "naive20", "EW-universe", "SPY"]
    ser = {v: [] for v in variants}; dts = []; prev = {v: set() for v in variants}
    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m = mom.loc[d0]; fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 60:
            continue
        off = bool(gate_off.loc[d0]); c = float(cash_mo.loc[d1])
        win = rets.loc[:d0].tail(252)
        order = [t for t in m.sort_values(ascending=False).index
                 if win[t].abs().max() < 0.50 and (0 < m[t] < 4.0)]

        def rec(name, sel):
            if off:
                r, held = c, set()
            else:
                sel = [x for x in sel if x in fwd.index]
                r, held = (fwd[sel].mean(), set(sel)) if sel else (c, set())
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1) if held or prev[name] else 0.0
            ser[name].append(r - to * COST / 1e4); prev[name] = held

        rec("sectorcap10", sector_pick(order, sec, N, MAX_PER_SEC))
        rec("naive10", order[:10])
        rec("naive20", order[:20])
        held = set(fwd.index)
        to = len(held ^ prev["EW-universe"]) / max(len(held) + len(prev["EW-universe"]), 1)
        ser["EW-universe"].append(fwd.mean() - to * COST / 1e4); prev["EW-universe"] = held
        ser["SPY"].append(float(spy.loc[d1] / spy.loc[d0] - 1.0))
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"), ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nSector-capped(<= {MAX_PER_SEC}) vs naive momentum (gated), net@{COST:.0f}bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==           {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'vol':>6s}")
        for v in variants:
            sh, cg, dd, vol = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            star = " *" if v == "sectorcap10" else "  "
            print(f"  {v:12s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {vol:>6.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
