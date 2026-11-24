"""
backtest_stock_bufgate.py only ever tested buf=0.03 against the exact gate
for the S&P 500 momentum sleeve (stockselection/predict_top_stocks.py) and
found 3% lost - "momentum's tail is sharp crashes where a lagged exit/
entry costs more than the whipsaw it saves." That file's hysteresis logic
is correct (proper sticky state machine, unlike the bug found and fixed in
backtest_gss_whipsaw_buf_sweep.py for the ETF sleeve) - but only one width
was ever tried. This sweeps several, before assuming "buffering loses" is
true at every width just because it was true at 3%.

Rebuilds the S&P 500 price cache fresh (the original's cache lived in a
different session's scratchpad, now gone) and reproduces the exact
simulation: top-quintile 12-1 momentum, monthly rebalance, net@10bps,
T-bill (^IRX) carry on cash. Same 4 windows as the original.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_DIR))

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/7df03e5f-535e-4d7f-a935-351b97f99370/scratchpad/sp500_close.parquet"
COST, QUINTILE = 10.0, 0.20
BUFS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]


def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 3
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d); cg = eq[-1] ** (12 / len(d)) - 1
    return sh, cg, float((eq / np.maximum.accumulate(eq) - 1).min())


def _load_close():
    if os.path.exists(CACHE):
        print(f"Using cached price data -> {CACHE}")
        return pd.read_parquet(CACHE)
    symbols = pd.read_csv(os.path.join(os.path.dirname(_DIR), "stockselection", "sp500.csv"))["Symbol"].tolist()
    tickers = sorted(set(symbols) | {"SPY"})
    print(f"Downloading {len(tickers)} tickers (no cache found) ...")
    raw = yf.download(tickers, start="2004-01-01", end="2026-08-21", auto_adjust=True,
                      progress=False, group_by="ticker", threads=True)
    cols = {}
    for t in tickers:
        try:
            s = raw[t]["Close"] if len(tickers) > 1 else raw["Close"]
            if not s.dropna().empty:
                cols[t] = s
        except Exception:
            continue
    close = pd.DataFrame(cols)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    close.to_parquet(CACHE)
    print(f"Cached -> {CACHE}")
    return close


def main():
    t0 = time.time()
    close = _load_close()
    spy = close["SPY"]; spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]; idx = cs.index
    mom = cs.shift(21) / cs.shift(252) - 1.0

    irx = yf.download("^IRX", start="2004-01-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100.0 / 12.0).fillna(0.0)

    sv, s2 = spy.values, spy200.values
    exact_off = (spy < spy200)

    def gate_series(buf):
        off = False; out = pd.Series(False, index=idx)
        for i in range(len(idx)):
            if np.isfinite(s2[i]):
                if off and sv[i] > s2[i] * (1 + buf):
                    off = False
                elif (not off) and sv[i] < s2[i] * (1 - buf):
                    off = True
            out.iloc[i] = off
        return out

    gates = {b: gate_series(b) for b in BUFS if b > 0}

    me = cs.resample("ME").last().index
    reb = sorted(set(idx[idx.searchsorted(d, side="right") - 1] for d in me
                     if idx.searchsorted(d, side="right") - 1 > 260))

    variants = ["mom_base", "mom_exact"] + [f"buf{int(b*100)}" for b in BUFS if b > 0]
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
        for b in BUFS:
            if b == 0:
                continue
            name = f"buf{int(b*100)}"
            off_ = bool(gates[b].loc[d0])
            rec(name, cashret if off_ else r_long, set() if off_ else set(longs))
        dts.append(d1)

    S = {v: pd.Series(ser[v], index=dts) for v in variants}
    wins = [("2005-2012", "2005-01-01", "2012-12-31"),
            ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-08-21"),
            ("FULL", "2005-01-01", "2026-08-21")]
    print(f"\nMomentum + gate (T-bill carry on cash), net@10bps\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==        {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s}")
        for v in variants:
            sh, cg, dd = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            tag = v + ("  <- current live" if v == "mom_exact" else "")
            print(f"  {tag:22s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%}")
        print()
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
