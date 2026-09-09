"""
Does RESIDUAL momentum beat raw 12-1 momentum in the sleeve?

Raw 12-1 momentum ranks on total return, which bundles two things: the stock's
own idiosyncratic run, and whatever the market did times the stock's beta. The
documented problem (Blitz/Huij/Martens; Daniel/Moskowitz on momentum crashes) is
that the beta half is what blows up - after a bear market the loser basket is
loaded with high-beta names, and the reversal rips through a momentum book.
Your 200DMA gate exists to sidestep exactly that. Residual momentum asks whether
the crash risk can be removed from the SIGNAL rather than patched by a gate.

Formation window t-252..t-21 (same 12-1 convention as production). For each name:
    beta_i   = cov(r_i, r_mkt) / var(r_mkt)      over the window
    resid_t  = r_i,t - beta_i * r_mkt,t
    RESID    = sum(resid)                        (raw residual momentum)
    RESID_T  = sum(resid) / sd(resid)            (Blitz's vol-scaled version)
    BETAMOM  = beta_i * sum(r_mkt)               (the DISCARDED half - control)

Everything downstream is production, unchanged: top-10, <=3/sector, SPY-200DMA
gate, monthly, net@10bps.

Pre-committed bar, all three, in BOTH halves:
    1. Sharpe improves
    2. CAGR falls by no more than 1pp
    3. maxDD does not worsen

Falsification control: BETAMOM is the component residual momentum throws away.
If ranking on it also wins, the result is not about isolating idiosyncratic
return and should not be believed.
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")

DIR = os.path.dirname(os.path.abspath(__file__))
CLOSE_PQ = os.path.join(os.path.dirname(DIR), ".cache", "selbuffer", "sp500_close.parquet")
SECTORS = os.path.join(os.path.dirname(DIR), "stockselection", "sp500_sectors.csv")

COST_BPS = 10.0
N, MAX_PER_SEC = 10, 3
MOM_CAP, JUMP_CAP = 4.0, 0.50
LOOKBACK, SKIP = 252, 21


def capped_pick(order, sec, n=N, cap=MAX_PER_SEC):
    picked, cnt = [], Counter()
    for t in order:
        s = sec.get(t, t)
        if cnt[s] < cap:
            picked.append(t); cnt[s] += 1
        if len(picked) == n:
            break
    return picked


def perf(r):
    d = r.dropna().values
    if len(d) < 12 or d.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan, vol=np.nan, calmar=np.nan)
    eq = np.cumprod(1 + d)
    cagr = eq[-1] ** (12 / len(d)) - 1
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return dict(sharpe=d.mean() / d.std() * np.sqrt(12), cagr=cagr, maxdd=mdd,
                vol=d.std() * np.sqrt(12), calmar=cagr / abs(mdd) if mdd else np.nan)


def signals(R, mkt):
    """R: (T,N) window of daily returns, mkt: (T,) market. Returns dict of (N,) scores."""
    m = mkt - mkt.mean()
    var = float(m @ m)
    beta = (R - R.mean(0)).T @ m / var if var > 0 else np.zeros(R.shape[1])
    resid = R - np.outer(mkt, beta)
    rs = resid.sum(0)
    sd = resid.std(0, ddof=1)
    return {
        "raw": R.sum(0),
        "resid": rs,
        "resid_t": np.divide(rs, sd, out=np.zeros_like(rs), where=sd > 0),
        "betamom": beta * float(mkt.sum()),
    }


def main():
    t0 = time.time()
    close = pd.read_parquet(CLOSE_PQ)
    sdf = pd.read_csv(SECTORS)
    sec = dict(zip(sdf["Symbol"], sdf["Sector" if "Sector" in sdf.columns else sdf.columns[-1]]))

    spy = close["SPY"].dropna()
    spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]
    idx = cs.index
    rets = cs.pct_change()
    mkt = spy.pct_change().reindex(idx)
    raw_mom = cs.shift(SKIP) / cs.shift(LOOKBACK) - 1.0

    irx = yf.download("^IRX", start="2004-01-01", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200).reindex(idx).ffill().fillna(False)

    me = cs.resample("ME").last().index
    reb = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in me
                  if idx.searchsorted(d, side="right") - 1 > 300})

    names = ["raw", "resid", "resid_t", "betamom"]
    ser = {k: [] for k in names}
    prev = {k: set() for k in names}
    dts = []

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        i0 = idx.get_loc(d0)
        fwd = cs.loc[d1] / cs.loc[d0] - 1.0
        rm = raw_mom.loc[d0]
        ok = rm.notna() & fwd.notna() & np.isfinite(rm) & np.isfinite(fwd)
        if ok.sum() < 60:
            continue

        win = rets.iloc[i0 - LOOKBACK + 1: i0 - SKIP + 1]           # formation window
        mw = mkt.iloc[i0 - LOOKBACK + 1: i0 - SKIP + 1]
        valid = ok & win.notna().all() & mw.notna().all()
        cols = [c for c in cs.columns if valid.get(c, False)]
        # production artifact filters, applied identically to every variant
        recent = rets.loc[:d0].tail(LOOKBACK)
        cols = [c for c in cols if recent[c].abs().max() < JUMP_CAP and 0 < rm[c] < MOM_CAP]
        if len(cols) < 40:
            continue

        sig = signals(win[cols].values, mw.values)
        c_ = float(cash_mo.loc[d1])
        off = bool(gate_off.loc[d0])
        for nm in names:
            if off:
                r, held = c_, set()
            else:
                order = [cols[i] for i in np.argsort(-sig[nm])]
                selq = [x for x in capped_pick(order, sec) if x in fwd.index]
                r, held = (fwd[selq].mean(), set(selq)) if selq else (c_, set())
            to = len(held ^ prev[nm]) / max(len(held) + len(prev[nm]), 1)
            ser[nm].append(r - to * COST_BPS / 1e4)
            prev[nm] = held
        dts.append(d1)

    S = {k: pd.Series(v, index=dts) for k, v in ser.items()}
    half = dts[len(dts) // 2]
    wins = [("FULL", dts[0], dts[-1]), ("first half", dts[0], half), ("second half", half, dts[-1])]
    lbl = {"raw": "raw 12-1 (base)", "resid": "residual mom",
           "resid_t": "residual mom /sd", "betamom": "BETA mom (control)"}

    print(f"\nResidual vs raw momentum ({len(dts)} monthly periods, "
          f"{dts[0].date()}..{dts[-1].date()}, net@{COST_BPS:.0f}bps)\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==            {'Sharpe':>7} {'CAGR':>8} {'maxDD':>8} {'vol':>7} {'Calmar':>7}")
        for nm in names:
            x = S[nm]; p = perf(x[(x.index >= lo) & (x.index <= hi)])
            print(f"  {lbl[nm]:20s} {p['sharpe']:>7.2f} {p['cagr']:>7.1%} {p['maxdd']:>8.1%} "
                  f"{p['vol']:>7.1%} {p['calmar']:>7.2f}")
        print()

    print("--- pre-committed bar (Sharpe up, CAGR within 1pp, maxDD not worse, BOTH halves) ---")
    for nm in names[1:]:
        allok = True
        for lab, lo, hi in wins[1:]:
            b = perf(S["raw"][(S["raw"].index >= lo) & (S["raw"].index <= hi)])
            v = perf(S[nm][(S[nm].index >= lo) & (S[nm].index <= hi)])
            c1 = v["sharpe"] > b["sharpe"]
            c2 = v["cagr"] >= b["cagr"] - 0.01
            c3 = abs(v["maxdd"]) <= abs(b["maxdd"])
            allok &= c1 and c2 and c3
            print(f"  {lbl[nm]:20s} {lab:12s} Sharpe {b['sharpe']:.2f}->{v['sharpe']:.2f} {'y' if c1 else 'n'} | "
                  f"CAGR {b['cagr']:.1%}->{v['cagr']:.1%} {'y' if c2 else 'n'} | "
                  f"maxDD {b['maxdd']:.1%}->{v['maxdd']:.1%} {'y' if c3 else 'n'}")
        print(f"  -> {lbl[nm]}: {'PASSES' if allok else 'fails'}\n")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
