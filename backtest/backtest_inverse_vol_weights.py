"""
Does INVERSE-VOL weighting inside the book beat equal-weight?

The sleeve holds 10 names equal-weight, but they are not the same risk - a semi
and a utility get identical dollars. This tests sizing each name by 1/sigma
instead. Same names, same gate, same sector cap, same total exposure: this
changes ALLOCATION WITHIN the book only, never how much of the book is on.

Pre-committed bar (fixed before looking at any result), all three, in BOTH halves:
    1. Sharpe improves
    2. CAGR falls by no more than 1pp
    3. maxDD does not worsen

Falsification control: the OPPOSITE weighting (w proportional to sigma) should
lose roughly symmetrically. If both directions win, whatever is happening is not
risk balancing and the result should not be believed.

Equal-risk view is also reported. Inverse-vol lowers the book's volatility, so a
raw maxDD comparison is partly just "less risk"; scaling the variant to the
base's realised vol asks the fairer question - at the SAME risk, which returns
more? (That framing came from the user's correction on the vol-managed test.)
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
VOL_WINDOWS = (20, 63, 126, 252)
PRIMARY_VOL = 63          # matches the vol63 field get_trends already computes


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
    mom = cs.shift(21) / cs.shift(252) - 1.0
    vols = {w: rets.rolling(w).std() for w in VOL_WINDOWS}

    irx = yf.download("^IRX", start="2004-01-01", progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200).reindex(idx).ffill().fillna(False)

    me = cs.resample("ME").last().index
    reb = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in me
                  if idx.searchsorted(d, side="right") - 1 > 300})

    schemes = ["equal"] + [f"invvol{w}" for w in VOL_WINDOWS] + [f"volwt{PRIMARY_VOL}", "random"]
    ser = {s: [] for s in schemes}
    prev = {s: {} for s in schemes}
    dts = []
    rng = np.random.default_rng(0)

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m, fwd = mom.loc[d0], cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 60:
            continue
        c = float(cash_mo.loc[d1])
        off = bool(gate_off.loc[d0])
        if off:
            sel = []
        else:
            win = rets.loc[:d0].tail(252)
            order = [t for t in m.sort_values(ascending=False).index
                     if win[t].abs().max() < JUMP_CAP and 0 < m[t] < MOM_CAP]
            sel = [x for x in capped_pick(order, sec) if x in fwd.index]

        for s in schemes:
            if not sel:
                r, w = c, {}
            else:
                if s == "equal":
                    raw = np.ones(len(sel))
                elif s == "random":
                    raw = rng.random(len(sel)) + 0.1
                else:
                    win_n = int(s.replace("invvol", "").replace("volwt", ""))
                    sig = np.array([vols[win_n].loc[d0, t] for t in sel], dtype=float)
                    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig))
                    raw = sig if s.startswith("volwt") else 1.0 / sig
                w_ = raw / raw.sum()
                r = float(np.dot(w_, [fwd[t] for t in sel]))
                w = dict(zip(sel, w_))
            # turnover on WEIGHTS, not just membership - reweighting costs too
            keys = set(w) | set(prev[s])
            to = sum(abs(w.get(t, 0.0) - prev[s].get(t, 0.0)) for t in keys)
            ser[s].append(r - to * COST_BPS / 1e4)
            prev[s] = w
        dts.append(d1)

    S = {s: pd.Series(ser[s], index=dts) for s in schemes}
    half = dts[len(dts) // 2]
    wins = [("FULL", dts[0], dts[-1]), ("first half", dts[0], half), ("second half", half, dts[-1])]

    print(f"\nInverse-vol weighting inside the book "
          f"({len(dts)} monthly periods, {dts[0].date()}..{dts[-1].date()}, net@{COST_BPS:.0f}bps)")
    print("weights sum to 1 always - this changes allocation WITHIN the book, not exposure\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==          {'Sharpe':>7} {'CAGR':>8} {'maxDD':>8} {'vol':>7} {'Calmar':>7}")
        for s in schemes:
            x = S[s]
            p = perf(x[(x.index >= lo) & (x.index <= hi)])
            print(f"  {s:16s} {p['sharpe']:>7.2f} {p['cagr']:>7.1%} {p['maxdd']:>8.1%} "
                  f"{p['vol']:>7.1%} {p['calmar']:>7.2f}")
        print()

    print("--- pre-committed bar (Sharpe up, CAGR within 1pp, maxDD not worse, BOTH halves) ---")
    for s in schemes[1:]:
        allok = True
        for lab, lo, hi in wins[1:]:
            b = perf(S["equal"][(S["equal"].index >= lo) & (S["equal"].index <= hi)])
            v = perf(S[s][(S[s].index >= lo) & (S[s].index <= hi)])
            c1, c2, c3 = v["sharpe"] > b["sharpe"], v["cagr"] >= b["cagr"] - 0.01, \
                abs(v["maxdd"]) <= abs(b["maxdd"])
            allok &= c1 and c2 and c3
            print(f"  {s:16s} {lab:12s} Sharpe {b['sharpe']:.2f}->{v['sharpe']:.2f} {'y' if c1 else 'n'} | "
                  f"CAGR {b['cagr']:.1%}->{v['cagr']:.1%} {'y' if c2 else 'n'} | "
                  f"maxDD {b['maxdd']:.1%}->{v['maxdd']:.1%} {'y' if c3 else 'n'}")
        print(f"  -> {s}: {'PASSES' if allok else 'fails'}\n")

    print("--- equal-risk view (scale variant to the base's realised vol, full sample) ---")
    b = S["equal"]; pb = perf(b)
    for s in schemes[1:]:
        v = S[s]; pv = perf(v)
        k = b.std() / v.std()
        pl = perf(k * v - (k - 1) * (pd.Series(0.0, index=v.index)))   # unfunded scale, no borrow cost
        print(f"  {s:16s} at {k:.2f}x -> CAGR {pl['cagr']:>6.1%} (base {pb['cagr']:.1%})  "
              f"maxDD {pl['maxdd']:>7.1%} (base {pb['maxdd']:.1%})  "
              f"{'better on both' if pl['cagr'] > pb['cagr'] and abs(pl['maxdd']) < abs(pb['maxdd']) else ''}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
