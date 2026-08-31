"""
Does SELECTION HYSTERESIS cut the momentum sleeve's turnover without costing return?

WHY. predict_top_stocks.py rebuilds the book from scratch every run against a hard
TOP_N=10 cutoff, and the book churns hard for reasons that have nothing to do with new
information. Measured on the live universe (2026-08-31):

  * median 1-day change in 12-1 momentum is 1.39%, while the gap deciding slots 11 vs 12
    was 0.34% - the noise is ~4x the decision margin, so membership flips are guaranteed.
  * HALF that daily movement comes from the BACK of the window: the median 1-day move of
    the year-old bar dropping out (0.97%) is as large as the recent bar coming in (0.99%),
    and for 50% of names the year-old bar moved more. The book churns on data a year old.
  * MAX_PER_SECTOR=3 makes the walk reach to rank ~19 to fill 10 slots (top momentum is
    nearly all Info Tech), and those deep slots sit in the dense part of the distribution
    where names trade places constantly. Measured turnover over 42 days was 5/10 names
    with the cap vs 2/10 without.

THE FIX TESTED HERE. Separate add and drop thresholds: buy into the top N, but hold until
a name falls out of the top B. Index providers use exactly this on momentum indices.

NOT THE SAME AS THE GATE BUFFER. predict_top_stocks.py's docstring notes that buffering
LOST - but that was buffering the SPY-200DMA gate, where a lagged exit is expensive
because momentum crashes are sharp. This buffers SELECTION and leaves the crash gate
exact. Different mechanism, untested until now.

Reports turnover explicitly, which the other backtests compute for costs but never print.
"""
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import yfinance as yf

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)
CACHE_DIR = os.path.join(ROOT, ".cache", "selbuffer")
CLOSE_PQ = os.path.join(CACHE_DIR, "sp500_close.parquet")
NAMES = os.path.join(DIR, "sp500.csv")
SECTORS = os.path.join(DIR, "sp500_sectors.csv")

COST_BPS = 10.0
N = 10                 # target book size
MAX_PER_SEC = 3
START = "2004-01-01"
MOM_CAP, JUMP_CAP = 4.0, 0.50


# --------------------------------------------------------------------------- data
def build_cache() -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    syms = pd.read_csv(NAMES)["Symbol"].tolist() + ["SPY"]
    frames, B = {}, 50
    for i in range(0, len(syms), B):
        batch = syms[i:i + B]
        print(f"  batch {i//B+1}/{(len(syms)+B-1)//B} ...", flush=True)
        for _ in range(3):
            try:
                d = yf.download(batch, start=START, auto_adjust=True, progress=False,
                                group_by="ticker", threads=True)
                for t in batch:
                    try:
                        s = (d[t]["Close"] if len(batch) > 1 else d["Close"]).dropna()
                        if len(s) > 300:
                            frames[t] = s
                    except Exception:
                        continue
                break
            except Exception:
                time.sleep(2)
    close = pd.DataFrame(frames).sort_index()
    close.to_parquet(CLOSE_PQ)
    return close


def load_close(refresh=False) -> pd.DataFrame:
    if refresh or not os.path.exists(CLOSE_PQ):
        print("building price cache (one-off, ~2-4 min) ...")
        return build_cache()
    return pd.read_parquet(CLOSE_PQ)


def load_sectors() -> dict:
    d = pd.read_csv(SECTORS)
    col = "Sector" if "Sector" in d.columns else d.columns[-1]
    return dict(zip(d["Symbol"], d[col]))


# ---------------------------------------------------------------------- selection
def capped_pick(order, sec, n=N, cap=MAX_PER_SEC):
    """Momentum-ranked walk, admitting a name only if its sector is under the cap."""
    picked, cnt = [], Counter()
    for t in order:
        s = sec.get(t, t)
        if cap is None or cnt[s] < cap:
            picked.append(t); cnt[s] += 1
        if len(picked) == n:
            break
    return picked


def buffered_pick(order, sec, prev, n=N, buf=20, cap=MAX_PER_SEC, cap_on_entry=False):
    """Hold anything still ranked inside `buf`; fill free slots from the top.

    cap_on_entry=True applies the sector cap only to NEW names, so an existing holding is
    never evicted merely because two sector-mates out-ranked it this month.
    """
    rank = {t: i for i, t in enumerate(order)}
    keep = [t for t in order if t in prev and rank[t] < buf]     # order preserves momentum rank
    cnt = Counter(sec.get(t, t) for t in keep)
    picked = list(keep[:n])
    if not cap_on_entry:                     # re-apply the cap to survivors too
        picked, cnt = [], Counter()
        for t in keep:
            s = sec.get(t, t)
            if cap is None or cnt[s] < cap:
                picked.append(t); cnt[s] += 1
            if len(picked) == n:
                break
    for t in order:
        if len(picked) >= n:
            break
        if t in picked:
            continue
        s = sec.get(t, t)
        if cap is None or cnt[s] < cap:
            picked.append(t); cnt[s] += 1
    return picked[:n]


# ------------------------------------------------------------------------ metrics
def perf(r, lo, hi):
    d = r[(r.index >= lo) & (r.index <= hi)].dropna().values
    if len(d) < 12 or d.std() == 0:
        return (np.nan,) * 4
    sh = d.mean() / d.std() * np.sqrt(12)
    eq = np.cumprod(1 + d)
    return sh, eq[-1] ** (12 / len(d)) - 1, float((eq / np.maximum.accumulate(eq) - 1).min()), d.std() * np.sqrt(12)


def main(refresh=False):
    t0 = time.time()
    close = load_close(refresh)
    sec = load_sectors()
    spy = close["SPY"].dropna()
    spy200 = spy.rolling(200).mean()
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]
    idx = cs.index
    rets = cs.pct_change()
    mom = cs.shift(21) / cs.shift(252) - 1.0

    irx = yf.download("^IRX", start=START, progress=False, auto_adjust=True)["Close"]
    irx = pd.Series(np.asarray(irx).ravel(), index=irx.index).reindex(idx).ffill()
    cash_mo = (irx / 100 / 12).fillna(0.0)
    gate_off = (spy < spy200).reindex(idx).ffill().fillna(False)

    me = cs.resample("ME").last().index
    reb = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in me
                  if idx.searchsorted(d, side="right") - 1 > 300})

    # name -> (selector, needs_prev)
    V = {
        "rebuild (current)": (lambda o, p: capped_pick(o, sec), False),
        "buffer 15":         (lambda o, p: buffered_pick(o, sec, p, buf=15), True),
        "buffer 20":         (lambda o, p: buffered_pick(o, sec, p, buf=20), True),
        "buffer 30":         (lambda o, p: buffered_pick(o, sec, p, buf=30), True),
        "buf20 cap-on-entry": (lambda o, p: buffered_pick(o, sec, p, buf=20, cap_on_entry=True), True),
        "naive10 (no cap)":  (lambda o, p: o[:N], False),
    }
    ser = {v: [] for v in V} | {"EW-universe": [], "SPY": []}
    turn = {v: [] for v in V}
    prev = {v: set() for v in list(V) + ["EW-universe"]}
    dts = []

    for k in range(len(reb) - 1):
        d0, d1 = reb[k], reb[k + 1]
        m, fwd = mom.loc[d0], cs.loc[d1] / cs.loc[d0] - 1.0
        ok = m.notna() & fwd.notna() & np.isfinite(m) & np.isfinite(fwd)
        m, fwd = m[ok], fwd[ok]
        if len(m) < 60:
            continue
        off, c = bool(gate_off.loc[d0]), float(cash_mo.loc[d1])
        win = rets.loc[:d0].tail(252)
        order = [t for t in m.sort_values(ascending=False).index
                 if win[t].abs().max() < JUMP_CAP and 0 < m[t] < MOM_CAP]

        for name, (fn, _) in V.items():
            if off:
                r, held = c, set()
            else:
                sel = [x for x in fn(order, prev[name]) if x in fwd.index]
                r, held = (fwd[sel].mean(), set(sel)) if sel else (c, set())
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1)
            ser[name].append(r - to * COST_BPS / 1e4)
            turn[name].append(to)
            prev[name] = held

        held = set(fwd.index)
        to = len(held ^ prev["EW-universe"]) / max(len(held) + len(prev["EW-universe"]), 1)
        ser["EW-universe"].append(fwd.mean() - to * COST_BPS / 1e4)
        prev["EW-universe"] = held
        ser["SPY"].append(float(spy.loc[d1] / spy.loc[d0] - 1.0))
        dts.append(d1)

    S = {v: pd.Series(x, index=dts) for v, x in ser.items()}
    variants = list(V) + ["EW-universe", "SPY"]
    wins = [("2005-2012", "2005-01-01", "2012-12-31"), ("2013-2019", "2013-01-01", "2019-12-31"),
            ("2020-2026", "2020-01-01", "2026-12-31"), ("FULL", "2005-01-01", "2026-12-31")]

    print(f"\nSelection hysteresis on the momentum sleeve (N={N}, <={MAX_PER_SEC}/sector, "
          f"gated, net@{COST_BPS:.0f}bps)\n")
    for lab, lo, hi in wins:
        print(f"== {lab} ==            {'Sharpe':>7} {'CAGR':>7} {'maxDD':>7} {'vol':>6} {'turn/yr':>8}")
        for v in variants:
            sh, cg, dd, vol = perf(S[v], pd.Timestamp(lo), pd.Timestamp(hi))
            t_ = f"{np.mean(turn[v])*12:>7.0%}" if v in turn else "      -"
            star = " *" if v.startswith("buffer 20") else "  "
            print(f"  {v:20s}{star} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {vol:>6.1%} {t_:>8}")
        print()
    print(f"turnover is two-way (symmetric difference / total), annualised from monthly rebalances")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
