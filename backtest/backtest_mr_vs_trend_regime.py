"""
When should the book overweight MEAN-REVERSION vs TREND?

Uses the production router's own legs (backtest_mom_vs_rev.router - exact
portfolio.py rules) on TREND_ASSETS, monthly, 14y. Each month both legs are
formed at d0 and held to d1; an empty leg holds BIL. The question is whether
any observable state at d0 predicts the forward SPREAD (REV minus MOM).

Candidate conditioning variables, all computable at d0:
    med_er       median 20d Kaufman ER across the universe (the router's own
                 regime measure, aggregated to book level)
    pct_chop     fraction of names with ER <= 0.35
    vix_z        VIX z vs 50d
    vix_ts       VIX / VIX3M term structure
    spy_vs200    SPY distance from its 200DMA
    mom_trail3   the MOM leg's own trailing 3m compounded return
    mom_vol6     the MOM leg's trailing 6m realised vol (annualised)

PRE-COMMITTED BAR for "overweight MR when X": across terciles of X the forward
spread must be monotonic, the extreme-tercile gap must exceed 50bp/month, and
the SIGN of the effect must hold in both halves. Monthly periods are already
non-overlapping, so no overlap correction is needed - but n(months) is ~170 and
n per tercile ~57, so borderline results are noise and will be called noise.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
import backtest.backtest_mom_vs_rev as M

YEARS = 15


def main():
    t0 = time.time()
    syms = sorted(set(TREND_ASSETS) | {"SPY", "BIL", "^VIX", "^VIX3M"})
    px = yf.download(syms, period=f"{YEARS}y", auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    spy, bil, vix = px["SPY"].dropna(), px["BIL"].dropna(), px["^VIX"].dropna()
    v3 = px["^VIX3M"].dropna() if "^VIX3M" in px.columns else pd.Series(dtype=float)
    spy200 = spy.rolling(200).mean()
    vz = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    assets = [s for s in TREND_ASSETS if s in px.columns]

    idx = px.index
    ends = px.resample("ME").last().index
    dates = sorted({idx[idx.searchsorted(d, side="right") - 1] for d in ends
                    if idx.searchsorted(d, side="right") - 1 > 260})

    def er20(c):
        w = c.tail(21).dropna()
        if len(w) < 21:
            return np.nan
        path = w.diff().abs().sum()
        return abs(w.iloc[-1] - w.iloc[0]) / path if path > 0 else 0.0

    rows, mom_hist = [], []
    for k in range(len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        ro = bool(spy.asof(d0) < spy200.asof(d0))
        mom, rev = M.router(px, d0, ro)
        cash = float(bil.asof(d1) / bil.asof(d0) - 1.0)

        def leg(sel):
            sel = [s for s in sel if np.isfinite(px[s].asof(d0)) and np.isfinite(px[s].asof(d1))]
            if not sel:
                return cash
            return float(np.mean([px[s].asof(d1) / px[s].asof(d0) - 1.0 for s in sel]))

        r_mom, r_rev = leg(mom), leg(rev)
        ers = [er20(px[s].loc[:d0]) for s in assets]
        ers = [e for e in ers if np.isfinite(e)]
        mom_hist.append(r_mom)
        trail3 = float(np.prod([1 + x for x in mom_hist[-4:-1]]) - 1) if len(mom_hist) > 3 else np.nan
        vol6 = float(np.std(mom_hist[-7:-1], ddof=1) * np.sqrt(12)) if len(mom_hist) > 6 else np.nan
        rows.append({
            "date": d1, "r_mom": r_mom, "r_rev": r_rev, "spread": r_rev - r_mom,
            "n_mom": len(mom), "n_rev": len(rev),
            "med_er": float(np.median(ers)) if ers else np.nan,
            "pct_chop": float(np.mean([e <= 0.35 for e in ers])) if ers else np.nan,
            "vix_z": float(vz.asof(d0)) if np.isfinite(vz.asof(d0)) else np.nan,
            "vix_ts": (float(vix.asof(d0) / v3.asof(d0))
                       if len(v3) and np.isfinite(v3.asof(d0)) else np.nan),
            "spy_vs200": float(spy.asof(d0) / spy200.asof(d0) - 1.0),
            "mom_trail3": trail3, "mom_vol6": vol6,
        })

    df = pd.DataFrame(rows).set_index("date")
    print(f"\n{len(df)} monthly periods {df.index[0].date()}..{df.index[-1].date()}")
    print(f"legs: MOM mean {df.r_mom.mean():+.2%}/mo  REV mean {df.r_rev.mean():+.2%}/mo  "
          f"spread mean {df.spread.mean():+.2%}/mo")
    print(f"REV leg held >=1 name in {int((df.n_rev > 0).sum())}/{len(df)} months "
          f"(else it IS cash - 'overweight MR' usually means 'hold more cash')\n")

    half = df.index[len(df) // 2]
    print(f"{'variable':>11} | tercile fwd SPREAD (REV-MOM) %/mo  lo/mid/hi | "
          f"gap | mono | sign holds h1/h2")
    verdicts = []
    for c in ("med_er", "pct_chop", "vix_z", "vix_ts", "spy_vs200", "mom_trail3", "mom_vol6"):
        s = df.dropna(subset=[c])
        if len(s) < 90:
            print(f"{c:>11} | n={len(s)} too few")
            continue
        q = s[c].quantile([1 / 3, 2 / 3]).values
        t = [s[s[c] <= q[0]], s[(s[c] > q[0]) & (s[c] <= q[1])], s[s[c] > q[1]]]
        m = [x.spread.mean() for x in t]
        gap = m[2] - m[0]
        mono = (m[0] <= m[1] <= m[2]) or (m[0] >= m[1] >= m[2])
        h1 = s[s.index <= half]; h2 = s[s.index > half]
        def tgap(x):
            if len(x) < 30:
                return np.nan
            qq = x[c].quantile([1/3, 2/3]).values
            return x[x[c] > qq[1]].spread.mean() - x[x[c] <= qq[0]].spread.mean()
        g1, g2 = tgap(h1), tgap(h2)
        sign_ok = np.isfinite(g1) and np.isfinite(g2) and np.sign(g1) == np.sign(g2) == np.sign(gap)
        passed = mono and abs(gap) > 0.005 and sign_ok
        verdicts.append((c, passed, gap))
        print(f"{c:>11} | {m[0]:>+7.2%} {m[1]:>+7.2%} {m[2]:>+7.2%} | {gap:>+7.2%} | "
              f"{'y' if mono else 'n'} | {g1:+.2%}/{g2:+.2%} {'<-- PASSES' if passed else ''}")

    print("\npre-committed bar: monotonic terciles, |gap|>0.50%/mo, sign holds both halves")
    winners = [v for v in verdicts if v[1]]
    if not winners:
        print("NO variable passes -> no regime rule; keep the fixed REV cap.")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
