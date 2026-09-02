"""
MOM vs REV: which leg of the adaptive router actually earned the return?

get_trends() routes each asset to MOM / REV / FLAT every day, but the two legs
are then blended by the optimizer, so the live book never shows which one is
carrying it. This runs the router's EXACT rules (portfolio.py:669-690) over the
trailing 5 years on the same TREND_ASSETS universe and reports each leg alone.

Router, per asset per date (production, reproduced verbatim):
    eff_ratio = Kaufman ER(20)
    ER >= 0.40  -> TREND: MOM if (px>200SMA and px>50SMA and slope>0) else FLAT
    ER <= 0.35  -> CHOP : REV if (px>200SMA and z20<-1.0 and outside_curves) else FLAT
    else        -> MID  : FLAT
where z20 = 20d return / (20d vol * sqrt(20)), and outside_curves is
|slope|*r2 < 0.18 * max(max|slope| across the universe, 1.5) on that date.

Each leg is held equal-weight to the next rebalance; idle capital earns BIL.
The production risk-off gate (SPY < 200DMA -> flat, defensives spared) is applied
to BOTH legs, since it is common to them and not part of what is being compared.

Costs at COST_BPS per unit of two-way turnover, same convention as the other
stockselection backtests.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS, DEFENSIVE_ASSETS
from macro.indicators.mathstats import _trend_stats, _efficiency_ratio

YEARS = 5
COST_BPS = 10.0
REBAL = "ME"          # month-end, matching the router's ~20d intended hold


def download():
    syms = sorted(set(list(TREND_ASSETS) + ["SPY", "BIL"]))
    px = yf.download(syms, period=f"{YEARS + 2}y", auto_adjust=True,
                     progress=False)["Close"]
    return px.dropna(how="all")


def router(close, date, spy_riskoff):
    """Return (mom, rev) ticker lists exactly as production would on `date`."""
    hist = close.loc[:date]
    if len(hist) < 210:
        return [], []

    # cross-sectional conviction, needed for the REV "outside the curves" test
    conv, abs_slopes, stats = {}, [], {}
    for s in TREND_ASSETS:
        if s not in hist.columns:
            continue
        c = hist[s].dropna()
        if len(c) < 210:
            continue
        slope, r2 = _trend_stats(c, 20, 10)
        stats[s] = (c, slope, r2)
        conv[s] = abs(slope) * r2
        abs_slopes.append(abs(slope))
    if not stats:
        return [], []
    rev_k1 = 0.18 * max(max(abs_slopes, default=0.0), 1.5)

    mom, rev = [], []
    for s, (c, slope, r2) in stats.items():
        last = float(c.iloc[-1])
        s50 = float(c.rolling(50).mean().iloc[-1])
        s200 = float(c.rolling(200).mean().iloc[-1])
        above200 = last > s200

        ret20 = last / float(c.iloc[-21]) - 1.0
        vol20 = float(c.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(20)
        z20 = ret20 / vol20 if (np.isfinite(vol20) and vol20 > 0) else 0.0

        er = _efficiency_ratio(c, 20)
        if er >= 0.40:
            sig = "MOM" if (above200 and last > s50 and slope > 0) else "FLAT"
        elif er <= 0.35:
            sig = ("REV" if (above200 and z20 < -1.0 and conv.get(s, np.inf) < rev_k1)
                   else "FLAT")
        else:
            sig = "FLAT"

        # production risk-off gate, applied to both legs alike
        if spy_riskoff and s not in DEFENSIVE_ASSETS and sig != "FLAT":
            sig = "FLAT"

        if sig == "MOM":
            mom.append(s)
        elif sig == "REV":
            rev.append(s)
    return mom, rev


def perf(r):
    d = r.dropna().values
    if len(d) < 6 or d.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan, vol=np.nan)
    eq = np.cumprod(1 + d)
    return dict(
        sharpe=d.mean() / d.std() * np.sqrt(12),
        cagr=eq[-1] ** (12 / len(d)) - 1,
        maxdd=float((eq / np.maximum.accumulate(eq) - 1).min()),
        vol=d.std() * np.sqrt(12),
    )


def main():
    t0 = time.time()
    close = download()
    idx = close.index
    spy = close["SPY"].dropna()
    spy200 = spy.rolling(200).mean()
    bil = close["BIL"].dropna()

    ends = close.resample(REBAL).last().index
    dates = [idx[idx.searchsorted(d, side="right") - 1] for d in ends]
    dates = sorted({d for d in dates if d >= idx[0]})
    start = close.index[-1] - pd.DateOffset(years=YEARS)
    dates = [d for d in dates if d >= start]

    rows, prev = [], {"MOM": set(), "REV": set(), "BOTH": set()}
    for k in range(len(dates) - 1):
        d0, d1 = dates[k], dates[k + 1]
        riskoff = bool(spy.loc[d0] < spy200.loc[d0])
        mom, rev = router(close, d0, riskoff)
        cash = float(bil.asof(d1) / bil.asof(d0) - 1.0)

        rec = {"date": d1, "n_mom": len(mom), "n_rev": len(rev), "riskoff": riskoff}
        for name, sel in (("MOM", mom), ("REV", rev), ("BOTH", mom + rev)):
            sel = [s for s in sel if s in close.columns
                   and np.isfinite(close[s].asof(d0)) and np.isfinite(close[s].asof(d1))]
            if sel:
                r = float(np.mean([close[s].asof(d1) / close[s].asof(d0) - 1.0 for s in sel]))
            else:
                r = cash
            held = set(sel)
            to = len(held ^ prev[name]) / max(len(held) + len(prev[name]), 1)
            rec[name] = r - to * COST_BPS / 1e4
            prev[name] = held
        rec["SPY"] = float(spy.asof(d1) / spy.asof(d0) - 1.0)
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("date")
    print(f"\nMOM vs REV — trailing {YEARS}y, {len(df)} monthly periods "
          f"({df.index[0].date()} .. {df.index[-1].date()}), net@{COST_BPS:.0f}bps\n")
    print(f"{'leg':6s} {'Sharpe':>7s} {'CAGR':>8s} {'maxDD':>8s} {'vol':>7s} "
          f"{'avg names':>10s} {'% months active':>16s}")
    for leg in ("MOM", "REV", "BOTH", "SPY"):
        p = perf(df[leg])
        if leg in ("MOM", "REV"):
            n = df[f"n_{leg.lower()}"]
            extra = f"{n.mean():>10.1f} {(n > 0).mean():>15.0%}"
        else:
            extra = f"{'-':>10s} {'-':>15s}"
        print(f"{leg:6s} {p['sharpe']:>7.2f} {p['cagr']:>7.1%} {p['maxdd']:>8.1%} "
              f"{p['vol']:>7.1%} {extra}")

    tot = lambda c: float(np.prod(1 + df[c].dropna()) - 1)   # noqa: E731
    print(f"\ntotal {YEARS}y return:  MOM {tot('MOM'):+.1%}   REV {tot('REV'):+.1%}   "
          f"BOTH {tot('BOTH'):+.1%}   SPY {tot('SPY'):+.1%}")
    print(f"months REV had at least one name: {(df.n_rev > 0).sum()}/{len(df)}; "
          f"MOM: {(df.n_mom > 0).sum()}/{len(df)}")
    print(f"(REV falls back to BIL cash in the {(df.n_rev == 0).sum()} months it "
          f"finds nothing, which is why its vol is low)")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
