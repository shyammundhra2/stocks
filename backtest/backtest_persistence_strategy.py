"""
Does the trend-vs-mean-revert label add tradable value?

The IC test (backtest_persistence_ic.py) showed the persistence score does not
*rank* forward trendiness cross-sectionally. This asks the better question:
if we gate a momentum rule vs. a mean-reversion rule on the label, does the
regime-conditional strategy beat applying either rule unconditionally?

Two upgrades over the IC test, per the improvement plan:
  1. ASSET-RELATIVE persistence. Instead of comparing VR levels across
     heterogeneous assets (TLT vs IBIT vs GLD - confounded by asset-class
     microstructure), each asset's current VR is z-scored against ITS OWN
     trailing VR history. "Is this asset unusually trending for itself."
  2. Validated on P&L (Sharpe / return), not a statistical proxy.

Per rebalance date t, for each asset (no lookahead - everything uses data
through t only):
  momentum signal   mom = sign(trailing MOM_LB-day return)      (trade with trend)
  reversion signal  rev = -sign(trailing REV_LB-day return)     (fade recent move)
  persistence       vr  = VR(q=21) on trailing LOOKBACK bars
  self-norm z       znorm = (vr - mean(own trailing VR)) / std(own trailing VR)

Strategies (equal-weight over active names, position in {-1,0,+1}):
  long_all        +1 everything                (beta benchmark)
  uncond_mom      momentum on every asset
  uncond_rev      reversion on every asset
  cond_selfnorm   TREND(znorm>+Z)->mom, REVERT(znorm<-Z)->rev, else flat
  cond_absolute   TREND(vr>1.1)->mom,   REVERT(vr<0.9)->rev,    else flat

Holding period = FWD trading days, rebalanced every FWD days (non-overlapping)
so the period returns are independent and the Sharpe t-stat is honest.

NOTE: signals are long/short; shorting some ETFs is unrealistic, so read this
as a signal-quality test, not a deployable P&L. Costs are ignored.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import _variance_ratio

END = pd.Timestamp("2026-08-20")
TEST_START = END - pd.DateOffset(years=2)
LOOKBACK = 252    # trailing bars for the VR estimate
FWD = 10          # holding period (~2 weeks) and rebalance cadence
NORM_WIN = 126    # trailing window for the asset-relative VR z-score (~6mo)
Z_THR = 0.5       # |znorm| gate for TREND / MEAN-REVERT
VR_HI, VR_LO = 1.1, 0.9   # absolute-VR gate for the comparison variant
MOM_LB = 63       # momentum lookback (~3mo)
REV_LB = 5        # reversion lookback (~1wk short-term reversal)
MIN_NAMES = 6     # min active names to record a period


def ann_stats(period_rets, ppy):
    r = np.asarray(period_rets, dtype=float)
    if len(r) < 3:
        return (float("nan"),) * 4
    mu, sd = r.mean(), r.std(ddof=1)
    ann_ret = mu * ppy
    ann_vol = sd * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    hit = float(np.mean(r > 0))
    return ann_ret, ann_vol, sharpe, hit


def main(fwd=FWD, step=None):
    step = step or fwd
    t0 = time.time()
    print(f"Holding period: {fwd}d | rebalance: {step}d | self-norm win: {NORM_WIN}d | Z: {Z_THR}\n")

    tickers = list(TREND_ASSETS.keys())
    data_start = (TEST_START - pd.DateOffset(days=620)).date()  # burn-in for VR + its own history
    print(f"Downloading {len(tickers)} tickers {data_start} -> {END.date()} ...")
    raw = yf.download(tickers, start=str(data_start), end=str((END + pd.Timedelta(days=1)).date()),
                      progress=False, group_by="column", auto_adjust=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(axis=1, how="all")
    present = [t for t in tickers if t in close.columns]
    skipped = [t for t in tickers if t not in present]
    if skipped:
        print(f"Skipped (no data): {', '.join(skipped)}")
    idx = close.index
    n = len(idx)
    print(f"Universe: {len(present)} tickers, {n} bars\n")

    # ---- precompute a daily VR series per asset (one pass; reused for z-scoring) ----
    print("Precomputing rolling VR per asset ...")
    vr = pd.DataFrame(index=idx, columns=present, dtype=float)
    for sym in present:
        s = close[sym]
        col = np.full(n, np.nan)
        vals = s.values
        for d in range(LOOKBACK, n):
            w = vals[d - LOOKBACK:d + 1]
            if np.isnan(w).any():
                continue
            col[d] = _variance_ratio(w, q=21)
        vr[sym] = col

    logc = np.log(close)

    start_i = max(LOOKBACK + NORM_WIN, MOM_LB)
    positions = [i for i in range(start_i, n - fwd) if idx[i] >= TEST_START][::step]

    strategies = ["long_all", "uncond_mom", "uncond_rev", "cond_selfnorm", "cond_absolute"]
    # mechanism decomposition: right-way vs wrong-way rule inside each bucket.
    # concept holds iff trend_MOM > trend_REV and revert_REV > revert_MOM.
    mech = ["trend_mom", "trend_rev", "revert_rev", "revert_mom"]
    strategies += mech
    rets = {s: [] for s in strategies}
    breadth = {s: [] for s in strategies}

    for i in positions:
        book = {s: [] for s in strategies}
        for sym in present:
            lp = logc[sym]
            p0, pf = lp.iloc[i], lp.iloc[i + fwd]
            pm, pr = lp.iloc[i - MOM_LB], lp.iloc[i - REV_LB]
            if not np.isfinite([p0, pf, pm, pr]).all():
                continue
            fret = np.expm1(pf - p0)               # forward simple return
            mom = np.sign(p0 - pm)
            rev = -np.sign(p0 - pr)

            vr_now = vr[sym].iloc[i]
            hist = vr[sym].iloc[i - NORM_WIN:i].dropna()
            if len(hist) >= 20 and np.isfinite(vr_now) and hist.std() > 0:
                znorm = (vr_now - hist.mean()) / hist.std()
            else:
                znorm = np.nan

            book["long_all"].append(fret)
            book["uncond_mom"].append(mom * fret)
            book["uncond_rev"].append(rev * fret)

            if np.isfinite(znorm):
                pos = mom if znorm > Z_THR else rev if znorm < -Z_THR else 0.0
            else:
                pos = 0.0
            if pos != 0.0:
                book["cond_selfnorm"].append(pos * fret)

            # mechanism buckets: which rule works inside each self-norm regime
            if np.isfinite(znorm):
                if znorm > Z_THR:
                    book["trend_mom"].append(mom * fret)
                    book["trend_rev"].append(rev * fret)
                elif znorm < -Z_THR:
                    book["revert_rev"].append(rev * fret)
                    book["revert_mom"].append(mom * fret)

            if np.isfinite(vr_now):
                pos2 = mom if vr_now > VR_HI else rev if vr_now < VR_LO else 0.0
            else:
                pos2 = 0.0
            if pos2 != 0.0:
                book["cond_absolute"].append(pos2 * fret)

        for s in strategies:
            vals = [x for x in book[s] if np.isfinite(x)]
            if len(vals) >= MIN_NAMES:
                rets[s].append(np.mean(vals))
                breadth[s].append(len(vals))

    ppy = 252 / fwd
    print(f"Periods: {len(rets['long_all'])}  "
          f"({idx[positions[0]].date()} -> {idx[positions[-1]].date()}), {ppy:.1f}/yr\n")

    print(f"{'strategy':15s} {'ann_ret':>8s} {'ann_vol':>8s} {'Sharpe':>7s} "
          f"{'hit':>5s} {'names':>6s}")
    print("-" * 56)
    sharpes = {}
    for s in strategies:
        ar, av, sh, hit = ann_stats(rets[s], ppy)
        nm = np.mean(breadth[s]) if breadth[s] else float("nan")
        sharpes[s] = sh
        print(f"{s:15s} {ar:>8.1%} {av:>8.1%} {sh:>7.2f} {hit:>5.0%} {nm:>6.1f}")

    # Sharpe standard error ~ sqrt(ppy / n_periods): the noise floor for "is
    # this Sharpe distinguishable from zero" at this sample size.
    npd = len(rets["long_all"])
    se = np.sqrt(ppy / npd) if npd else float("nan")
    print(f"\n(Sharpe standard error at n={npd}: +/-{se:.2f} - anything inside that is noise)")

    print("\nMechanism (right-way should beat wrong-way if the label works):")
    print(f"  TREND names : momentum {sharpes['trend_mom']:+.2f}  vs  reversion {sharpes['trend_rev']:+.2f}")
    print(f"  REVERT names: reversion {sharpes['revert_rev']:+.2f}  vs  momentum {sharpes['revert_mom']:+.2f}")

    # Conviction sweep: does gating on a STRONGER score help?
    print("\nConviction sweep (cond_selfnorm at higher |z| gates):")
    print(f"{'z_gate':>7s} {'ann_ret':>8s} {'Sharpe':>7s} {'hit':>5s} {'names':>6s}")
    for z in (0.5, 1.0, 1.5, 2.0):
        pr, br = [], []
        for i in positions:
            vals = []
            for sym in present:
                lp = logc[sym]
                p0, pf = lp.iloc[i], lp.iloc[i + fwd]
                pm, pr_ = lp.iloc[i - MOM_LB], lp.iloc[i - REV_LB]
                if not np.isfinite([p0, pf, pm, pr_]).all():
                    continue
                vn = vr[sym].iloc[i]
                hist = vr[sym].iloc[i - NORM_WIN:i].dropna()
                if len(hist) < 20 or not np.isfinite(vn) or hist.std() == 0:
                    continue
                zz = (vn - hist.mean()) / hist.std()
                pos = np.sign(p0 - pm) if zz > z else -np.sign(p0 - pr_) if zz < -z else 0.0
                if pos != 0.0:
                    vals.append(pos * np.expm1(pf - p0))
            if len(vals) >= 3:
                pr.append(np.mean(vals)); br.append(len(vals))
        ar, av, sh, hit = ann_stats(pr, ppy)
        nm = np.mean(br) if br else float("nan")
        print(f"{z:>7.1f} {ar:>8.1%} {sh:>7.2f} {hit:>5.0%} {nm:>6.1f}")

    print(f"\nBenchmark long_all Sharpe {sharpes['long_all']:.2f} is pure beta (bull market).")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Persistence-gated momentum/reversion strategy backtest")
    ap.add_argument("--fwd", type=int, default=FWD, help="holding period in trading days")
    ap.add_argument("--step", type=int, default=None, help="rebalance cadence (default = fwd, non-overlapping)")
    a = ap.parse_args()
    main(fwd=a.fwd, step=a.step)
