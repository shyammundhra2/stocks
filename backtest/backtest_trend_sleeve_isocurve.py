"""
The standalone trend sleeve (trendselection/predict_top_trends.py,
validated by backtest_trend_dd_tilt.py) has NO trend-quality check at all -
any name with positive 12-1 momentum qualifies, even a noisy/choppy one.
Does adding an iso-conviction-curve quality filter help?

"Iso-curve positive" = conv = |slope| x R^2 (the live book's ONE validated
single signal, per macro/indicators/portfolio.py's own comment) at or above
a real curve level (1, the live map's own BUY-tier threshold - not just
slope>0, which includes plenty of near-zero-conviction noise), AND slope>0
specifically (the trend side, not a high-conviction breakdown).

Reuses backtest_mom_dd_tilt_context.py's panel (same live _trend_stats/
_efficiency_ratio functions, same universe) for fidelity. Within the
sleeve's actual eligibility (m12>0), compares the iso-positive subset
against the rest, fwd 20d.
"""
import sys
import time

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from backtest.backtest_mom_dd_tilt_context import build_panel


def rep(sub, label, pool_mean=None):
    if len(sub) < 50:
        print(f"  {label:40s} n={len(sub):>5}  (too few)"); return
    m = sub.fwd.mean()
    edge = f"  vs base {m - pool_mean:>+6.2%}" if pool_mean is not None else ""
    print(f"  {label:40s} n={len(sub):>5}  fwd {m:>+7.2%}{edge}  hit {(sub.fwd > 0).mean():>5.0%}")


def main():
    t0 = time.time()
    d = build_panel()

    # Sleeve's current eligibility: m12 (12-1 momentum) > 0, no quality gate.
    sleeve = d[d.m12 > 0].dropna(subset=["slope", "r2", "fwd"])
    base_mean = sleeve.fwd.mean()
    conv = sleeve.slope.abs() * sleeve.r2
    iso_pos = (conv >= 1.0) & (sleeve.slope > 0)

    print(f"\nSleeve-eligible events (m12>0, live sleeve rule): n={len(sleeve)}, "
          f"base fwd {base_mean:+.2%}\n")
    rep(sleeve, "base (m12>0 only, current sleeve)")
    rep(sleeve[iso_pos], "+ iso-curve positive (conv>=1, slope>0)", base_mean)
    rep(sleeve[~iso_pos], "  complement (below the curve or slope<=0)", base_mean)

    # Finer look: does the curve threshold matter, or is any positive slope enough?
    for k in (0.5, 1.0, 2.0, 4.0):
        sub = sleeve[(conv >= k) & (sleeve.slope > 0)]
        rep(sub, f"  conv>={k}, slope>0", base_mean)

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
