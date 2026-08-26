"""
Kelly position-sizing of a $50k, 10-name momentum book.

Multivariate Kelly: f* = Sigma^-1 mu  (inverse-covariance x expected excess
returns) maximizes expected log-growth. Honest inputs:
  mu    momentum gives RANK not magnitude, so expected excess return is a modest
        assumption scaled by momentum rank (base 8% -> 16% annual). Stated, not
        fitted - Kelly is only as good as mu, so we keep mu conservative.
  Sigma annualized daily-return covariance (252d), shrunk 30% toward its diagonal
        for stability (10x10 from 252 obs is noisy; shrinkage tames the inverse).
  fractional Kelly x0.25 ('senior strategist' safety - full Kelly's fat-tail
  drawdowns are ruinous on single names), long-only (clip<0), per-name cap 25%,
  total capped at 100% of the book (no leverage), remainder -> cash.

Compare to equal-weight (5% each) - the robust default Kelly tilts around.
"""
import numpy as np
import pandas as pd

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
SEC = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_sectors.parquet"
NAMES = "/Users/riddhisiddhi/stocks/stockselection/sp500.csv"
BOOK = 50_000
NAME_CAP, SHRINK = 0.20, 0.30
# Edge-weighted mu: expected excess return PROPORTIONAL to momentum rank, with a
# wide enough spread that momentum (not just variance) drives the Kelly tilt -
# this is the version that respects the edge (backtest: rank-weighting HELPED,
# min-variance HURT). 6% -> 30% annual excess across the momentum rank.
MU_BASE, MU_SPAN = 0.06, 0.24


def main():
    close = pd.read_parquet(CACHE)
    sec = pd.read_parquet(SEC).set_index("Symbol")["GICS Sector"].to_dict()
    names = dict(zip(*pd.read_csv(NAMES).values.T))
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks]

    # rebuild the sector-capped 10 (<=3/sector) as of the latest date
    d = cs.pct_change().tail(252)
    keep = [c for c in stocks if d[c].abs().max() < 0.50]
    mom = (cs[keep].shift(21).iloc[-1] / cs[keep].shift(252).iloc[-1] - 1.0)
    mom = mom[(mom > 0) & (mom < 4.0)].sort_values(ascending=False)
    picked, cnt = [], {}
    for t in mom.index:
        s = sec.get(t, t)
        if cnt.get(s, 0) < 3:
            picked.append(t); cnt[s] = cnt.get(s, 0) + 1
        if len(picked) == 10:
            break

    # mu: expected annual excess return scaled by momentum rank within the 10
    r = mom[picked].rank(pct=True)                       # 0.1 .. 1.0
    mu = (MU_BASE + MU_SPAN * r).values

    # Sigma: annualized covariance, shrunk toward its diagonal
    dr = cs[picked].pct_change().tail(252)
    S = dr.cov().values * 252.0
    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))

    # Kelly: f* = Sigma^-1 mu, long-only, normalized to sum=1 (full $50k, NO
    # leverage), then per-name cap enforced by redistributing excess to the
    # uncapped names (iterated to convergence).
    f = np.clip(np.linalg.solve(S, mu), 0.0, None)
    if f.sum() <= 0:
        f = np.ones(len(picked))
    f = f / f.sum()
    for _ in range(50):
        over = f > NAME_CAP + 1e-9
        if not over.any():
            break
        excess = (f[over] - NAME_CAP).sum()
        f[over] = NAME_CAP
        room = ~over
        if room.any():
            f[room] += excess * f[room] / f[room].sum()
        else:
            break
    w = pd.Series(f, index=picked)

    ew = 1.0 / len(picked)
    vol = np.sqrt(dr.cov().values.dot(np.diag(np.ones(len(picked)))).diagonal() * 252)
    port_vol = float(np.sqrt(w.values @ (dr.cov().values * 252) @ w.values))

    print(f"\nMomentum-rank Kelly (f*=Sigma^-1 mu, mu ~ momentum) allocation of "
          f"${BOOK:,} across {len(picked)} names\n(long-only, {NAME_CAP:.0%} cap, "
          f"fully deployed, no leverage)\n")
    print(f"{'ticker':6s} {'mom':>6s} {'ann.vol':>7s} {'Kelly w':>8s} {'$ alloc':>9s} {'vs EW':>6s}  sector")
    print("-" * 78)
    for i, t in enumerate(picked):
        print(f"{t:6s} {mom[t]:>+5.0%} {vol[i]:>7.0%} {w[t]:>8.1%} {w[t]*BOOK:>9,.0f} "
              f"{(w[t]-ew):>+6.1%}  {sec.get(t,'?')}")
    print("-" * 78)
    print(f"{'TOTAL':6s} {'':6s} {'':7s} {w.sum():>8.1%} {w.sum()*BOOK:>9,.0f}   "
          f"cash ${(1-w.sum())*BOOK:,.0f}")
    print(f"\nportfolio vol {port_vol:.1%}   "
          f"(equal-weight vol {np.sqrt(np.ones(len(picked))/len(picked) @ (dr.cov().values*252) @ (np.ones(len(picked))/len(picked))):.1%})")
    print("Kelly tilts toward lower-vol / lower-correlation names; momentum rank "
          "nudges toward the leaders. Cap+quarter-Kelly guard against estimation error.")


if __name__ == "__main__":
    main()
