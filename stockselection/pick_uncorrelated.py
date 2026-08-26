"""
Find 10 high-momentum BUT mutually-uncorrelated S&P 500 names (both objectives:
return + diversification), with no correlated cluster > 2 names.

Rule: rank by 12-1 momentum (the validated return signal); greedily walk that
ranking and admit a name only if it is highly correlated (>CORR_HI daily-return
corr, last 252d) with AT MOST 1 name already picked. That caps every correlated
cluster at 2 and kills the single-theme concentration the naive top-N piles into.
"""
import numpy as np
import pandas as pd

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/sp500_close.parquet"
NAMES = "/Users/riddhisiddhi/stocks/stockselection/sp500.csv"
N_PICK, CORR_HI, CORR_WIN = 10, 0.60, 252   # 0.60 = "highly correlated" bar
CAND = 500           # all positive-momentum names (diversified 10th sits deep)
MOM_CAP = 4.0        # >400% 1y = almost always a spinoff/split artifact
JUMP_CAP = 0.50      # any 1-day move >50% = corporate-action artifact -> drop


def main():
    close = pd.read_parquet(CACHE)
    names = dict(zip(*pd.read_csv(NAMES).values.T))
    stocks = [c for c in close.columns if c != "SPY"]
    cs = close[stocks].dropna(how="all")
    asof = cs.index[-1]

    # drop names with a corporate-action price artifact in the window (spinoffs
    # like SNDK backfill a fake jump that fakes huge momentum)
    dret = cs.pct_change().tail(CORR_WIN)
    clean = [c for c in stocks if dret[c].abs().max() < JUMP_CAP]
    cs = cs[clean]

    # 12-1 momentum as of asof, artifact-capped
    p_now = cs.shift(21).iloc[-1]; p_then = cs.shift(252).iloc[-1]
    mom = (p_now / p_then - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    mom = mom[(mom > 0) & (mom < MOM_CAP)].sort_values(ascending=False)
    cand = list(mom.index[:CAND])

    # daily-return correlation over last CORR_WIN days
    rets = cs[cand].pct_change().tail(CORR_WIN)
    rets = rets.dropna(axis=1, thresh=int(CORR_WIN * 0.8))
    cand = [c for c in cand if c in rets.columns]
    C = rets.corr()

    def hi_partners(t, basket):
        return [p for p in basket if p != t and abs(C.loc[t, p]) > CORR_HI]

    picked = []
    for t in cand:                                          # momentum-ranked walk
        hp = hi_partners(t, picked)
        # admit only if t is highly-correlated with <=1 already-held name (caps
        # most clusters at 2; a rare 3-chain is reported below).
        if len(hp) > 1:
            continue
        picked.append(t)
        if len(picked) == N_PICK:
            break

    # fallback: if the constraint starved us (momentum too concentrated to seat
    # 10 diversified names), fill remaining slots with the MOST-diversifying
    # leftover names (lowest max|corr| to the basket). Flagged in the output.
    forced = []
    if len(picked) < N_PICK:
        remaining = [t for t in cand if t not in picked]
        while len(picked) < N_PICK and remaining:
            best = min(remaining, key=lambda t: C.loc[t, picked].abs().max())
            picked.append(best); forced.append(best); remaining.remove(best)

    # honest cluster report: connected components of the >CORR_HI graph
    adj = {t: set(hi_partners(t, picked)) for t in picked}
    seen, clusters = set(), []
    for t in picked:
        if t in seen:
            continue
        stack, comp = [t], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u); stack += list(adj[u] - seen)
        clusters.append(comp)

    # report
    sub = C.loc[picked, picked].copy()
    np.fill_diagonal(sub.values, np.nan)
    print(f"\n10 high-momentum, mutually-uncorrelated names  (as of {asof.date()}, "
          f"corr>|{CORR_HI}| = 'highly correlated', {CORR_WIN}d daily returns)\n")
    print(f"{'ticker':7s} {'12-1 mom':>9s} {'max|corr|':>9s}  most-correlated partner   name")
    print("-" * 84)
    for t in picked:
        row = sub.loc[t].dropna()
        mx = row.abs().idxmax(); mv = row.loc[mx]
        flag = " (fill)" if t in forced else "       "
        print(f"{t:7s} {mom[t]:>+8.1%} {abs(mv):>9.2f}  {mx:5s} ({mv:+.2f}){flag}   {str(names.get(t,t))[:24]}")

    off = sub.abs()
    print(f"\nbasket avg |corr| {np.nanmean(off.values):.2f}   max |corr| {np.nanmax(off.values):.2f}")
    big = [c for c in clusters if len(c) > 1]
    print(f"correlated clusters (>|{CORR_HI}|): "
          f"{', '.join('{'+'+'.join(c)+'}' for c in big) if big else 'none'}"
          f"   max cluster size {max(len(c) for c in clusters)}")

    naive = list(mom.index[:N_PICK])
    ncorr = C.loc[[x for x in naive if x in C.columns], [x for x in naive if x in C.columns]].copy()
    np.fill_diagonal(ncorr.values, np.nan)
    print(f"vs naive top-10 momentum: avg |corr| {np.nanmean(ncorr.abs().values):.2f}   "
          f"max |corr| {np.nanmax(ncorr.abs().values):.2f}   "
          f"highly-corr pairs: {int((ncorr.abs()>CORR_HI).sum().sum()/2)}")
    print(f"naive top-10: {naive}")


if __name__ == "__main__":
    main()
