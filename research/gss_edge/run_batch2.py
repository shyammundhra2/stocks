import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, engine as E
from run_ideas import run, BASE

B2 = {
 # control: 20% permanently idle in BIL, NO capitulation deploy -> isolates de-risking from the capitulation edge
 "C1 control: 80% book, 20% BIL, no deploy": dict(reserve="carve0"),
 # own ideas (grids declared up front, no post-hoc tuning)
 "5a vol-managed target 10%": dict(volt=0.10), "5b vol-managed target 15%": dict(volt=0.15), "5c vol-managed target 20%": dict(volt=0.20),
 "6a breadth<35% halve": dict(breadth=0.35), "6b breadth<50% halve": dict(breadth=0.50),
 "7a book-DD breaker 10%": dict(ddbrk=0.10), "7b book-DD breaker 15%": dict(ddbrk=0.15),
 "8a SPY 20d vol>25% halve": dict(spyvol=0.25), "8b SPY 20d vol>35% halve": dict(spyvol=0.35),
 "9 death-cross also exits": dict(deathcross=True),
 "10 idle cash -> IEF/SHY if beats BIL": dict(haven="best", idle_haven=True),
 # robustness neighbours of the first-round passers
 "2c-25% haven size 25%": dict(haven="best", haven_frac=0.25), "2c-75% haven size 75%": dict(haven="best", haven_frac=0.75),
 "2c-100% haven size 100%": dict(haven="best", haven_frac=1.0),
 "4a-10% reserve 10%": dict(reserve="carve", reserve_frac=0.10), "4a-30% reserve 30%": dict(reserve="carve", reserve_frac=0.30),
 "4b-30% idle reserve cap 30%": dict(reserve="idle", reserve_frac=0.30),
 # combos of first-round passers
 "11 haven best + idle reserve": dict(haven="best", reserve="idle"),
 "12 haven best + carve reserve": dict(haven="best", reserve="carve"),
}
if __name__ == "__main__":
    df = run("batch2", B2)
    # paired block bootstrap vs baseline (21d blocks, 2000 draws): P(Sharpe_variant > Sharpe_base) per period
    rb = E.sim(); rng = np.random.default_rng(0); lines = []
    cr = pd.Series(E.cashr, index=E.idx)
    for name in df[df.PASS].variant:
        cfg = {**B2, **__import__("run_ideas").BATCH1}[name]; rv = E.sim(cfg); parts = []
        for p, lo, hi in E.PERIODS:
            a = (rv.loc[lo:hi] - cr.loc[lo:hi]).values; b = (rb.loc[lo:hi] - cr.loc[lo:hi]).values; m = len(a); nb = m // 21 + 1; wins = 0
            for _ in range(2000):
                st = rng.integers(0, m - 21, nb); ix = np.concatenate([np.arange(s, s + 21) for s in st])[:m]
                wins += (a[ix].mean() / a[ix].std()) > (b[ix].mean() / b[ix].std())
            parts.append(f"{p}: P(Sharpe>base)={wins/2000:.2f}")
        lines.append(f"{name:36s} " + "  ".join(parts))
    print("\nBOOTSTRAP (survivors of batch2):"); print("\n".join(lines))
