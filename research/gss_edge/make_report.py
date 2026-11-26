import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, engine as E
from run_ideas import BATCH1, BASE
from run_batch2 import B2
ALL = {**BATCH1, **B2}
df = pd.read_csv("results_all.csv"); rb = E.sim(); cr = pd.Series(E.cashr, index=E.idx); rng = np.random.default_rng(1)
def boot(rv, p, lo, hi):
    a = (rv.loc[lo:hi] - cr.loc[lo:hi]).values; b = (rb.loc[lo:hi] - cr.loc[lo:hi]).values; m = len(a); w = 0
    for _ in range(2000):
        st = rng.integers(0, m - 21, m // 21 + 1); ix = np.concatenate([np.arange(s, s + 21) for s in st])[:m]
        w += (a[ix].mean() / a[ix].std()) > (b[ix].mean() / b[ix].std())
    return w / 2000
bp = {}
for v in df[df.PASS].variant:
    rv = E.sim(ALL[v]); bp[v] = [boot(rv, p, lo, hi) for p, lo, hi in E.PERIODS]
df["boot_2007_15"] = df.variant.map(lambda v: bp[v][0] if v in bp else np.nan)
df["boot_2016_now"] = df.variant.map(lambda v: bp[v][1] if v in bp else np.nan)
df.to_csv("results_all.csv", index=False)
b = BASE; f = lambda x: f"{x:.3f}"
L = ["BASELINE  2007-2015: Sharpe %.3f maxDD %.1f%% | 2016-now: Sharpe %.3f maxDD %.1f%%" % (b['2007-2015'][0], b['2007-2015'][2]*100, b['2016-now'][0], b['2016-now'][2]*100), ""]
L.append("| variant | Sh 07-15 | DD 07-15 | Sh 16-now | DD 16-now | P(Sh>base) 07-15 / 16-now |"); L.append("|---|---|---|---|---|---|")
for _, r in df[df.PASS].iterrows():
    L.append(f"| {r.variant} | {r['sh_2007-2015']:.3f} | {r['dd_2007-2015']*100:.1f}% | {r['sh_2016-now']:.3f} | {r['dd_2016-now']*100:.1f}% | {r.boot_2007_15:.2f} / {r.boot_2016_now:.2f} |")
print("\n".join(L)); print("\nvariants tried:", len(df), " passed:", int(df.PASS.sum()))
open("survivors_table.md", "w").write("\n".join(L) + f"\n\nvariants tried: {len(df)}, passed: {int(df.PASS.sum())}\n")
