import warnings, io, contextlib; warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()): import regime_mom as M
import numpy as np, pandas as pd, regime_study as R
base_name = "T0 current (slope*R2, filter a200/>50DMA/slope>0)"; sub = M.sub
def bar(Wn, label):
    R.COST = 5e-4; e = R.ev(R.run(Wn)); b = M.res[base_name]
    ok = all(e[p][0] > b[p][0] and e[p][2] >= b[p][2] - 0.01 for p in e)
    R.COST = 1e-3; ec = R.ev(R.run(Wn)); bc = R.ev(R.run(M.WS[base_name])); R.COST = 5e-4
    c2 = all(ec[p][0] > bc[p][0] and ec[p][2] >= bc[p][2] - 0.01 for p in ec)
    sb = [R.stats(R.run(Wn), lo, hi)[0] - R.stats(R.run(M.WS[base_name]), lo, hi)[0] for _, lo, hi in sub]
    print(f"{label:58s}" + " | ".join(f"{p}: {e[p][0]:.3f}/{e[p][1]*100:4.1f}%/{e[p][2]*100:6.1f}%" for p in e) + f" | bar:{ok} 2xcost:{c2} sub-wins {sum(x>0 for x in sb)}/4 | 2xcost Sh {ec['2007-2015'][0]:.3f}/{ec['2016-now'][0]:.3f}")
print("EXPLORATORY controls (added after seeing the null; not pre-declared):")
bar(M.trend_W(np.where(M.OK_M, 1.0, 0.0)), "T6 equal-weight of eligible (a200 & 12-1>0)")
bar(M.trend_W(np.where(M.OK_CUR, 1.0, 0.0)), "T7 equal-weight of current filter (a200,>50DMA,slope>0)")
bar(M.trend_W(np.where(M.OK_M & (M.E.PX > M.E.M50), 1.0, 0.0)), "T8 equal-weight a200 & >50DMA & 12-1>0")
print("\nPre-declared T1 detail (the 12-1 ranking):"); bar(M.WS["T1 12-1 rank: elig a200 & 12-1>0, size by 12-1"], "T1")
print("\nPart 1 near-miss + combos with days-since-252d-low:")
d1 = M.d1; keys = [r for r in d1.rule if r.startswith("mom121_spy|lo33") or "bottom_days_252" in r]
b = M.bt
for r in keys:
    row = d1[d1.rule == r].iloc[0]; mr = M.RULES[r]; R.COST = 1e-3; e2 = R.ev(R.run(np.where(mr[:, None], R.WM, R.WT))); R.COST = 5e-4
    print(f"{r:55s} sw {row.switches:3d} {row['sh_2007-2015']:.3f}/{row['dd_2007-2015']*100:.1f}% {row['sh_2016-now']:.3f}/{row['dd_2016-now']*100:.1f}% beats-trend-Sharpe-both:{row.m1>0 and row.m2>0} PASS:{row.PASS} 2xcost margin {e2['2007-2015'][0]-b['2007-2015']:+.3f}/{e2['2016-now'][0]-b['2016-now']:+.3f}")
