import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, regime_study as R
from scipy.stats import spearmanr
df = pd.read_csv("regime_results.csv"); P1, P2 = "2007-2015", "2016-now"
bt = {p: R.BASES["always_trend"][p][0] for p in (P1, P2)}
df["m1"] = df[f"sh_{P1}"] - bt[P1]; df["m2"] = df[f"sh_{P2}"] - bt[P2]
rho, pv = spearmanr(df.m1, df.m2); print(f"Spearman corr of Sharpe margin-vs-trend across all {len(df)} rules, P1 vs P2: {rho:.3f} (p={pv:.3f})")
for lab, sel, tst in [("select on 2007-15 -> test 2016-now", "m1", "m2"), ("select on 2016-now -> test 2007-15", "m2", "m1")]:
    top = df.sort_values(sel, ascending=False).head(10)
    print(f"{lab}: top-10 by selection margin avg {top[sel].mean():+.3f}; same rules in test period avg {top[tst].mean():+.3f}; test-period margin>0 for {int((top[tst]>0).sum())}/10")
sub = [("07-10", "2007-01-01", "2010-12-31"), ("11-15", "2011-01-01", "2015-12-31"), ("16-20", "2016-01-01", "2020-12-31"), ("21-now", "2021-01-01", "2100-01-01")]
tr = {s: R.stats(R.RT, lo, hi)[0] for s, lo, hi in sub}
P = df[df.PASS].copy(); print("\nPASSERS (", len(P), ") 4-sub-period Sharpe margin vs always-trend, and 2x-cost check:")
out = []
for _, r in P.iterrows():
    mr = R.RULES[r.rule]; rc = R.composite(mr); m = [R.stats(rc, lo, hi)[0] - tr[s] for s, lo, hi in sub]
    R.COST = 1e-3; rc2 = R.run(np.where(mr[:, None], R.WM, R.WT)); R.COST = 5e-4
    e2 = R.ev(rc2); m2x = [e2[p][0] - R.stats(R.RT, lo, hi)[0] if False else e2[p][0] - bt[p] for p in (P1, P2)]
    print(f"{r.rule:48s} switches {r.switches:3d} | sub-margins " + " ".join(f"{x:+.2f}" for x in m) + f" | wins {sum(x>0 for x in m)}/4 | 2xcost margin {m2x[0]:+.3f}/{m2x[1]:+.3f}")
    out.append(dict(rule=r.rule, wins4=sum(x > 0 for x in m), **{f"sub_{s}": x for (s, _, _), x in zip(sub, m)}, margin_2xcost_p1=m2x[0], margin_2xcost_p2=m2x[1]))
pd.DataFrame(out).to_csv("regime_passers_robustness.csv", index=False)
print("\nPassers full numbers:"); print(P[["rule","mr_share","sh_2007-2015","cagr_2007-2015","dd_2007-2015","sh_2016-now","cagr_2016-now","dd_2016-now"]].round(3).to_string(index=False))
# single-signal ranking (alone)
S = df[~df.rule.str.contains(" AND ")].copy(); S["margin"] = np.minimum(S.m1, S.m2)
print("\nSINGLE signals alone (best of 4 modes each, by worst-period margin):")
S["sig"] = S.rule.str.split("|").str[0]; print(S.sort_values("margin", ascending=False).groupby("sig").head(1).sort_values("margin", ascending=False)[["rule","margin","PASS","switches"]].round(3).to_string(index=False))
