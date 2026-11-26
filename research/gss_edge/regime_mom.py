"""12-1 momentum: (1) as an MR-vs-trend regime signal; (2) as the trend-sleeve ranking. Same bar as REGIME_REPORT."""
import itertools, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, regime_study as R
from scipy.stats import spearmanr
E, idx, n, K, SI = R.E, R.idx, R.n, R.K, R.SI
P1, P2 = "2007-2015", "2016-now"
# ---- 12-1 momentum: close[t-21]/close[t-252]-1 (skips the latest month), causal
PXdf = E.P[E.UNIV].reindex(idx); M121 = (PXdf.shift(21) / PXdf.shift(252) - 1).values
spy_m121 = R.spy.reindex(idx).shift(21) / R.spy.reindex(idx).shift(252) - 1
med_m121 = pd.Series(np.nanmedian(np.where(np.isfinite(M121), M121, np.nan), axis=1), index=idx)

# ================= PART 1: regime signal =================
NEW = {"mom121_spy": spy_m121, "mom121_univ_median": med_m121}
for k, v in NEW.items(): R.RAW[k] = v; R.PCT[k] = R.exp_pct(v)
RULES = {f"{k}|{m}": R.state(k, m) for k in NEW for m in ("hi33", "lo33", "hi50", "lo50")}
partners = [k for k in R.RAW if k not in NEW]            # all 12 existing signals (4 are the bottom signals incl. days-since-252d-low)
for k in NEW:
    for o in partners:
        for mk, mo in itertools.product(("hi50", "lo50"), repeat=2): RULES[f"{k}|{mk} AND {o}|{mo}"] = R.state(k, mk) & R.state(o, mo)
rows = []
for name, mr in RULES.items():
    e = R.ev(R.composite(mr)); row = dict(rule=name, mr_share=float(mr[SI:].mean()), switches=R.n_switch(mr))
    for p, _, _ in R.PER: row[f"sh_{p}"], row[f"cagr_{p}"], row[f"dd_{p}"] = e[p]
    row["PASS"] = R.passes(e); rows.append(row)
d1 = pd.DataFrame(rows); d1.to_csv("regime_mom_results.csv", index=False)
bt = {p: R.BASES["always_trend"][p][0] for p in (P1, P2)}
d1["m1"], d1["m2"] = d1[f"sh_{P1}"] - bt[P1], d1[f"sh_{P2}"] - bt[P2]
print(f"PART 1: {len(d1)} rules (2 mom signals x [4 alone + 12 partners x 4 combos]); PASS {int(d1.PASS.sum())}")
S = d1[~d1.rule.str.contains(" AND ")]; print("singles:\n", S[["rule", "switches", f"sh_{P1}", f"dd_{P1}", f"sh_{P2}", f"dd_{P2}", "m1", "m2", "PASS"]].round(3).to_string(index=False))
rho, pv = spearmanr(d1.m1, d1.m2); print(f"Spearman margin P1 vs P2 over these rules: {rho:.3f} (p={pv:.3f})")
for lab, sel, tst in [("select 07-15 -> test 16-now", "m1", "m2"), ("select 16-now -> test 07-15", "m2", "m1")]:
    t = d1.sort_values(sel, ascending=False).head(10); print(f"{lab}: top10 sel {t[sel].mean():+.3f} test {t[tst].mean():+.3f} positive {int((t[tst]>0).sum())}/10")
sub = [("07-10", "2007-01-01", "2010-12-31"), ("11-15", "2011-01-01", "2015-12-31"), ("16-20", "2016-01-01", "2020-12-31"), ("21-now", "2021-01-01", "2100-01-01")]
tr = {s: R.stats(R.RT, lo, hi)[0] for s, lo, hi in sub}; out = []
print("passers:")
for _, r in d1[d1.PASS].iterrows():
    mr = RULES[r.rule]; rc = R.composite(mr); m = [R.stats(rc, lo, hi)[0] - tr[s] for s, lo, hi in sub]
    R.COST = 1e-3; e2 = R.ev(R.run(np.where(mr[:, None], R.WM, R.WT))); R.COST = 5e-4
    m2x = [e2[p][0] - bt[p] for p in (P1, P2)]
    print(f"  {r.rule:55s} sw {r.switches:3d} {r['sh_'+P1]:.3f}/{r['dd_'+P1]*100:.1f}% {r['sh_'+P2]:.3f}/{r['dd_'+P2]*100:.1f}% | sub wins {sum(x>0 for x in m)}/4 | 2xcost {m2x[0]:+.3f}/{m2x[1]:+.3f}")
    out.append(dict(rule=r.rule, wins4=sum(x > 0 for x in m), m2x_p1=m2x[0], m2x_p2=m2x[1]))
pd.DataFrame(out).to_csv("regime_mom_passers.csv", index=False)
rng = np.random.default_rng(11); N = 400; npass = 0
for _ in range(N):
    dw = rng.choice([10, 30, 60]); st = np.zeros(n, bool); cur = rng.random() < .5
    for t in range(n):
        if rng.random() < 1 / dw: cur = rng.random() < .5
        st[t] = cur
    npass += R.passes(R.ev(R.composite(st)))
print(f"null pass-rate {npass/N:.3f} -> expected false passes among {len(d1)} rules ~ {npass/N*len(d1):.1f}")

# ================= PART 2: trend-sleeve ranking =================
CAPW = 0.15
OK_CUR = E.a200 & (E.PX > E.M50) & (E.SL > 0); S_CUR = np.where(OK_CUR, np.maximum(E.SL * E.R2, 0), 0.0)
OK_M = E.a200 & np.isfinite(M121) & (M121 > 0)
def trend_W(score, top_n=None):
    W = np.zeros((n, K)); cur = np.zeros(K); prev = False; score = np.where(np.isfinite(score), score, 0.0)
    for t in range(SI, n):
        if t in E.month_end or R.OFF[t] != prev:
            if R.OFF[t]: cur = np.zeros(K)
            else:
                s = score[t].copy()
                if top_n and (s > 0).sum() > top_n: s[s < np.sort(s)[-top_n]] = 0
                tot = s.sum()
                if tot > 0: w = np.where(s > 0, np.minimum(s / tot, CAPW), 0); cur = w / w.sum()
                else: cur = np.zeros(K)
        prev = R.OFF[t]; W[t] = cur
    return W
def rank_avg(a, b, ok):
    ra = pd.DataFrame(np.where(ok, a, np.nan)).rank(axis=1, pct=True).values; rb = pd.DataFrame(np.where(ok, b, np.nan)).rank(axis=1, pct=True).values
    return np.where(ok, np.nan_to_num((ra + rb) / 2), 0.0)
V = {
 "T0 current (slope*R2, filter a200/>50DMA/slope>0)": S_CUR, 
 "T1 12-1 rank: elig a200 & 12-1>0, size by 12-1": np.where(OK_M, M121, 0.0),
 "T2 12-1 top-8 of T1": ("top8", np.where(OK_M, M121, 0.0)),
 "T3 current filter, size by 12-1": np.where(OK_CUR & np.isfinite(M121) & (M121 > 0), M121, 0.0),
 "T4 current score, only names with 12-1>0": np.where(OK_M, S_CUR, 0.0),
 "T5 rank-average (slope*R2, 12-1), current filter": rank_avg(S_CUR, M121, OK_CUR & np.isfinite(M121)),
 "T0b control: current score top-8 (concentration only)": ("top8", S_CUR),
}
res = {}; WS = {}
for name, v in V.items():
    W = trend_W(v[1], 8) if isinstance(v, tuple) else trend_W(v); WS[name] = W; res[name] = R.ev(R.run(W))
base = res["T0 current (slope*R2, filter a200/>50DMA/slope>0)"]
print("\nPART 2: trend-sleeve ranking (Sharpe / CAGR / maxDD)")
for name, e in res.items():
    ok = all(e[p][0] > base[p][0] and e[p][2] >= base[p][2] - 0.01 for p in e) if not name.startswith("T0 ") else None
    R.COST = 1e-3; ec = R.ev(R.run(WS[name])); bc = R.ev(R.run(WS["T0 current (slope*R2, filter a200/>50DMA/slope>0)"])); R.COST = 5e-4
    c2 = all(ec[p][0] > bc[p][0] and ec[p][2] >= bc[p][2] - 0.01 for p in ec) if ok is not None else None
    sb = [R.stats(R.run(WS[name]), lo, hi)[0] - R.stats(R.run(WS["T0 current (slope*R2, filter a200/>50DMA/slope>0)"]), lo, hi)[0] for _, lo, hi in sub]
    print(f"{name:55s}" + " | ".join(f"{p}: {e[p][0]:.3f}/{e[p][1]*100:4.1f}%/{e[p][2]*100:6.1f}%" for p in e) + f" | bar:{'-' if ok is None else ok} 2xcost:{'-' if c2 is None else c2} sub-wins {sum(x>0 for x in sb)}/4")
# null: random ranking among the T1 eligible set, same sizing -> how much does ANY ranking matter?
rng = np.random.default_rng(3); sh = {P1: [], P2: []}
for _ in range(120):
    rs = np.where(OK_M, rng.random((n, K)), 0.0); e = R.ev(R.run(trend_W(rs)))
    for p in sh: sh[p].append(e[p][0])
t1 = res["T1 12-1 rank: elig a200 & 12-1>0, size by 12-1"]
for p in sh: print(f"null random-ranking (120 draws, same eligibility {P1 if p==P1 else P2}): Sharpe mean {np.mean(sh[p]):.3f} sd {np.std(sh[p]):.3f}; T1 percentile {np.mean(np.array(sh[p]) < t1[p][0])*100:.0f}%; T0 percentile {np.mean(np.array(sh[p]) < base[p][0])*100:.0f}%")
pd.DataFrame([dict(variant=k, **{f"{a}_{p}": x for p, e in v.items() for a, x in zip(("sh", "cagr", "dd"), e)}) for k, v in res.items()]).to_csv("trend_ranking_results.csv", index=False)
