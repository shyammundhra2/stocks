"""Score T6-style MOM eligibility variants run through the LIVE optimizer, vs the live baseline.
Bar (same as before): Sharpe(excess BIL) > baseline in BOTH periods and maxDD no more than 1pp worse; then 2x cost
and 4 sub-periods. Also an equal-weight control: same selected names, equal weights, same gross as the live book."""
import pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, replay_sim as S

def load(n): return pickle.load(open(n, "rb"))
base = load("weights_w1000.pkl")
variants = {"baseline (live)": base, "m12 (TREND state)": load("weights_m12_w1000.pkl"),
            "m12_noer (any ER state)": load("weights_m12_noer_w1000.pkl")}

def equalize(W):
    out = {}
    for d, v in W.items():
        w = v["w"]; g = sum(w.values())
        out[d] = dict(v, w={s: g / len(w) for s in w} if w else {})
    return out
variants["m12_noer equal-weight (same gross)"] = equalize(variants["m12_noer (any ER state)"])

SUB = [("07-10", "2007-01-01", "2010-12-31"), ("11-15", "2011-01-01", "2015-12-31"),
       ("16-20", "2016-01-01", "2020-12-31"), ("21-now", "2021-01-01", "2100-01-01")]
res = {}
for name, W in variants.items():
    S.COST = 5e-4; r = S.sim(W); S.COST = 1e-3; r2 = S.sim(W); S.COST = 5e-4
    res[name] = (r, r2)
b_r, b_r2 = res["baseline (live)"]; bev = S.evaluate(b_r); bev2 = S.evaluate(b_r2)
for name, (r, r2) in res.items():
    ev, ev2 = S.evaluate(r), S.evaluate(r2)
    print(f"\n{name}")
    for p in ev: print(f"  {p}: Sh {ev[p][0]:.3f} CAGR {ev[p][1]*100:5.1f}% DD {ev[p][2]*100:6.1f}%   (2x cost Sh {ev2[p][0]:.3f})")
    if name != "baseline (live)":
        ok = all(ev[p][0] > bev[p][0] and ev[p][2] >= bev[p][2] - 0.01 for p in ev)
        ok2 = all(ev2[p][0] > bev2[p][0] and ev2[p][2] >= bev2[p][2] - 0.01 for p in ev2)
        subs = sum(S.stats(r, lo, hi)[0] > S.stats(b_r, lo, hi)[0] for _, lo, hi in SUB)
        print(f"  bar: {'PASS' if ok else 'fail'} | at 2x cost: {'PASS' if ok2 else 'fail'} | sub-periods beating baseline Sharpe: {subs}/4")
        nn = np.mean([len(v['w']) for v in variants[name].values()]); print(f"  avg names {nn:.1f} (baseline {np.mean([len(v['w']) for v in base.values()]):.1f})")
