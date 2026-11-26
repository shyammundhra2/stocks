import pickle, warnings; warnings.filterwarnings("ignore")
import pandas as pd, replay_sim as S
rows = []
for win in (1000, 252):
    W = pickle.load(open(f"weights_w{win}.pkl", "rb")); offs = sum(v["off"] for v in W.values())
    print(f"\n=== replay window {win} bars | {len(W)} decision dates, {offs} risk-off | errors: {sum('err' in v for v in W.values())}")
    base = None
    for name, kw in [("baseline (live optimizer, no overlays)", {}), ("haven only", dict(haven=True)),
                     ("reserve folded in only", dict(reserve=True)), ("haven + reserve", dict(haven=True, reserve=True))]:
        r = S.sim(W, **kw); ev = S.evaluate(r); base = base or ev
        ok = all(ev[p][0] > base[p][0] and ev[p][2] >= base[p][2] - 0.01 for p in ev) if kw else None
        print(f"{name:40s}" + " | ".join(f"{p}: Sh {ev[p][0]:.3f} CAGR {ev[p][1]*100:5.1f}% DD {ev[p][2]*100:6.1f}%" for p in ev), "" if ok is None else ("PASS" if ok else "fail"))
        rows.append(dict(window=win, variant=name, **{f"{k}_{p}": v for p, e in ev.items() for k, v in zip(("sharpe", "cagr", "maxdd"), e)}, passes=ok))
pd.DataFrame(rows).to_csv("live_backtest_results.csv", index=False)
