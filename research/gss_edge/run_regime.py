import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, regime_study as R
rows = []
for name, mr in R.RULES.items():
    e = R.ev(R.composite(mr)); row = dict(rule=name, mr_share=float(mr[R.SI:].mean()), switches=R.n_switch(mr))
    for p, _, _ in R.PER: row[f"sh_{p}"], row[f"cagr_{p}"], row[f"dd_{p}"] = e[p]
    row["PASS"] = R.passes(e); row["beats_trend_both"] = all(e[p][0] > R.BASES["always_trend"][p][0] for p in e)
    rows.append(row)
df = pd.DataFrame(rows); df.to_csv("regime_results.csv", index=False)
bt = {p: R.BASES["always_trend"][p] for p in ("2007-2015", "2016-now")}
print("rules tested:", len(df), "| PASS (beat all 3 baselines' Sharpe in BOTH periods, DD <=1pp worse than best baseline):", int(df.PASS.sum()),
      "| beat always-trend Sharpe in both:", int(df.beats_trend_both.sum()))
for p in ("2007-2015", "2016-now"): print(p, "rules beating always-trend Sharpe:", int((df[f"sh_{p}"] > bt[p][0]).sum()), "of", len(df))
print("\nTop 12 by worst-period Sharpe margin vs always-trend:")
df["margin"] = np.minimum(df["sh_2007-2015"] - bt["2007-2015"][0], df["sh_2016-now"] - bt["2016-now"][0])
print(df.sort_values("margin", ascending=False).head(12)[["rule", "switches", "mr_share", "sh_2007-2015", "dd_2007-2015", "sh_2016-now", "dd_2016-now", "margin", "PASS"]].round(3).to_string(index=False))
df.to_csv("regime_results.csv", index=False)

# --- null calibration: random persistent regimes (same switching style), same pass test ---
rng = np.random.default_rng(7); N = 400; passes = 0; both = 0
for _ in range(N):
    mean_dwell = rng.choice([10, 30, 60]); st = np.zeros(R.n, bool); cur = rng.random() < 0.5
    for t in range(R.n):
        if rng.random() < 1 / mean_dwell: cur = rng.random() < 0.5
        st[t] = cur
    e = R.ev(R.composite(st)); passes += R.passes(e); both += all(e[p][0] > R.BASES["always_trend"][p][0] for p in e)
print(f"\nNULL (random regimes, n={N}): pass-rate {passes/N:.3f} -> expected false passes among {len(df)} rules ~ {passes/N*len(df):.1f}; beat-trend-both rate {both/N:.3f} (~{both/N*len(df):.1f} rules)")
