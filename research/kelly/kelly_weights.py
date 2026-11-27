import numpy as np, pandas as pd
from scipy.optimize import minimize
d = pd.read_csv("/Users/riddhisiddhi/stocks/research/kelly/sleeve_monthly_excess.csv", index_col=0)
S = ["stock_sieve", "etf_12_1", "vix_capitulation", "qqq_rsi2", "trend_etf"]; X = d[S]
def kelly(mu, cov, cap=1.0):
    n = len(mu); f = lambda w: -(w @ mu - .5 * w @ cov @ w)
    r = minimize(f, np.ones(n) / n * .5, bounds=[(0, 1)] * n, constraints=[{"type": "ineq", "fun": lambda w: cap - w.sum()}], method="SLSQP")
    return r.x
def shrink(c, a): return (1 - a) * c + a * np.diag(np.diag(c))
mu, cov = X.mean().values, X.cov().values
print("monthly corr:\n", X.corr().round(2).to_string(), "\n")
v = {"raw": (mu, cov), "mu x0.5 (in-sample haircut), cov shrink 30%": (mu * .5, shrink(cov, .3)),
     "stock_sieve mu x0.5 only if bias: all mu x0.35, cov shrink 30%": (mu * .35, shrink(cov, .3))}
res = {}
for k, (m, c) in v.items():
    w = kelly(m, c); res[k] = w
    print(f"{k}\n  full-Kelly weights: " + ", ".join(f"{s}={x:.0%}" for s, x in zip(S, w)) + f"  | cash={1 - w.sum():.0%}")
    print("  half-Kelly:", ", ".join(f"{s}={x / 2:.0%}" for s, x in zip(S, w)), f"| cash={1 - w.sum() / 2:.0%}")
# bootstrap stability (6m blocks), haircut variant
rng = np.random.default_rng(0); n = len(X); B = []; blk = 6
for _ in range(500):
    idx = np.concatenate([np.arange(s, s + blk) % n for s in rng.integers(0, n, n // blk + 1)])[:n]
    Y = X.iloc[idx]; B.append(kelly(Y.mean().values * .5, shrink(Y.cov().values, .3)))
B = np.array(B); print("\nbootstrap (haircut variant) weight 10/50/90 pct:")
for i, s in enumerate(S): print(f"  {s:18s} {np.percentile(B[:, i], 10):.0%} / {np.percentile(B[:, i], 50):.0%} / {np.percentile(B[:, i], 90):.0%}")
# sub-period stability
h = n // 2
for nm, Y in {"first half": X.iloc[:h], "second half": X.iloc[h:]}.items():
    w = kelly(Y.mean().values * .5, shrink(Y.cov().values, .3)); print(nm, ", ".join(f"{s}={x:.0%}" for s, x in zip(S, w)))
# dollars on 500k: haircut full and half
w = res["mu x0.5 (in-sample haircut), cov shrink 30%"]
print("\n$500k (haircut variant): full / half Kelly")
for s, x in zip(S, w): print(f"  {s:18s} ${x*5e5:>9,.0f} / ${x*2.5e5:>9,.0f}")
print(f"  {'cash':18s} ${(1-w.sum())*5e5:>9,.0f} / ${(1-w.sum()/2)*5e5:>9,.0f}")
# realized behaviour of the haircut half-Kelly mix
mix = X.values @ (w / 2); eq = np.cumprod(1 + mix + d["cash_m"].values); print("\nhalf-Kelly mix in-sample: ann excess %.1f%%, vol %.1f%%, maxDD(total incl cash) %.1f%%" % (mix.mean() * 1200, mix.std() * 12 ** .5 * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100))

print("\n--- sensitivity: scale ONLY stock_sieve's mean excess return, others at 0.5x haircut ---")
for k in (1.0, .6, .4, .3, .2):
    m = mu * .5; m[0] = mu[0] * .5 * k; w = kelly(m, shrink(cov, .3))
    print(f"sieve mu x{k:<4} (sieve ann excess {m[0]*12:.0%}): " + ", ".join(f"{s.split('_')[0]}{'_'+s.split('_')[1] if len(s.split('_'))>1 else ''}={x:.0%}" for s, x in zip(S, w)) + f" cash={1-w.sum():.0%}")
