"""Crash-odds research: P(SPY falls >=10% from today's close at any point in next 63 trading days).
Walk-forward logistic (expanding window, yearly refit, 63d purge) vs baselines. Overlapping targets =>
few independent episodes; report AUC/Brier/log-loss by sub-period and per-episode hit rates."""
import os, sys, warnings
import numpy as np, pandas as pd, yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings("ignore")
ROOT = "/Users/riddhisiddhi/stocks"; OUT = os.path.join(ROOT, "research", "crash")
H, DD = 63, -0.10

def load():
    px = yf.download(["SPY", "^VIX", "^VIX3M"], start="1999-01-01", auto_adjust=True, progress=False)["Close"]
    px = px.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
    baa = pdr.DataReader("BAA10Y", "fred", "1999-01-01").squeeze()
    cl = pd.read_parquet(os.path.join(ROOT, ".cache", "selbuffer", "sp500_close.parquet"))
    return px, baa, cl

def build():
    px, baa, cl = load()
    spy = px["SPY"].dropna(); d = pd.DataFrame(index=spy.index)
    d["trend"] = spy / spy.rolling(200).mean() - 1
    d["dd252"] = spy / spy.rolling(252, min_periods=60).max() - 1
    d["mom126"] = spy / spy.shift(126) - 1
    d["rv21"] = np.log(spy).diff().rolling(21).std() * np.sqrt(252)
    d["vix"] = px["VIX"].reindex(d.index).ffill()
    d["lvix"] = np.log(d["vix"])
    d["ts"] = (px["VIX"] / px["VIX3M"]).reindex(d.index).ffill()        # >1 backwardation (2007+)
    b = baa.reindex(d.index).ffill()
    d["baa"] = b; d["dbaa63"] = b - b.shift(63)
    c = cl[[x for x in cl.columns if x != "SPY"]].reindex(d.index).ffill()
    d["breadth"] = (c > c.rolling(200).mean()).sum(1) / c.notna().sum(1).replace(0, np.nan)
    fwd_min = pd.concat([spy.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    d["y"] = (fwd_min / spy - 1 <= DD).astype(float)
    d.loc[spy.shift(-H).isna(), "y"] = np.nan
    return d

SETS = {
    "A trend+vol":        ["trend", "dd252", "rv21", "lvix"],
    "B +credit":          ["trend", "dd252", "rv21", "lvix", "dbaa63", "baa"],
    "C +credit+breadth":  ["trend", "dd252", "rv21", "lvix", "dbaa63", "baa", "breadth"],
    "D +termstruct(07+)": ["trend", "dd252", "rv21", "lvix", "dbaa63", "baa", "breadth", "ts"],
}

def walk(d, cols, first_test=2005):
    dd = d.dropna(subset=cols + ["y"]); preds = pd.Series(np.nan, index=dd.index)
    for yr in range(first_test, dd.index.year.max() + 1):
        te = dd.index.year == yr
        tr = dd.index < (pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=95))   # purge horizon
        if te.sum() == 0 or tr.sum() < 750 or dd.loc[tr, "y"].sum() < 20: continue
        mu, sd = dd.loc[tr, cols].mean(), dd.loc[tr, cols].std()
        m = LogisticRegression(C=0.3, max_iter=1000).fit((dd.loc[tr, cols] - mu) / sd, dd.loc[tr, "y"])
        preds[te] = m.predict_proba((dd.loc[te, cols] - mu) / sd)[:, 1]
    return preds.dropna()

def score(y, p):
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), ll=log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)))

if __name__ == "__main__":
    d = build(); d.to_csv(os.path.join(OUT, "features.csv"))
    print(f"rows {len(d)}  base rate {d['y'].mean():.3f}  {d.index[0].date()}..{d.index[-1].date()}")
    res, allp = [], {}
    # baselines on the same test index (set A rows)
    for name, cols in SETS.items():
        p = walk(d, cols); allp[name] = p
    idx = allp["A trend+vol"].index
    for n in allp: idx = idx.intersection(allp[n].index)
    # restrict to D's range for a like-for-like table, plus A-only full range
    y = d.loc[idx, "y"]
    print(f"\n== like-for-like test {idx[0].date()}..{idx[-1].date()}  n={len(idx)} base {y.mean():.3f}")
    base = pd.Series(d["y"].dropna().loc[:idx[0]].mean(), index=idx)
    out = {"base rate": score(y, base + np.linspace(0, 1e-9, len(idx))) | {"auc": 0.5}}
    out["200DMA gate (trend<0)"] = score(y, (d.loc[idx, "trend"] < 0).astype(float) * 0.5 + 0.1)
    out["VIX level only"] = score(y, 1 / (1 + np.exp(-(d.loc[idx, "lvix"] - 3.2) * 2)) * 0.5)
    for n in allp: out[n] = score(y, allp[n].loc[idx])
    t = pd.DataFrame(out).T; print(t.round(4).to_string()); t.to_csv(os.path.join(OUT, "scores.csv"))
    # sub-periods for best sets
    for n in ["A trend+vol", "C +credit+breadth", "D +termstruct(07+)"]:
        p = allp[n].loc[idx]
        for lab, m in [("2007-14", idx.year <= 2014), ("2015-26", idx.year >= 2015)]:
            if m.sum() > 200 and d.loc[idx[m], "y"].nunique() > 1:
                print(f"  {n:22s} {lab}: AUC {roc_auc_score(d.loc[idx[m],'y'], p[m]):.3f}  mean p {p[m].mean():.3f} actual {d.loc[idx[m],'y'].mean():.3f}")
    # calibration, best-available set D
    p = allp["D +termstruct(07+)"].loc[idx]
    q = pd.qcut(p, 5, duplicates="drop"); cal = pd.DataFrame({"p": p, "y": y}).groupby(q).mean()
    print("\ncalibration D (quintile of predicted):\n", cal.round(3).to_string())
    # decision value: exposure scaled by 1-p vs 200DMA gate (daily, excess ignored: SPY only)
    spy = yf.download("SPY", start="1999-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    r = spy.pct_change().reindex(idx).shift(-1)
    def stats(w, lab):
        x = (w.shift(1).reindex(idx).fillna(0) if False else w) * r; x = x.dropna()
        eq = (1 + x).cumprod(); print(f"  {lab:28s} CAGR {eq.iloc[-1]**(252/len(x))-1:6.1%}  vol {x.std()*np.sqrt(252):5.1%}  Sharpe {x.mean()/x.std()*np.sqrt(252):4.2f}  maxDD {(eq/eq.cummax()-1).min():6.1%}  avg exp {w.reindex(x.index).mean():.2f}")
    print("\n== timing value on SPY (weights known at close t, applied t+1; no costs, 0% cash)")
    stats(pd.Series(1.0, index=idx), "buy&hold")
    g = (d.loc[idx, "trend"] > 0).astype(float); stats(g, "200DMA gate")
    for thr in (0.15, 0.25, 0.35):
        stats((p < thr).astype(float), f"exit if P(crash)>{thr:.0%}")
    stats((1 - p / p.quantile(0.95)).clip(0, 1), "scaled 1-p/q95")
    # exposure-matched random null for the P<0.25 rule
    w = (p < 0.25).astype(float); k = w.mean(); rng = np.random.default_rng(0); sh = []
    for _ in range(500):
        rw = pd.Series((rng.random(len(idx)) < k).astype(float), index=idx); x = (rw * r).dropna(); sh.append(x.mean() / x.std() * np.sqrt(252))
    x = (w * r).dropna(); print(f"  random-exposure null Sharpe mean {np.mean(sh):.2f} p95 {np.percentile(sh,95):.2f} (rule {x.mean()/x.std()*np.sqrt(252):.2f})")
    p.to_csv(os.path.join(OUT, "walkforward_p_D.csv"))
