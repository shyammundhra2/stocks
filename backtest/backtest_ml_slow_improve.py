"""
Can ml_slow be improved? Two specific, evidence-motivated attempts (not
blind tuning - both come from this session's IC studies):

  1. MACRO FEATURES: the breadth/credit IC study found HYG/LQD level was the
     strongest absolute-direction signal measured anywhere (21d IC -0.094 vs
     ~0 for every technical). None of these are in ml_slow's 7 features.
     Add: VIX z + level, HYG/LQD vs 50dMA + 20d chg, RSP/SPY 20d chg,
     universe breadth (% > 200DMA).
  2. CRASH TARGET: direction fights a 65% up base rate; but the vol cap
     (ml_slow's only sizing role) really needs P(crash). Relabel to
     target = (fwd 21d ret < -5%) and evaluate THAT.

Same honest protocol as backtest_ml_slow_walkforward: annual walk-forward
retrain on strictly-prior data, strided 21d non-overlapping eval, 2011-26.
Variants:
  tech_rf        - baseline 7 tech features, RF (established: 0.51)
  macro_rf       - tech + macro features, RF
  macro_gbm      - tech + macro, HistGradientBoosting
  crash_tech_rf  - crash label, tech features
  crash_macro_rf - crash label, tech + macro
Downloads: universe + BIL-free; ^VIX/HYG/LQD/RSP are safe to co-download
(the corruption bug was FX/yield tickers only).
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

DATA_START, WF_START, END = "2000-01-01", 2011, "2026-08-21"
SLOW_H, STRIDE, CRASH = 21, 21, -0.05
TECH = ["RSI14", "SMA50_dist", "SMA200_dist", "LR14_slope", "LR14_r2", "RealVol21", "ATR21_pct"]
MACRO = ["vix", "vix_z", "hyg_lqd_ma", "hyg_lqd_chg", "rsp_spy_chg", "breadth200"]


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def linreg(series, period=14):
    sl = np.full(len(series), np.nan); r2 = np.full(len(series), np.nan)
    x = np.arange(period); xc = x - x.mean(); ss_xx = (xc ** 2).sum()
    v = series.values
    for i in range(period - 1, len(v)):
        y = v[i - period + 1:i + 1]
        if np.isnan(y).any():
            continue
        ym = y.mean(); sxy = (xc * (y - ym)).sum(); syy = ((y - ym) ** 2).sum()
        sl[i] = (sxy / ss_xx) / ym; r2[i] = (sxy ** 2) / (ss_xx * syy + 1e-9)
    return pd.Series(sl, index=series.index), pd.Series(r2, index=series.index)


def main():
    t0 = time.time()
    tickers = list(TREND_ASSETS.keys())
    dl = tickers + [t for t in ["^VIX", "HYG", "LQD", "RSP"] if t not in tickers]
    print(f"Downloading {len(dl)} tickers ...")
    raw = yf.download(dl, start=DATA_START, end=END, progress=False, auto_adjust=True)
    close = raw["Close"]
    spy = close["SPY"].dropna()
    idx = close.index

    # --- tech features (identical to training) ---
    df = pd.DataFrame(index=spy.index)
    df["RSI14"] = compute_RSI(spy, 14)
    df["SMA50_dist"] = (spy - spy.rolling(50).mean()) / spy.rolling(50).mean()
    df["SMA200_dist"] = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
    df["LR14_slope"], df["LR14_r2"] = linreg(spy, 14)
    lr = np.log(spy / spy.shift(1))
    df["RealVol21"] = lr.rolling(21).std() * np.sqrt(252)
    hl = spy.rolling(2).max() - spy.rolling(2).min()
    df["ATR21_pct"] = hl.rolling(21).mean() / spy

    # --- macro features ---
    vix = close["^VIX"].reindex(spy.index).ffill()
    df["vix"] = vix
    df["vix_z"] = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    hyg_lqd = (close["HYG"] / close["LQD"]).reindex(spy.index).ffill()
    df["hyg_lqd_ma"] = hyg_lqd / hyg_lqd.rolling(50).mean() - 1.0
    df["hyg_lqd_chg"] = hyg_lqd.pct_change(20)
    rsp_spy = (close["RSP"] / close["SPY"]).reindex(spy.index).ffill()
    df["rsp_spy_chg"] = rsp_spy.pct_change(20)
    above = pd.DataFrame({c: (close[c] > close[c].rolling(200).mean()).astype(float)
                          .where(close[c].notna()) for c in tickers})
    df["breadth200"] = above.mean(axis=1).reindex(spy.index)

    df["fwd"] = spy.shift(-SLOW_H) / spy - 1.0
    df["up"] = (df["fwd"] > 0).astype(int)
    df["crash"] = (df["fwd"] < CRASH).astype(int)
    df = df.dropna(subset=TECH + ["fwd"])
    # macro columns start later (HYG 2007) - variants using them drop those rows per-fold
    print(f"Rows: {len(df)}  crash base rate: {df['crash'].mean():.1%}\n")

    def model_for(kind):
        if kind == "gbm":
            return HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                                  learning_rate=0.05, random_state=42)
        return RandomForestClassifier(n_estimators=500, max_depth=5,
                                      class_weight=None, random_state=42, n_jobs=-1)

    variants = [("tech_rf", TECH, "up", "rf"), ("macro_rf", TECH + MACRO, "up", "rf"),
                ("macro_gbm", TECH + MACRO, "up", "gbm"),
                ("crash_tech_rf", TECH, "crash", "rf"),
                ("crash_macro_rf", TECH + MACRO, "crash", "rf")]
    preds = {v[0]: [] for v in variants}

    for yr in range(WF_START, 2027):
        tr_all = df[df.index < pd.Timestamp(f"{yr}-01-01")]
        te_all = df[(df.index >= pd.Timestamp(f"{yr}-01-01"))
                    & (df.index < pd.Timestamp(f"{yr + 1}-01-01"))].iloc[::STRIDE]
        if len(te_all) < 3:
            continue
        for name, feats, label, kind in variants:
            tr = tr_all.dropna(subset=feats)
            te = te_all.dropna(subset=feats)
            if len(tr) < 500 or len(te) < 3 or tr[label].nunique() < 2:
                continue
            sc = StandardScaler().fit(tr[feats])
            m = model_for(kind)
            m.fit(sc.transform(tr[feats]), tr[label])
            p = m.predict_proba(sc.transform(te[feats]))[:, list(m.classes_).index(1)]
            for d, pi, t, fr in zip(te.index, p, te[label], te["fwd"]):
                preds[name].append((yr, pi, t, fr))

    print(f"{'variant':>15s} {'label':>6s} {'pooled AUC':>10s} {'n':>5s}   recent years")
    print("-" * 95)
    for name, feats, label, kind in variants:
        arr = preds[name]
        ps = np.array([x[1] for x in arr]); ts = np.array([x[2] for x in arr])
        yrs_ = np.array([x[0] for x in arr])
        auc = roc_auc_score(ts, ps) if len(set(ts)) > 1 else np.nan
        yby = []
        for yr in range(2015, 2027):
            m = yrs_ == yr
            if m.sum() > 5 and len(set(ts[m])) > 1:
                yby.append(f"{yr % 100:02d}:{roc_auc_score(ts[m], ps[m]):.2f}")
        print(f"{name:>15s} {label:>6s} {auc:>10.3f} {len(arr):>5d}   " + " ".join(yby))

    # economic test for the crash models: vol-cap proxy w = clip(1 - 4*p_crash, 0.25, 1)
    print(f"\nEconomic test (crash models as vol throttle: w=clip(1-4*p_crash,0.25,1), strided, no costs):")
    print(f"{'variant':>15s} {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'avg w':>6s}")
    for name in ["crash_tech_rf", "crash_macro_rf"]:
        arr = preds[name]
        ps = np.array([x[1] for x in arr]); frs = np.array([x[3] for x in arr])
        w = np.clip(1 - 4 * ps, 0.25, 1.0)
        r = w * frs; eq = np.cumprod(1 + r); ppy = 252 / STRIDE
        sh = (r.mean() * ppy) / (r.std() * np.sqrt(ppy))
        print(f"{name:>15s} {sh:>7.2f} {eq[-1] ** (ppy / len(r)) - 1:>7.1%} "
              f"{float((eq / np.maximum.accumulate(eq) - 1).min()):>7.1%} {w.mean():>6.2f}")
    arr = preds["crash_tech_rf"]
    frs = np.array([x[3] for x in arr]); eq = np.cumprod(1 + frs); ppy = 252 / STRIDE
    sh = (frs.mean() * ppy) / (frs.std() * np.sqrt(ppy))
    print(f"{'SPY buy&hold':>15s} {sh:>7.2f} {eq[-1] ** (ppy / len(frs)) - 1:>7.1%} "
          f"{float((eq / np.maximum.accumulate(eq) - 1).min()):>7.1%} {'1.00':>6s}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
