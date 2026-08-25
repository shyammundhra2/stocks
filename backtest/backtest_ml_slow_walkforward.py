"""
Honest walk-forward evaluation of the ml_slow pipeline + tested improvements.

Review found the production training (train_trends.py) has: no holdout (fit
on full 25y), scaler leak, class_weight="balanced" (distorts predict_proba),
an untested inference-time vol_adj haircut, and overlapping daily labels.
The OOS variant fixed some of this but is one split, never promoted.

This does it properly: WALK-FORWARD - retrain each January on data strictly
before that year, predict that year, evaluate on STRIDED (21d, non-
overlapping) points only. Variants:

  rf_balanced   - production config (RF, class_weight="balanced")
  rf_plain      - RF, class_weight=None (honest probabilities)
  rf_plain_vadj - rf_plain with the live vol_adj haircut applied (tests
                  whether ml_engine's ad-hoc adjustment helps or hurts)
  logistic      - L2 logistic regression baseline (does the forest earn
                  its complexity?)
  always_up     - predict base rate (calibration floor / sanity)

Metrics: strided AUC per year + pooled, hit rate, and a simple economic
test - scale SPY 21d exposure by w=clip((p-0.40)/0.20, 0, 1), strided,
vs buy&hold. Features/labels identical to training scripts. SPY only
(the asset ml_slow is actually used for). 2011-2026 walk-forward.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_START, WF_START, END = "2000-01-01", 2011, "2026-08-21"
SLOW_H, STRIDE = 21, 21
FEATURE_COLS = ["RSI14", "SMA50_dist", "SMA200_dist",
                "LR14_slope", "LR14_r2", "RealVol21", "ATR21_pct"]


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_linreg_features(series, period=14):
    slopes = np.full(len(series), np.nan); r2s = np.full(len(series), np.nan)
    x = np.arange(period); x_mean = x.mean(); ss_xx = ((x - x_mean) ** 2).sum()
    vals = series.values
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1:i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum(); ss_yy = ((y - y_mean) ** 2).sum()
        slopes[i] = (ss_xy / ss_xx) / y_mean
        r2s[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def build_features(close):
    df = pd.DataFrame(index=close.index)
    df["RSI14"] = compute_RSI(close, 14)
    df["SMA50_dist"] = (close - close.rolling(50).mean()) / close.rolling(50).mean()
    df["SMA200_dist"] = (close - close.rolling(200).mean()) / close.rolling(200).mean()
    df["LR14_slope"], df["LR14_r2"] = compute_linreg_features(close, 14)
    lr = np.log(close / close.shift(1))
    df["RealVol21"] = lr.rolling(21).std() * np.sqrt(252)
    hl = close.rolling(2).max() - close.rolling(2).min()
    df["ATR21_pct"] = hl.rolling(21).mean() / close
    return df


def main():
    t0 = time.time()
    print("Downloading SPY ...")
    spy = yf.download("SPY", start=DATA_START, end=END, progress=False,
                      auto_adjust=True)["Close"].squeeze().dropna()
    df = build_features(spy)
    df["target"] = (spy.shift(-SLOW_H) > spy).astype(int)
    df["fwd_ret"] = spy.shift(-SLOW_H) / spy - 1.0
    df = df.dropna()
    print(f"Rows: {len(df)}  ({df.index.min().date()} -> {df.index.max().date()})\n")

    def make_model(kind):
        if kind == "rf_balanced":
            return RandomForestClassifier(n_estimators=500, max_depth=5,
                                          class_weight="balanced", random_state=42, n_jobs=-1)
        if kind in ("rf_plain", "rf_plain_vadj"):
            return RandomForestClassifier(n_estimators=500, max_depth=5,
                                          class_weight=None, random_state=42, n_jobs=-1)
        if kind == "logistic":
            return LogisticRegression(max_iter=2000, C=1.0)
        return None

    variants = ["rf_balanced", "rf_plain", "rf_plain_vadj", "logistic", "always_up"]
    preds = {v: [] for v in variants}   # (date, p, target, fwd_ret)
    years = list(range(WF_START, 2027))

    for yr in years:
        train = df[df.index < pd.Timestamp(f"{yr}-01-01")]
        test_full = df[(df.index >= pd.Timestamp(f"{yr}-01-01"))
                       & (df.index < pd.Timestamp(f"{yr + 1}-01-01"))]
        test = test_full.iloc[::STRIDE]                  # strided, non-overlapping
        if len(train) < 500 or len(test) < 3:
            continue
        scaler = StandardScaler().fit(train[FEATURE_COLS])          # train only
        Xtr = scaler.transform(train[FEATURE_COLS])
        Xte = scaler.transform(test[FEATURE_COLS])
        base = train["target"].mean()
        for v in variants:
            if v == "always_up":
                p = np.full(len(test), base)
            else:
                m = make_model(v); m.fit(Xtr, train["target"])
                p = m.predict_proba(Xte)[:, list(m.classes_).index(1)]
                if v == "rf_plain_vadj":
                    # live vol_adj haircut: p *= 1 - min(ATR21_pct, 0.1)
                    vadj = 1 - np.minimum(test["ATR21_pct"].values, 0.1)
                    p = p * vadj
            for d, pi, t, fr in zip(test.index, p, test["target"], test["fwd_ret"]):
                preds[v].append((yr, d, pi, t, fr))

    print(f"{'variant':>14s} {'pooled AUC':>10s} {'hit@p>0.5':>10s} {'n':>5s}   per-year AUC")
    print("-" * 110)
    for v in variants:
        arr = preds[v]
        ps = np.array([x[2] for x in arr]); ts = np.array([x[3] for x in arr])
        yrs_ = np.array([x[0] for x in arr])
        try:
            auc = roc_auc_score(ts, ps)
        except ValueError:
            auc = np.nan
        hit = ts[ps > 0.5].mean() if (ps > 0.5).sum() > 5 else np.nan
        yby = []
        for yr in years:
            m = yrs_ == yr
            if m.sum() > 5 and len(set(ts[m])) > 1:
                yby.append(f"{yr % 100:02d}:{roc_auc_score(ts[m], ps[m]):.2f}")
        print(f"{v:>14s} {auc:>10.3f} {hit if hit==hit else float('nan'):>10.1%} {len(arr):>5d}   " + " ".join(yby))

    # economic test: SPY exposure = clip((p-0.40)/0.20, 0, 1) at each strided date
    print(f"\nEconomic test (strided {STRIDE}d, w=clip((p-0.40)/0.20,0,1), no costs):")
    print(f"{'variant':>14s} {'Sharpe':>7s} {'CAGR':>7s} {'maxDD':>7s} {'avg w':>6s}")
    for v in variants:
        arr = preds[v]
        ps = np.array([x[2] for x in arr]); frs = np.array([x[4] for x in arr])
        w = np.clip((ps - 0.40) / 0.20, 0, 1)
        r = w * frs
        eq = np.cumprod(1 + r); ppy = 252 / STRIDE
        sh = (r.mean() * ppy) / (r.std() * np.sqrt(ppy)) if r.std() > 0 else np.nan
        cg = eq[-1] ** (ppy / len(r)) - 1
        dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        print(f"{v:>14s} {sh:>7.2f} {cg:>7.1%} {dd:>7.1%} {w.mean():>6.2f}")
    frs = np.array([x[4] for x in preds["always_up"]])
    eq = np.cumprod(1 + frs); ppy = 252 / STRIDE
    sh = (frs.mean() * ppy) / (frs.std() * np.sqrt(ppy))
    print(f"{'SPY buy&hold':>14s} {sh:>7.2f} {eq[-1] ** (ppy / len(frs)) - 1:>7.1%} "
          f"{float((eq / np.maximum.accumulate(eq) - 1).min()):>7.1%} {'1.00':>6s}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
