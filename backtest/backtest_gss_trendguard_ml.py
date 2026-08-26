"""
ChatGPT's Trendguard + ML trend-SURVIVAL experiment, run honestly on the GSS ETF
universe. Question: does an ML probability of trend-continuation add edge OVER the
Trendguard rules + Residual/MAX filters, on WALK-FORWARD data, and is it CALIBRATED?

Entry (Trendguard):  SMA50>SMA200 & Close>prior-20d-high & ATR14<2*mean(ATR14,126)
                     & RSI14<70.  Only TG=1 obs enter the model.
Target (survival):   Y=1 iff over next 21d  min(Close/SMA50)>0.99  AND  ret_21>0.
Features: M21/63/126/252, P/SMA20, P/SMA50, SMA50/SMA200, breakout20/60, ATR/px,
          vol63, RSI14, residual-mom & residual-vol (vs SPY), MAX21, VIX-z.
Walk-forward: for each test year, train ONLY on entries whose 21d label completed
          >= 1 month before the test year starts (embargo, no shuffle).
Reports: base rate, OOS AUC (RF, logistic), Brier, calibration table, and the
A/B/C/D economics (avg 21d trade return + win rate) so we see if ML beats rules.
"""
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS

CACHE = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/05364212-73f2-4de4-8a3b-2936e85f384c/scratchpad/tg_ohlc.parquet"
START, END = "2004-01-01", "2026-08-21"
HOLD = 21


def rsi(s, k=14):
    d = s.diff(); up = d.clip(lower=0).rolling(k).mean(); dn = (-d.clip(upper=0)).rolling(k).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def main():
    t0 = time.time()
    import os
    syms = list(TREND_ASSETS)
    if os.path.exists(CACHE):
        raw = pd.read_parquet(CACHE)
    else:
        print(f"downloading OHLC for {len(syms)} + SPY + ^VIX ...")
        raw = yf.download(syms + ["SPY", "^VIX"], start=START, end=END, auto_adjust=True, progress=False)
        raw.to_parquet(CACHE)
    close = raw["Close"]; high = raw["High"]; low = raw["Low"]
    spy = close["SPY"]; spy_ret = spy.pct_change()
    vix = close["^VIX"] if "^VIX" in close.columns else pd.Series(index=close.index, dtype=float)
    vix_z = ((vix - vix.rolling(50).mean()) / vix.rolling(50).std())
    names = [s for s in syms if s in close.columns and close[s].notna().sum() > 400]

    rows = []
    for c in names:
        px = close[c]; hi = high[c]; lo = low[c]
        sma20 = px.rolling(20).mean(); sma50 = px.rolling(50).mean(); sma200 = px.rolling(200).mean()
        high20 = hi.rolling(20).max().shift(1); high60 = hi.rolling(60).max().shift(1)
        tr = pd.concat([hi - lo, (hi - px.shift()).abs(), (lo - px.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        r = px.pct_change()
        beta = r.rolling(126).cov(spy_ret) / spy_ret.rolling(126).var()
        resid = r - beta * spy_ret
        rmom = resid.rolling(126).sum(); rvol = resid.rolling(126).std()
        feat = pd.DataFrame({
            "M21": px / px.shift(21) - 1, "M63": px / px.shift(63) - 1,
            "M126": px / px.shift(126) - 1, "M252": px / px.shift(252) - 1,
            "P_SMA20": px / sma20 - 1, "P_SMA50": px / sma50 - 1,
            "TrendStr": sma50 / sma200 - 1, "BO20": px / high20 - 1, "BO60": px / high60 - 1,
            "ATRpx": atr / px, "vol63": r.rolling(63).std() * np.sqrt(252),
            "RSI": rsi(px, 14), "rmom": rmom, "rvol": rvol,
            "MAX21": r.rolling(21).max(), "VIXz": vix_z.reindex(px.index),
        })
        # Trendguard entry
        tg = ((sma50 > sma200) & (px > high20) & (atr < 2 * atr.rolling(126).mean()) & (rsi(px, 14) < 70))
        # survival target over next HOLD days
        surv = pd.Series(True, index=px.index)
        for k in range(1, HOLD + 1):
            surv &= (px.shift(-k) / sma50.shift(-k) > 0.99)
        y = (surv & (px.shift(-HOLD) / px - 1 > 0)).astype(float)
        fwd = px.shift(-HOLD) / px - 1
        df = feat.copy(); df["y"] = y; df["fwd"] = fwd; df["tg"] = tg
        df["date"] = px.index; df["sym"] = c
        rows.append(df[df["tg"] & df.notna().all(axis=1)])
    panel = pd.concat(rows).dropna()
    panel = panel[panel["date"] <= pd.Timestamp(END) - pd.Timedelta(days=35)]  # need matured labels
    fcols = ["M21", "M63", "M126", "M252", "P_SMA20", "P_SMA50", "TrendStr", "BO20", "BO60",
             "ATRpx", "vol63", "RSI", "rmom", "rvol", "MAX21", "VIXz"]
    print(f"\nTrendguard obs: {len(panel):,}  base continuation rate: {panel['y'].mean():.1%}\n")

    # walk-forward
    oof = []
    for yr in range(2012, 2027):
        cut = pd.Timestamp(f"{yr}-01-01") - pd.Timedelta(days=35)
        tr = panel[panel["date"] <= cut]
        te = panel[(panel["date"] >= pd.Timestamp(f"{yr}-01-01")) & (panel["date"] < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 800 or len(te) < 20 or tr["y"].nunique() < 2:
            continue
        Xtr, ytr = tr[fcols].values, tr["y"].values
        Xte = te[fcols].values
        rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=30,
                                    random_state=42, n_jobs=-1).fit(Xtr, ytr)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        te = te.copy()
        te["p_rf"] = rf.predict_proba(Xte)[:, 1]
        te["p_lr"] = lr.predict_proba(sc.transform(Xte))[:, 1]
        oof.append(te)
    oof = pd.concat(oof)
    y = oof["y"].values
    print(f"Walk-forward OOS obs: {len(oof):,}  ({oof['date'].min().date()} -> {oof['date'].max().date()})\n")
    for m in ["p_rf", "p_lr"]:
        p = oof[m].values
        print(f"  {m}: AUC {roc_auc_score(y, p):.3f}   Brier {brier_score_loss(y, p):.3f}   "
              f"(baseline Brier {np.mean((y-y.mean())**2):.3f})")

    # calibration table (RF)
    print("\nCalibration (RF): predicted bucket -> realized continuation rate")
    oof["bkt"] = pd.cut(oof["p_rf"], [0, .5, .6, .7, .8, 1.0])
    cal = oof.groupby("bkt").agg(n=("y", "size"), pred=("p_rf", "mean"), realized=("y", "mean"),
                                 fwd=("fwd", "mean"))
    for b, r in cal.iterrows():
        print(f"  p in {str(b):12s}  n={int(r['n']):>5d}  predicted {r['pred']:.0%}  "
              f"realized {r['realized']:.0%}  avg21d-ret {r['fwd']:+.2%}")

    # A/B/C/D economics: avg 21d trade return + win rate on the OOS obs
    print("\nA/B/C/D economics (OOS Trendguard entries, avg 21d fwd return & win rate):")
    rv_med, mx_med = oof["rvol"].median(), oof["MAX21"].median()
    variants = {
        "A Trendguard (all)": oof,
        "B TG + ML p>=0.70": oof[oof["p_rf"] >= 0.70],
        "C TG + Resid/MAX filter": oof[(oof["rvol"] <= rv_med) & (oof["MAX21"] <= mx_med)],
        "D TG + Resid/MAX + ML>=0.70": oof[(oof["rvol"] <= rv_med) & (oof["MAX21"] <= mx_med) & (oof["p_rf"] >= 0.70)],
    }
    print(f"  {'variant':30s} {'n':>6s} {'avg21d':>8s} {'win%':>6s} {'contin%':>8s}")
    for lab, d in variants.items():
        if len(d):
            print(f"  {lab:30s} {len(d):>6d} {d['fwd'].mean():>+8.2%} {(d['fwd']>0).mean():>6.0%} {d['y'].mean():>8.0%}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
