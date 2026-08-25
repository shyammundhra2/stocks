"""
Can the regime HMM be improved into something non-redundant? The base HMM
fails because it coincides with vol (doesn't lead) and loses to raw vol_z at
crash detection. The ONE mechanism-driven fix: credit spreads (HYG/LQD) lead
equity vol, so a credit feature might make p_crisis actually LEAD. Also test
state count. Blind tuning is skipped - the feature set is the ceiling.

Variants (causal walk-forward, fit each Jan on prior data, endpoint-filtered):
  base3       - [ret, vol_z, vix_z], 3 states           (current)
  credit3     - [ret, vol_z, vix_z, credit_z], 3 states (credit leads?)
  credit4     - same features, 4 states
  credit_only - HMM on [ret, credit_z] (drop the vol features it duplicates)

credit_z = z-score of HYG/LQD ratio 20d change (falling = credit stress).
Bench: raw vol_z crash AUC 0.654, shipped -scalar 0.669. To be an improvement,
a variant must (a) beat vol_z on crash AUC AND (b) LEAD -scalar (corr at k=+1
> corr at k=0). 2011-26.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score

DATA_START, WF_START, END = "2004-01-01", 2011, "2026-08-21"
STRIDE = 21


def fit_hmm(X, n_states):
    from hmmlearn import hmm
    import warnings
    if len(X) < 60:
        return None, None
    tp = np.ones((n_states, n_states)); np.fill_diagonal(tp, 10.0)
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=100, random_state=42, transmat_prior=tp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X)
    means = m.means_
    # crisis = lowest (ret - 0.5*vol_z - 0.5*vix_z); col0=ret, col1=vol_z, col2=vix_z
    sc = {i: means[i, 0] - 0.5 * means[i, 1] - 0.5 * (means[i, 2] if means.shape[1] > 2 else 0)
          for i in range(n_states)}
    return m, int(min(sc, key=sc.get))


def causal_eval(df, feat_cols, n_states, label):
    import warnings
    Xall = df[feat_cols].values
    recs = []
    for yr in range(WF_START, 2027):
        tr = df[df.index < pd.Timestamp(f"{yr}-01-01")]
        te = df[(df.index >= pd.Timestamp(f"{yr}-01-01")) & (df.index < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 500 or len(te) < 3:
            continue
        model, crisis = fit_hmm(tr[feat_cols].values, n_states)
        if model is None:
            continue
        for d in te.index[::STRIDE]:
            pos = df.index.get_loc(d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = model.predict_proba(Xall[:pos + 1])[-1][crisis]
            recs.append((d, float(p)))
    P = pd.DataFrame(recs, columns=["date", "p"]).set_index("date")["p"]
    sub = df.loc[P.index]
    crash = (sub["fwd"] < -0.05).astype(int)
    auc = roc_auc_score(crash, P) if crash.nunique() > 1 else np.nan
    negs = -sub["scalar"]
    c0 = P.corr(negs); c1 = P.corr(negs.shift(-1))
    leads = c1 > c0
    print(f"{label:>12s} {auc:>9.3f} {c0:>7.3f} {c1:>7.3f}   {'LEADS' if leads else 'coincident/lags'}")
    return auc


def main():
    t0 = time.time()
    print("Downloading SPY + VIX + HYG + LQD ...")
    raw = yf.download(["SPY", "^VIX", "HYG", "LQD"], start=DATA_START, end=END,
                      progress=False, auto_adjust=True)["Close"]
    spy = raw["SPY"].dropna(); vix = raw["^VIX"].reindex(spy.index).ffill()
    hyg = raw["HYG"].reindex(spy.index).ffill(); lqd = raw["LQD"].reindex(spy.index).ffill()

    daily = np.log(spy / spy.shift(1))
    ret = np.log(spy / spy.shift(21))
    vol = daily.rolling(20).std() * np.sqrt(252)
    vol_z = (vol - vol.rolling(126).mean()) / vol.rolling(126).std()
    vix_z = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    credit = (hyg / lqd)
    credit_chg = credit.pct_change(20)
    credit_z = (credit_chg - credit_chg.rolling(126).mean()) / credit_chg.rolling(126).std()
    credit_z = -credit_z    # falling credit ratio = stress -> positive stress signal
    fwd = spy.shift(-21) / spy - 1.0
    rv21 = daily.rolling(21).std() * np.sqrt(252)
    scalar = (rv21.rolling(252, min_periods=60).median() / rv21).clip(0.25, 1.0)

    df = pd.DataFrame({"ret": ret, "vol_z": vol_z, "vix_z": vix_z, "credit_z": credit_z,
                       "fwd": fwd, "scalar": scalar}).dropna()

    crash = (df["fwd"] < -0.05).astype(int)
    print(f"\nBench: vol_z crashAUC {roc_auc_score(crash.loc[df.index], df['vol_z']):.3f}  "
          f"-scalar {roc_auc_score(crash, -df['scalar']):.3f}\n")
    print(f"{'variant':>12s} {'crashAUC':>9s} {'corr@0':>7s} {'corr@+1':>7s}   lead?")
    print("-" * 56)
    causal_eval(df, ["ret", "vol_z", "vix_z"], 3, "base3")
    causal_eval(df, ["ret", "vol_z", "vix_z", "credit_z"], 3, "credit3")
    causal_eval(df, ["ret", "vol_z", "vix_z", "credit_z"], 4, "credit4")
    causal_eval(df, ["ret", "credit_z"], 3, "credit_only")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
