"""
Evaluate the regime HMM honestly. It's telemetry-only today; its own
promotion criterion (regime.py) is: does p_crisis LEAD the existing
composite/vol signal, or only confirm it? Plus this session's standing
question: does the 3-state Gaussian HMM beat its own raw inputs (vol_z,
vix_z), given crash risk is already ~= current vol?

Protocol (CAUSAL - no lookahead):
  - Features exactly as regime.py: ret=21d log return, vol_z (20d rv vs
    126d mean/std), vix_z (VIX vs 50d mean/std).
  - Refit the HMM each January on data STRICTLY BEFORE that year (same
    sticky prior, diag cov, 3 states, sorted so lowest score = crisis).
  - For strided dates in-year, p_crisis = predict_proba(X[:t+1])[-1][crisis]
    - at a subsequence endpoint, smoothed == filtered, so this is causal.
  - Compare p_crisis vs raw vix_z and vol_z as predictors of:
      crash21  = fwd 21d SPY return < -5%
      fwd 21d return (IC)
  - Lead/lag: does p_crisis rise BEFORE the shipped vol-target scalar drops?
    (cross-correlate d(p_crisis) with future d(-scalar)).
2011-2026.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score

DATA_START, WF_START, END = "2004-01-01", 2011, "2026-08-21"
STRIDE = 21


def fit_hmm(X, n_states=3):
    from hmmlearn import hmm
    if len(X) < 60:
        return None, None
    tp = np.ones((n_states, n_states)); np.fill_diagonal(tp, 10.0)
    m = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=100, random_state=42, transmat_prior=tp)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X)
    means = m.means_
    scores = {i: means[i, 0] - 0.5 * means[i, 1] - 0.5 * means[i, 2] for i in range(n_states)}
    order = sorted(scores, key=scores.get)
    crisis_state = int(order[0])
    return m, crisis_state


def main():
    t0 = time.time()
    print("Downloading SPY + VIX ...")
    raw = yf.download(["SPY", "^VIX"], start=DATA_START, end=END, progress=False, auto_adjust=True)["Close"]
    spy = raw["SPY"].dropna(); vix = raw["^VIX"].reindex(spy.index).ffill()

    daily = np.log(spy / spy.shift(1))
    ret = np.log(spy / spy.shift(21))
    vol = daily.rolling(20).std() * np.sqrt(252)
    vol_z = (vol - vol.rolling(126).mean()) / vol.rolling(126).std()
    vix_z = (vix - vix.rolling(50).mean()) / vix.rolling(50).std()
    fwd = spy.shift(-21) / spy - 1.0
    # shipped vol-target scalar (for lead/lag)
    rv21 = daily.rolling(21).std() * np.sqrt(252)
    scalar = (rv21.rolling(252, min_periods=60).median() / rv21).clip(0.25, 1.0)

    df = pd.DataFrame({"ret": ret, "vol_z": vol_z, "vix_z": vix_z,
                       "fwd": fwd, "scalar": scalar}).dropna()
    Xall = df[["ret", "vol_z", "vix_z"]].values

    recs = []
    import warnings
    for yr in range(WF_START, 2027):
        tr = df[df.index < pd.Timestamp(f"{yr}-01-01")]
        te = df[(df.index >= pd.Timestamp(f"{yr}-01-01")) & (df.index < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 500 or len(te) < 3:
            continue
        model, crisis = fit_hmm(tr[["ret", "vol_z", "vix_z"]].values)
        if model is None:
            continue
        # causal p_crisis at each strided date: predict_proba on the sequence
        # from data start up to that date; endpoint gamma == filtered posterior
        te_dates = te.index[::STRIDE]
        for d in te_dates:
            pos = df.index.get_loc(d)
            seq = Xall[:pos + 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = model.predict_proba(seq)[-1][crisis]
            recs.append((d, float(p), float(df.loc[d, "vix_z"]), float(df.loc[d, "vol_z"]),
                         float(df.loc[d, "fwd"]), float(df.loc[d, "scalar"])))

    R = pd.DataFrame(recs, columns=["date", "p_crisis", "vix_z", "vol_z", "fwd", "scalar"]).set_index("date")
    crash = (R["fwd"] < -0.05).astype(int)

    print(f"\nCausal walk-forward, {len(R)} strided obs, crash rate {crash.mean():.1%}\n")
    print(f"{'predictor':>12s} {'crashAUC':>9s} {'fwd IC':>8s}")
    print("-" * 34)
    for name, s in [("p_crisis", R["p_crisis"]), ("vix_z", R["vix_z"]),
                    ("vol_z", R["vol_z"]), ("-scalar", -R["scalar"])]:
        auc = roc_auc_score(crash, s) if crash.nunique() > 1 else np.nan
        ic = s.rank().corr(R["fwd"].rank())
        print(f"{name:>12s} {auc:>9.3f} {ic:>8.3f}")

    # does the HMM add anything over its inputs? incremental AUC
    print("\nIncremental value of the HMM over raw inputs:")
    print(f"  p_crisis AUC {roc_auc_score(crash, R['p_crisis']):.3f} vs "
          f"max(vix_z,vol_z) AUC {max(roc_auc_score(crash, R['vix_z']), roc_auc_score(crash, R['vol_z'])):.3f}")
    print(f"  corr(p_crisis, vix_z)={R['p_crisis'].corr(R['vix_z']):.2f}  "
          f"corr(p_crisis, vol_z)={R['p_crisis'].corr(R['vol_z']):.2f}")

    # lead/lag: does p_crisis lead the vol-target scalar dropping (risk rising)?
    # correlate p_crisis[t] with -scalar at t+k (k>0 = p_crisis leads)
    print("\nLead/lag vs shipped vol-target scalar (risk = -scalar):")
    print("  k(obs)  corr(p_crisis[t], -scalar[t+k])   (k>0: HMM leads, k<0: lags)")
    neg_scalar = -R["scalar"]
    for k in [-2, -1, 0, 1, 2]:
        c = R["p_crisis"].corr(neg_scalar.shift(-k))
        tag = "LEADS" if k > 0 else ("lags" if k < 0 else "coincident")
        print(f"    {k:>+3d}   {c:>7.3f}   {tag}")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
