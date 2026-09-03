"""
Walk-forward accuracy check for get_regime_hmm_state (macro/indicators/regime.py).

Reuses the actual _fit_regime_hmm() from production so this tests the fitting
code exactly as it runs today. (The stale shift(63)-vs-shift(21) bug this
docstring used to warn about was fixed in the source on 2026-06-18.)

IMPORTANT SCOPE LIMIT: this builds its OWN feature matrix over START..END
(~2500 obs), so it exercises _fit_regime_hmm on abundant data and cannot
reproduce production's data supply. Until 2026-09-02 production fed the same
function only 108 observations (1y shared store minus vol_z's 146-bar warm-up),
which is a materially different and much worse problem than anything measured
here - see the _get_hmm_data docstring. This file validates the FIT; it does
not validate what production hands the fit.

Ground truth is *not* the internal ML composite (too expensive/leaky to
reproduce walk-forward here) - instead we use two simple, hard-to-dispute
market-stress proxies:
    - SPY drawdown from trailing peak > 15%
    - VIX > 30

For each day in the test window:
    - refit the HMM every 7 calendar days (matches _HMM_REFIT_INTERVAL),
      using only data available up to and including that day (no lookahead)
    - recompute predict_proba on the full expanding history up to that day
      (matches production, which always recomputes on the full X)
    - record p_crisis and most_likely state

Then report:
    - confusion matrix / precision / recall / F1 for "most_likely==crisis"
      vs. each ground-truth proxy
    - cross-correlation of p_crisis against the drawdown proxy at lags
      -30..+30 trading days, to see whether p_crisis leads or lags
    - average dwell time (consecutive days) per state

Last run 2026-09-02 (2439 daily reads, 2016-08..2026-07):
    drawdown>15% : precision 0.162  recall 0.583  F1 0.253  accuracy 0.709
    VIX>30       : precision 0.160  recall 0.773  F1 0.265  accuracy 0.730
    lead/lag     : peak corr 0.214 at lag +9 sessions (p_crisis leads)
    dwell (mean) : 16.5 crisis / 17.4 trending_calm / 9.4 choppy
The old "state dwell ~4 obs" claim this file was written to check turned out to
be ~3x too pessimistic. High recall, low precision: it catches most real stress
but fires ~5 false alarms per real event, so it stays telemetry.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.indicators import _fit_regime_hmm

START = "2016-01-01"
END = "2026-07-10"
REFIT_INTERVAL_DAYS = 7  # calendar days, matches _HMM_REFIT_INTERVAL
BURN_IN = 60             # min obs before first fit, matches production gate


def download():
    print("Downloading SPY, ^VIX ...")
    raw = yf.download(["SPY", "^VIX"], start=START, end=END, progress=False)
    close = raw["Close"].dropna()
    return close["SPY"], close["^VIX"]


def build_features(spy, vix):
    daily_ret = np.log(spy / spy.shift(1))
    ret_trend = np.log(spy / spy.shift(21))  # fixed to match production

    vol = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_mean = vol.rolling(126).mean()
    vol_std = vol.rolling(126).std()
    vol_z = (vol - vol_mean) / vol_std

    vix_mean = vix.rolling(50).mean()
    vix_std = vix.rolling(50).std()
    vix_z = (vix - vix_mean) / vix_std

    df = pd.DataFrame({"ret": ret_trend, "vol_z": vol_z, "vix_z": vix_z}).dropna()
    return df


def walk_forward(df):
    results = []
    model, labels, last_fit_date = None, None, None

    dates = df.index
    for i, dt in enumerate(dates):
        if i + 1 < BURN_IN:
            continue

        X_so_far = df.iloc[: i + 1].values

        needs_refit = (
            model is None
            or last_fit_date is None
            or (dt - last_fit_date).days > REFIT_INTERVAL_DAYS
        )

        if needs_refit:
            model, labels = _fit_regime_hmm(X_so_far, n_states=3, n_iter=100)
            if model is None:
                continue
            last_fit_date = dt

        probs_seq = model.predict_proba(X_so_far)
        current = probs_seq[-1]
        state_probs = {labels[j]: float(p) for j, p in enumerate(current)}
        most_likely = labels[int(np.argmax(current))]

        results.append({
            "date": dt,
            "p_crisis": state_probs.get("crisis", 0.0),
            "p_choppy": state_probs.get("choppy", 0.0),
            "p_calm": state_probs.get("trending_calm", 0.0),
            "most_likely": most_likely,
        })

    return pd.DataFrame(results).set_index("date")


def dwell_times(most_likely_series):
    dwells = []
    cur_state = None
    cur_len = 0
    for s in most_likely_series:
        if s == cur_state:
            cur_len += 1
        else:
            if cur_state is not None:
                dwells.append((cur_state, cur_len))
            cur_state = s
            cur_len = 1
    if cur_state is not None:
        dwells.append((cur_state, cur_len))
    return pd.DataFrame(dwells, columns=["state", "length"])


def confusion(pred, truth):
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
             "precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def main():
    t0 = time.time()
    spy, vix = download()
    feat = build_features(spy, vix)
    print(f"Feature window: {feat.index[0].date()} to {feat.index[-1].date()}, {len(feat)} obs")

    wf = walk_forward(feat)
    print(f"Walk-forward produced {len(wf)} daily regime reads in {time.time()-t0:.0f}s")

    # ground truth proxies, aligned to wf's index
    drawdown = spy / spy.cummax() - 1.0
    drawdown = drawdown.reindex(wf.index)
    vix_aligned = vix.reindex(wf.index)

    truth_dd = (drawdown < -0.15).astype(int)
    truth_vix = (vix_aligned > 30).astype(int)

    pred_crisis = (wf["most_likely"] == "crisis").astype(int)

    print("\n--- Dwell times (consecutive days in same most-likely state) ---")
    dw = dwell_times(wf["most_likely"].tolist())
    print(dw.groupby("state")["length"].agg(["mean", "median", "count"]))

    print("\n--- Confusion: predicted crisis vs. drawdown>15% ---")
    print(confusion(pred_crisis, truth_dd))

    print("\n--- Confusion: predicted crisis vs. VIX>30 ---")
    print(confusion(pred_crisis, truth_vix))

    print("\n--- Lead/lag cross-correlation: p_crisis vs drawdown<-15% indicator ---")
    best_lag, best_corr = None, -2
    for lag in range(-30, 31):
        shifted = truth_dd.shift(-lag)  # positive lag: does p_crisis(t) predict truth(t+lag)?
        valid = wf["p_crisis"].notna() & shifted.notna()
        if valid.sum() < 30:
            continue
        corr = np.corrcoef(wf["p_crisis"][valid], shifted[valid])[0, 1]
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    print(f"Best correlation {best_corr:.3f} at lag={best_lag} "
          f"({'p_crisis leads' if best_lag and best_lag > 0 else 'p_crisis lags/coincident'})")

    # Was hardcoded to a session scratchpad path that no longer exists, so this
    # silently wrote nothing useful. Write next to the script instead.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hmm_backtest_output.csv")
    wf.to_csv(out)
    print(f"\nSaved daily series to {out}")


if __name__ == "__main__":
    main()
