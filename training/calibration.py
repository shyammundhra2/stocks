"""
calibration.py — combined calibration harness + driver (single file).

v2: passes named DataFrames to scaler/model (kills sklearn feature-name
warnings, enforces column order by name) and suppresses UserWarnings.

Prerequisite: train_trend_model_oos.py has been run, producing
trend_model_oos.joblib with a train_end cutoff.

Usage (save as calibration.py in repo root):
    python calibration.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro.paths import model_path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


# ===========================================================================
# PART 1 — HARNESS (library functions)
# ===========================================================================

def reliability_report(df: pd.DataFrame, n_bins: int = 10) -> str:
    """
    Text reliability report: per-bin predicted vs realized, ECE, Brier,
    and a focused readout on the Kelly-active region (p > 0.50).
    Expects DataFrame with columns 'pred' and 'label'.
    """
    pred = df["pred"].values
    label = df["label"].values
    n = len(pred)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)

    lines = []
    lines.append(f"Samples: {n}  (effective N lower if horizon > stride)")
    lines.append(f"Base rate (realized): {label.mean():.3f}")
    lines.append(f"Brier score: {np.mean((pred - label) ** 2):.4f}")
    lines.append("")
    lines.append(f"{'bin':>12} {'count':>6} {'mean_pred':>10} "
                 f"{'realized':>9} {'gap':>7}")

    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mp = pred[mask].mean()
        rr = label[mask].mean()
        gap = mp - rr
        ece += (cnt / n) * abs(gap)
        lines.append(f"{bins[b]:>5.2f}-{bins[b+1]:<5.2f} {cnt:>6} "
                     f"{mp:>10.3f} {rr:>9.3f} {gap:>+7.3f}")

    lines.append("")
    lines.append(f"ECE (expected calibration error): {ece:.4f}")

    active = pred > 0.50
    if active.sum() >= 20:
        mp = pred[active].mean()
        rr = label[active].mean()
        lines.append("")
        lines.append(f"KELLY-ACTIVE REGION (pred > 0.50): n={int(active.sum())}")
        lines.append(f"  mean predicted: {mp:.3f}   realized: {rr:.3f}   "
                     f"gap: {mp - rr:+.3f}")
        if mp - rr > 0.05:
            lines.append("  >> Model is HOT where you size. Every Kelly "
                         "position is oversized until calibrated.")
        elif rr - mp > 0.05:
            lines.append("  >> Model is conservative where you size — "
                         "leaving edge on the table.")
        else:
            lines.append("  >> Roughly calibrated in the active region.")
    return "\n".join(lines)


@dataclass
class Calibrator:
    """
    Fitted calibration map plus a safety shrink toward 0.5.

    shrink: p_final = 0.5 + (p_cal - 0.5) * shrink. Kelly's loss from
            overestimating p exceeds the loss from underestimating by
            the same amount — shrink hedges the calibrator's own
            sampling error.
    clip:   hard output bounds keeping Kelly finite and sane.
    """
    method: str
    model: object
    shrink: float = 0.90
    clip: tuple = (0.05, 0.90)
    fit_n: int = 0
    fit_brier_raw: float = field(default=np.nan)
    fit_brier_cal: float = field(default=np.nan)

    def calibrate(self, p_raw):
        p_raw = np.atleast_1d(np.asarray(p_raw, dtype=float))
        p_raw = np.clip(p_raw, 1e-6, 1 - 1e-6)

        if self.method == "platt":
            logit = np.log(p_raw / (1 - p_raw))
            z = self.model["a"] * logit + self.model["b"]
            p_cal = 1.0 / (1.0 + np.exp(-z))
        elif self.method == "isotonic":
            p_cal = self.model.predict(p_raw)
        elif self.method == "identity":
            p_cal = p_raw
        else:
            raise ValueError(f"Unknown method {self.method}")

        p_final = 0.5 + (p_cal - 0.5) * self.shrink
        p_final = np.clip(p_final, self.clip[0], self.clip[1])
        return float(p_final[0]) if p_final.size == 1 else p_final


def _fit_platt(pred: np.ndarray, label: np.ndarray) -> dict:
    """Logistic regression on the logit of the raw probability (2 params)."""
    from scipy.optimize import minimize as sp_minimize

    p = np.clip(pred, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))

    def nll(params):
        a, b = params
        z = a * logit + b
        return np.mean(np.logaddexp(0, -z) * label
                       + np.logaddexp(0, z) * (1 - label))

    res = sp_minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return {"a": float(res.x[0]), "b": float(res.x[1])}


def fit_calibrator(pred, label, method: str = "auto",
                   shrink: float = 0.90) -> Calibrator:
    """
    Fit a calibrator on OOS (prediction, label) pairs.
    method="auto": Platt below 1000 samples, isotonic above.
    """
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label, dtype=int)
    n = len(pred)
    if n < 50:
        raise ValueError(
            f"Only {n} OOS samples — too few to calibrate. Extend the "
            "replay window or reduce stride."
        )

    if method == "auto":
        method = "platt" if n < 1000 else "isotonic"

    if method == "platt":
        model = _fit_platt(pred, label)
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        model = IsotonicRegression(out_of_bounds="clip",
                                   y_min=0.01, y_max=0.99)
        model.fit(pred, label)
    else:
        raise ValueError(f"Unknown method {method}")

    cal = Calibrator(method=method, model=model, shrink=shrink, fit_n=n)

    raw_clip = np.clip(pred, 1e-6, 1 - 1e-6)
    cal.fit_brier_raw = float(np.mean((raw_clip - label) ** 2))
    p_cal = np.atleast_1d(cal.calibrate(pred))
    cal.fit_brier_cal = float(np.mean((p_cal - label) ** 2))
    return cal


def save_calibrator(cal: Calibrator, path: str):
    joblib.dump(cal, path)


def load_calibrator(path: str) -> Calibrator:
    """At integration time, import from streams.py:
       from calibration import load_calibrator"""
    return joblib.load(path)


def kelly_sensitivity_table(rr_ratio: float = 2.0,
                            kelly_fraction: float = 0.25) -> str:
    """Fractional-Kelly allocation as a function of p — cost of p-error."""
    lines = [f"rr_ratio (b) = {rr_ratio}, kelly_fraction = {kelly_fraction}",
             f"{'p':>6} {'full_kelly':>11} {'fractional':>11}"]
    for p in np.arange(0.50, 0.76, 0.05):
        q = 1 - p
        k = (rr_ratio * p - q) / rr_ratio
        lines.append(f"{p:>6.2f} {max(k, 0):>11.3f} "
                     f"{max(k * kelly_fraction, 0):>11.3f}")
    return "\n".join(lines)


# ===========================================================================
# PART 2 — DRIVER (runs on `python calibration.py`)
# ===========================================================================

BUNDLE_PATH = model_path("trend_model_oos.joblib")
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib_price_cache.parquet")
CALIBRATOR_OUT = model_path("slow_model_calibrator.joblib")
AUC_GATE = 0.52          # pooled OOS AUC below this -> don't calibrate noise
MIN_HISTORY_DAYS = 260   # ~200 bars for SMA200 + warmup


def main():
    from macro.constants import TREND_ASSETS
    # Feature construction MUST match training exactly — single definition
    # lives in the training script.
    from train_trend_model_oos import build_feature_frame, FEATURE_COLS

    # ---- 1. Load bundle, establish replay window ----
    bundle = joblib.load(BUNDLE_PATH)
    model_slow = bundle["model_slow"]
    scaler = bundle["scaler"]
    horizon = bundle["slow_horizon"]
    train_end = pd.Timestamp(bundle["train_end"])

    print(f"Bundle: train_end={train_end.date()}  slow_horizon={horizon}d  "
          f"class_weight={bundle['class_weight']}")
    print(f"Reported holdout AUC at train time (daily, overlapping): "
          f"{bundle['slow_auc_holdout']:.4f}")

    # ---- 2. Price data — one bulk download, cached ----
    tickers = sorted(set(list(TREND_ASSETS.keys()) + ["SPY"]))

    if os.path.exists(CACHE_PATH):
        print(f"Loading cached prices from {CACHE_PATH}...")
        closes = pd.read_parquet(CACHE_PATH)
    else:
        print(f"Downloading {len(tickers)} tickers (25y, one call)...")
        raw = yf.download(tickers, period="25y", interval="1d",
                          auto_adjust=True, progress=False)
        closes = (raw["Close"] if isinstance(raw.columns, pd.MultiIndex)
                  else raw)
        closes.to_parquet(CACHE_PATH)
        print(f"Cached to {CACHE_PATH} (delete file to refresh).")

    # ---- 3. Walk-forward replay ----
    # Dates strictly after train_end, strided by horizon (labels barely
    # overlap), stopping `horizon` days early so every label resolves.
    all_dates = closes.index
    replay_pool = all_dates[(all_dates > train_end)]
    if len(replay_pool) <= horizon:
        raise SystemExit("Holdout too short — nothing to replay.")
    replay_dates = replay_pool[:-horizon:horizon]

    print(f"\nReplaying {len(replay_dates)} dates x {len(tickers)} tickers "
          f"({replay_dates[0].date()} -> {replay_dates[-1].date()}, "
          f"stride={horizon}d)...")

    rows = []
    for tk in tickers:
        series = closes[tk].dropna()
        if len(series) < MIN_HISTORY_DAYS:
            print(f"  skip {tk}: insufficient history")
            continue

        # Computing features once over the full series is point-in-time
        # safe ONLY because every feature is a backward-looking rolling
        # window — value at t uses data <= t. If a feature ever becomes
        # centered/forward-smoothed/full-series-normalised, this shortcut
        # becomes leakage and per-date truncation is required.
        feats = build_feature_frame(series)

        for ts in replay_dates:
            if ts not in feats.index:
                continue
            x_row = feats.loc[ts, FEATURE_COLS]
            if x_row.isna().any():
                continue

            # Label: asset's OWN forward return over the slow horizon —
            # same rule as training, applied per-asset (the claim Kelly
            # acts on).
            pos = series.index.get_loc(ts)
            if pos + horizon >= len(series):
                continue
            label = int(series.iloc[pos + horizon] > series.iloc[pos])

            # Named DataFrames so scaler/model see the feature names they
            # were fitted with — no sklearn warnings, and column order is
            # enforced by name rather than position.
            x_df = pd.DataFrame([x_row[FEATURE_COLS].values],
                                columns=FEATURE_COLS)
            x_scaled = pd.DataFrame(scaler.transform(x_df),
                                    columns=FEATURE_COLS)
            p = float(model_slow.predict_proba(x_scaled)[0, 1])

            rows.append({"date": ts, "ticker": tk, "pred": p,
                         "label": label})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No samples collected — check data and dates.")

    print(f"Collected {len(df)} (prediction, label) pairs across "
          f"{df['ticker'].nunique()} tickers.")
    print("Caveat: same-date predictions across tickers are cross-"
          "correlated (shared equity beta). Effective N is meaningfully "
          "lower than row count.")

    # ---- 4. THE GATE — pooled OOS AUC + transfer split ----
    pooled_auc = roc_auc_score(df["label"], df["pred"])
    spy_mask = df["ticker"] == "SPY"
    spy_auc = (roc_auc_score(df.loc[spy_mask, "label"],
                             df.loc[spy_mask, "pred"])
               if spy_mask.sum() >= 20
               and df.loc[spy_mask, "label"].nunique() > 1
               else float("nan"))
    nonspy = df[~spy_mask]
    nonspy_auc = (roc_auc_score(nonspy["label"], nonspy["pred"])
                  if len(nonspy) >= 50 and nonspy["label"].nunique() > 1
                  else float("nan"))

    print("\n================ OOS SIGNAL GATE ================")
    print(f"Pooled OOS AUC (strided):   {pooled_auc:.4f}")
    print(f"SPY-only OOS AUC:           {spy_auc:.4f}   "
          f"<- the asset it trained on")
    print(f"Non-SPY OOS AUC:            {nonspy_auc:.4f}   <- transfer test")
    print("=================================================")

    if pooled_auc < AUC_GATE:
        print(f"\nVERDICT: pooled OOS AUC < {AUC_GATE} — no detectable "
              "edge out of sample.")
        print("Do NOT fit a calibrator (it would be fitting noise).")
        print("Recommended path: vol-targeting sizing; demote ml_conf to "
              "a regime tilt or remove from Kelly entirely.")
        return

    if not np.isnan(nonspy_auc) and nonspy_auc < AUC_GATE <= pooled_auc:
        print("\nWARNING: edge exists on SPY but does NOT transfer to "
              "the trend universe (non-SPY AUC below gate).")
        print("Kelly consumes per-asset probabilities — for non-SPY "
              "names those are currently uninformative.")

    # ---- 5. Reliability — pooled + stability splits ----
    print("\n================ RELIABILITY: POOLED ================")
    print(reliability_report(df))

    mid = df["date"].sort_values().iloc[len(df) // 2]
    first, second = df[df["date"] <= mid], df[df["date"] > mid]
    print(f"\n========== SUB-PERIOD A (-> {mid.date()}) ==========")
    print(reliability_report(first))
    print(f"\n========== SUB-PERIOD B ({mid.date()} ->) ==========")
    print(reliability_report(second))

    def active_gap(d):
        a = d[d["pred"] > 0.5]
        return ((a["pred"].mean() - a["label"].mean())
                if len(a) >= 20 else float("nan"))

    gap_a, gap_b = active_gap(first), active_gap(second)
    print(f"\nKelly-active gap stability:  A={gap_a:+.3f}   B={gap_b:+.3f}")
    unstable = (not np.isnan(gap_a) and not np.isnan(gap_b)
                and (np.sign(gap_a) != np.sign(gap_b))
                and (abs(gap_a) > 0.03 or abs(gap_b) > 0.03))
    if unstable:
        print(">> Gap flips sign across sub-periods — miscalibration is "
              "regime-dependent.")
        print(">> A fitted calibrator chases a moving target. Prefer hard "
              "shrink (p = 0.5 + (raw-0.5)*0.5) or vol-targeting.")

    # ---- 6. Fit calibrator (shrink scaled to evidence quality) ----
    shrink = 0.60 if unstable or pooled_auc < 0.56 else 0.85
    print(f"\nFitting calibrator (auto method, shrink={shrink})...")
    cal = fit_calibrator(df["pred"].values, df["label"].values,
                         shrink=shrink)
    print(f"Method: {cal.method}   fit_n: {cal.fit_n}")
    print(f"Brier raw -> calibrated: {cal.fit_brier_raw:.4f} -> "
          f"{cal.fit_brier_cal:.4f}")

    print("\nRaw -> calibrated (what Kelly will actually see):")
    for p in [0.45, 0.55, 0.60, 0.65, 0.70, 0.75]:
        print(f"  {p:.2f} -> {cal.calibrate(p):.3f}")

    save_calibrator(cal, CALIBRATOR_OUT)
    print(f"\nSaved {CALIBRATOR_OUT}.")
    print("\nIntegration (streams.py, _compute_kelly_size):")
    print('    from calibration import load_calibrator')
    print('    _CAL = load_calibrator("slow_model_calibrator.joblib")'
          '  # module level')
    print("    p = _CAL.calibrate(ml_conf_slow / 100.0)"
          "   # replaces /100.0 line")
    print("Leave raw ml_conf_slow in the status thresholds and dashboard "
          "— those were tuned on raw scores.")


if __name__ == "__main__":
    main()