"""
train_trend_model_oos.py

Same model as train_trend_model.py, with one structural change:
a hard TRAIN_END cutoff. Scaler and both models are fit ONLY on data
through TRAIN_END. Everything after is genuine out-of-sample — usable
for the honest AUC gate and for calibration (run_calibration.py).

Differences from the original script:
  1. TRAIN_END cutoff — no training on the holdout, ever.
  2. Honest evaluation — in-sample AND holdout AUC reported side by side.
     The gap between them is your overfitting measurement.
  3. class_weight is configurable (default None, not "balanced").
     "balanced" deliberately distorts predict_proba away from true
     frequencies — correct for classification, wrong when proba feeds
     Kelly as a literal probability. If you keep "balanced", the
     calibrator must absorb a known distortion; dropping it removes
     the distortion at the source.
  4. Bundle saved with metadata (train_end, class_weight, label spec)
     so run_calibration.py can verify it's replaying true OOS dates.

Output: trend_model_oos.joblib
        (kept separate from production trend_model.joblib — this bundle
         is for measurement; promote it only after you've seen the numbers)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro.constants import TREND_ASSETS
from macro.paths import model_path

# ===============================
# Config
# ===============================
TRAIN_END = "2023-12-31"      # fit on data <= this date, nothing after
CLASS_WEIGHT = None           # None (honest probas) or "balanced" (original)
FAST_HORIZON = 5              # business days
SLOW_HORIZON = 21             # business days — matches original target_s
OUTPUT_PATH = model_path("trend_model_oos.joblib")

MACRO_TICKERS = {"SPY": "SP500"}
ALL_TICKERS = list(set(list(TREND_ASSETS.keys()) + list(MACRO_TICKERS.keys())))


# ===============================
# Feature functions — IDENTICAL to train_trend_model.py.
# If you change these, change run_calibration.py to match.
# ===============================
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_linreg_features(series, period=14):
    slopes = np.full(len(series), np.nan)
    r2s = np.full(len(series), np.nan)
    x = np.arange(period)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    vals = series.values
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1:i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_yy = ((y - y_mean) ** 2).sum()
        slope = ss_xy / ss_xx
        slopes[i] = slope / y_mean
        r2s[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def compute_realised_vol(series, period=21):
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


def compute_atr_pct(series, period=21):
    high_low = series.rolling(2).max() - series.rolling(2).min()
    atr = high_low.rolling(period).mean()
    return atr / series


FEATURE_COLS = [
    "RSI14", "SMA50_dist", "SMA200_dist",
    "LR14_slope", "LR14_r2", "RealVol21", "ATR21_pct",
]


def build_feature_frame(close: pd.Series) -> pd.DataFrame:
    """Feature matrix from a single close series. Shared with calibration."""
    df = pd.DataFrame(index=close.index)
    df["RSI14"] = compute_RSI(close, 14)
    df["SMA50_dist"] = (close - close.rolling(50).mean()) / close.rolling(50).mean()
    df["SMA200_dist"] = (close - close.rolling(200).mean()) / close.rolling(200).mean()
    df["LR14_slope"], df["LR14_r2"] = compute_linreg_features(close, 14)
    df["RealVol21"] = compute_realised_vol(close, 21)
    df["ATR21_pct"] = compute_atr_pct(close, 21)
    return df


# ===============================
# Data
# ===============================
print("Downloading market data (25y)...")
raw_data = yf.download(ALL_TICKERS, period="25y", interval="1d",
                       auto_adjust=True, progress=False)

close_prices = raw_data["Close"] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
spy = close_prices["SPY"].dropna()

# ===============================
# Features & labels (SPY, as in original)
# ===============================
print("Engineering features...")
df = build_feature_frame(spy)
df["target_f"] = (spy.shift(-FAST_HORIZON) > spy).astype(int)
df["target_s"] = (spy.shift(-SLOW_HORIZON) > spy).astype(int)
df.dropna(inplace=True)

# ===============================
# THE CUT — train vs holdout
# ===============================
cutoff = pd.Timestamp(TRAIN_END)
train_mask = df.index <= cutoff
holdout_mask = df.index > cutoff

df_train = df[train_mask]
df_hold = df[holdout_mask]

print(f"\nTrain:   {df_train.index.min().date()} -> {df_train.index.max().date()}"
      f"  ({len(df_train)} rows)")
print(f"Holdout: {df_hold.index.min().date()} -> {df_hold.index.max().date()}"
      f"  ({len(df_hold)} rows)")
print(f"Holdout base rates: fast={df_hold['target_f'].mean():.3f}  "
      f"slow={df_hold['target_s'].mean():.3f}")

X_train = df_train[FEATURE_COLS]
X_hold = df_hold[FEATURE_COLS]

# Scaler fit ON TRAIN ONLY — fitting on full history leaks holdout
# distribution into the features
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                         columns=FEATURE_COLS, index=X_train.index)
X_hold_s = pd.DataFrame(scaler.transform(X_hold),
                        columns=FEATURE_COLS, index=X_hold.index)

# ===============================
# Train
# ===============================
print(f"\nTraining FAST model ({FAST_HORIZON}d), class_weight={CLASS_WEIGHT}...")
model_fast = RandomForestClassifier(n_estimators=500, max_depth=4,
                                    class_weight=CLASS_WEIGHT,
                                    random_state=42, n_jobs=-1)
model_fast.fit(X_train_s, df_train["target_f"])

print(f"Training SLOW model ({SLOW_HORIZON}d), class_weight={CLASS_WEIGHT}...")
model_slow = RandomForestClassifier(n_estimators=500, max_depth=5,
                                    class_weight=CLASS_WEIGHT,
                                    random_state=42, n_jobs=-1)
model_slow.fit(X_train_s, df_train["target_s"])


# ===============================
# Honest evaluation — in-sample vs holdout, side by side
# ===============================
def auc_block(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    print(f"  {label:<22} AUC: {auc:.4f}   "
          f"(n={len(y)}, base_rate={y.mean():.3f}, "
          f"mean_pred={proba.mean():.3f})")
    return auc


print("\n================ HONEST EVALUATION ================")
print("FAST model:")
fast_auc_in = auc_block(model_fast, X_train_s, df_train["target_f"], "in-sample (train)")
fast_auc_oos = auc_block(model_fast, X_hold_s, df_hold["target_f"], "HOLDOUT (honest)")
print("SLOW model:")
slow_auc_in = auc_block(model_slow, X_train_s, df_train["target_s"], "in-sample (train)")
slow_auc_oos = auc_block(model_slow, X_hold_s, df_hold["target_s"], "HOLDOUT (honest)")
print("====================================================")

print("\nInterpretation guide (HOLDOUT AUC, slow model):")
print("  < 0.52  no detectable edge — do not calibrate; move sizing to")
print("          vol-targeting with ML removed or demoted to a tilt")
print("  0.52-0.56  weak edge — usable only as a gate, not a Kelly p;")
print("          calibrate with heavy shrink (0.5-0.7)")
print("  > 0.56  real OOS signal — proceed to run_calibration.py")
print("\nNote: holdout AUC here is daily-overlapping (autocorrelated).")
print("Treat it as descriptive; run_calibration.py uses strided samples.")

# ===============================
# Save bundle with metadata
# ===============================
joblib.dump(
    {
        "model_fast": model_fast,
        "model_slow": model_slow,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "train_end": TRAIN_END,
        "class_weight": CLASS_WEIGHT,
        "fast_horizon": FAST_HORIZON,
        "slow_horizon": SLOW_HORIZON,
        "label_spec": "close.shift(-h) > close (own-series forward return > 0)",
        "fast_auc_insample": fast_auc_in,
        "fast_auc_holdout": fast_auc_oos,
        "slow_auc_insample": slow_auc_in,
        "slow_auc_holdout": slow_auc_oos,
    },
    OUTPUT_PATH,
)
print(f"\nSaved {OUTPUT_PATH} (measurement bundle — not promoted to production).")