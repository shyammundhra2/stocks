import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro.paths import model_path

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ===============================
# 1️⃣ Setup & Asset Universe
# ===============================
from macro.constants import TREND_ASSETS

MACRO_TICKERS = {
    "SPY": "SP500"
}

ALL_TICKERS = list(set(list(TREND_ASSETS.keys()) + list(MACRO_TICKERS.keys())))

# ===============================
# 2️⃣ Helper Functions
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
        slopes[i] = slope / y_mean  # normalise by price level
        r2s[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def compute_realised_vol(series, period=21):
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


def compute_atr_pct(series, period=21):
    high_low = series.rolling(2).max() - series.rolling(2).min()
    atr = high_low.rolling(period).mean()
    return atr / series


def rank_features(model, X, y, label, n_splits=5):
    """
    Time-series aware permutation importance
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = []

    for _, test_idx in tscv.split(X):
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        importances.append(result.importances_mean)

    mean_importance = np.mean(importances, axis=0)

    ranking = (
        pd.DataFrame({
            "Feature": X.columns,
            "Importance": mean_importance
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n📊 Feature Importance — {label}")
    print(ranking.to_string(index=False))

    return ranking


def evaluate_model(model, X, y, label):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    print(f"\n📊 {label} Model Evaluation")
    print("===================================")
    print(classification_report(y, preds, target_names=["Down", "Up"]))
    print(f"AUC: {auc:.4f}")
    print("===================================")

    return auc


# ===============================
# 3️⃣ Data Acquisition
# ===============================
print("📡 Downloading market data (25y)...")

raw_data = yf.download(
    ALL_TICKERS,
    period="25y",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if isinstance(raw_data.columns, pd.MultiIndex):
    close_prices = raw_data["Close"]
else:
    close_prices = raw_data

# ===============================
# 4️⃣ Feature Engineering
# ===============================
print("🛠️ Engineering features...")

df = pd.DataFrame(index=close_prices.index)

spy = close_prices["SPY"]

# --- Trend / Technical ---
df["RSI14"] = compute_RSI(spy, 14)
df["SMA50_dist"] = (spy - spy.rolling(50).mean()) / spy.rolling(50).mean()
df["SMA200_dist"] = (spy - spy.rolling(200).mean()) / spy.rolling(200).mean()
df["LR14_slope"], df["LR14_r2"] = compute_linreg_features(spy, 14)

# --- Volatility Regime ---
df["RealVol21"] = compute_realised_vol(spy, 21)
df["ATR21_pct"] = compute_atr_pct(spy, 21)

# ===============================
# 5️⃣ Targets
# ===============================
df["target_f"] = (spy.shift(-5) > spy).astype(int)
df["target_s"] = (spy.shift(-21) > spy).astype(int)

df.dropna(inplace=True)

FEATURE_COLS = [
    "RSI14",
    "SMA50_dist",
    "SMA200_dist",
    "LR14_slope",
    "LR14_r2",
    "RealVol21",
    "ATR21_pct",
]

X = df[FEATURE_COLS]
y_f = df["target_f"]
y_s = df["target_s"]

# ===============================
# 6️⃣ Scaling
# ===============================
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# ===============================
# 7️⃣ Model Training
# ===============================
print("🚀 Training FAST model (5d)...")
model_fast = RandomForestClassifier(
    n_estimators=500,
    max_depth=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model_fast.fit(X_scaled, y_f)

print("🚀 Training SLOW model (21d)...")
model_slow = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model_slow.fit(X_scaled, y_s)

# ===============================
# 8️⃣ Feature Ranking
# ===============================
fast_rank = rank_features(model_fast, X_scaled, y_f, "FAST (5D)")
slow_rank = rank_features(model_slow, X_scaled, y_s, "SLOW (21D)")

# ===============================
# 9️⃣ Model Evaluation
# ===============================
fast_auc = evaluate_model(model_fast, X_scaled, y_f, "FAST (5D)")
slow_auc = evaluate_model(model_slow, X_scaled, y_s, "SLOW (21D)")

# ===============================
# 🔟 Save Bundle
# ===============================
joblib.dump(
    {
        "model_fast": model_fast,
        "model_slow": model_slow,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "fast_feature_rank": fast_rank,
        "slow_feature_rank": slow_rank,
        "fast_auc": fast_auc,
        "slow_auc": slow_auc,
    },
    model_path("trend_model.joblib")
)

print("\n✅ trend_model.joblib saved.")
