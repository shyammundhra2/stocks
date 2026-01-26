import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
import joblib

# ===============================
# 1️⃣ Setup & Asset Universe
# ===============================
from macro.constants import TREND_ASSETS
from sklearn.metrics import accuracy_score


MACRO_TICKERS = {
    "DX-Y.NYB": "DXY",     # Dollar Index
    "^VIX": "VIX",         # Equity Volatility
    "^MOVE": "MOVE",       # Bond Volatility
    "^TNX": "10Y_Yield",   # 10Y Treasury
    "^IRX": "3M_Yield",    # 3M T-Bill
    "HYG": "High_Yield",   # Credit Risk
    "IEF": "Treasuries",   # Risk-Free Proxy
    "XLY": "Disc",         # Consumer Discretionary
    "XLP": "Staples",       # Consumer Staples
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


# ===============================
# 3️⃣ Data Acquisition
# ===============================
print("📡 Downloading market & macro data (25y)...")

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

# --- Macro / Regime ---
df["DXY_mom"] = close_prices["DX-Y.NYB"].pct_change(63)
df["VIX_level"] = close_prices["^VIX"].rolling(21).mean()
df["MOVE_level"] = close_prices["^MOVE"].ffill()
df["Yield_Curve"] = close_prices["^TNX"] - close_prices["^IRX"]
df["Credit_Stress"] = close_prices["HYG"] / close_prices["IEF"]
df["Consumer_Health"] = close_prices["XLY"] / close_prices["XLP"]

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
    "DXY_mom",
    "VIX_level",
    "MOVE_level",
    "Yield_Curve",
    "Credit_Stress",
    "Consumer_Health"
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
# 9️⃣ Save Bundle
# ===============================
joblib.dump(
    {
        "model_fast": model_fast,
        "model_slow": model_slow,
        "scaler": scaler,
        "features": FEATURE_COLS,
        "fast_feature_rank": fast_rank,
        "slow_feature_rank": slow_rank
    },
    "trend_model.joblib"
)


# ------------------ MODEL ACCURACY ------------------

def compute_topk_accuracy(model, X, y, label, k=3):
    proba = model.predict_proba(X)
    classes = model.classes_

    # Top-1
    top1_preds = classes[np.argmax(proba, axis=1)]
    top1_acc = accuracy_score(y, top1_preds)

    # Top-k
    topk_correct = 0
    for i in range(len(y)):
        topk_idx = np.argsort(proba[i])[-k:]
        topk_classes = classes[topk_idx]
        if y.iloc[i] in topk_classes:
            topk_correct += 1
    topk_acc = topk_correct / len(y)

    print(f"\n📊 {label} Model Accuracy")
    print("===================================")
    print(f"✅ Top-1 Accuracy: {top1_acc:.2%}")
    print(f"✅ Top-{k} Accuracy: {topk_acc:.2%}")
    print("===================================")

    return top1_acc, topk_acc


# Compute accuracies
fast_top1, fast_top3 = compute_topk_accuracy(model_fast, X_scaled, y_f, "FAST (5D)")
slow_top1, slow_top3 = compute_topk_accuracy(model_slow, X_scaled, y_s, "SLOW (21D)")

print("\n✅ trend_model.joblib saved with macro-aware feature rankings.")

# ===============================
# 🔟 Macro Danger Zone Alert
# ===============================
current_curve = df["Yield_Curve"].iloc[-1]

if current_curve < 0:
    print("\n🚨 DANGER ZONE: Yield Curve Inverted")
elif 0 < current_curve < 0.2:
    print("\n⚠️ WARNING: Curve De-Inverting — Sahm Watch Active")
else:
    print("\n🟢 Curve Normal — No Macro Stress Signal")
