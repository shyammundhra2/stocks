# train_predict_hpa_20city_composite.py
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1) FRED API Setup
# -----------------------------
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# -----------------------------
# 2) Macro series to fetch
# -----------------------------
macro_ids = {
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "fed_funds_rate",
    "GS10": "yield_10y",
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "DRSFRMACBS": "mortgage_serious_delinquency_rate"
}

# -----------------------------
# 3) Fetch macro data (monthly, 1991+)
# -----------------------------
macro_dfs = []
for sid, name in macro_ids.items():
    try:
        s = fred.get_series(sid, observation_start="1991-01-01")
        df_series = s.to_frame(name)
        df_series.index.name = "date"
        macro_dfs.append(df_series)
    except Exception as e:
        print(f"⚠️  Failed to fetch {sid}: {e}")

macro_df = pd.concat(macro_dfs, axis=1).sort_index()
macro_df = macro_df.resample("MS").ffill().bfill()

# -----------------------------
# 4) Fetch Case-Shiller 20-City Composite
# -----------------------------
cs20 = fred.get_series("SPCS20RSA", observation_start="1991-01-01")
cs20_df = cs20.to_frame("cs20_index")
cs20_df.index.name = "date"
cs20_df = cs20_df.resample("MS").ffill().bfill()

# -----------------------------
# 5) Feature engineering
# -----------------------------
df_all = macro_df.join(cs20_df).reset_index()

# Add derived features
df_all["yoy_price_change"] = df_all["cs20_index"].pct_change(12) * 100
df_all["price_momentum_3m"] = df_all["cs20_index"].pct_change(3) * 100
df_all["yield_curve"] = df_all["yield_10y"] - df_all["fed_funds_rate"]
df_all["housing_permits_ma6"] = df_all["building_permits"].rolling(6).mean()

# Target: 5-year forward return
df_all["5y_return"] = (
    (df_all["cs20_index"].shift(-60) - df_all["cs20_index"])
    / df_all["cs20_index"] * 100
)

# Drop rows with NaN
df_all = df_all.dropna()
print(f"Training samples: {df_all.shape[0]}")
print(f"Date range: {df_all['date'].min()} to {df_all['date'].max()}")

# -----------------------------
# 6) Features and target
# -----------------------------
feature_cols = [
    "unemployment_rate", "fed_funds_rate", "yield_10y",
    "housing_starts", "building_permits", "mortgage_serious_delinquency_rate",
    "cs20_index", "yoy_price_change", "price_momentum_3m",
    "yield_curve", "housing_permits_ma6"
]

X = df_all[feature_cols]
y = df_all["5y_return"]

# -----------------------------
# 7) Time-series cross-validation
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(
    RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    X, y, cv=tscv, scoring="neg_mean_absolute_error"
)
print(f"\nCV MAE: {-cv_scores.mean():.2f}% ± {cv_scores.std():.2f}%")

# -----------------------------
# 8) Train final model
# -----------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Feature importance
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📊 Feature Importance:")
print(importances.to_string(index=False))

# Save model and metadata
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_hpa_20city_composite.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

# Calculate training metrics
y_pred_train = model.predict(X)
train_mae = mean_absolute_error(y, y_pred_train)
train_r2 = r2_score(y, y_pred_train)
print(f"\n✅ Model trained | MAE: {train_mae:.2f}% | R²: {train_r2:.3f}")

# -----------------------------
# 9) Predict latest 5-year HPA
# -----------------------------
latest_macro = {}
for sid, name in macro_ids.items():
    try:
        latest_macro[name] = fred.get_series(sid).iloc[-1]
    except:
        latest_macro[name] = np.nan

latest_price = cs20.iloc[-1]
latest_yoy = (cs20.iloc[-1] - cs20.iloc[-13]) / cs20.iloc[-13] * 100
latest_mom3 = (cs20.iloc[-1] - cs20.iloc[-4]) / cs20.iloc[-4] * 100
latest_yield_curve = latest_macro.get("yield_10y", 0) - latest_macro.get("fed_funds_rate", 0)
latest_permits_ma6 = fred.get_series("PERMIT").iloc[-6:].mean()

X_pred = pd.DataFrame([{
    **latest_macro,
    "cs20_index": latest_price,
    "yoy_price_change": latest_yoy,
    "price_momentum_3m": latest_mom3,
    "yield_curve": latest_yield_curve,
    "housing_permits_ma6": latest_permits_ma6
}])

# Prediction with confidence interval
pred = model.predict(X_pred)[0]
tree_preds = np.array([tree.predict(X_pred.to_numpy())[0] for tree in model.estimators_])
confidence_std = np.std(tree_preds)
ci_lower = np.percentile(tree_preds, 10)
ci_upper = np.percentile(tree_preds, 90)

print("\n" + "="*50)
print("📈 5-Year Forward HPA Prediction")
print("="*50)
print(f"Case-Shiller 20-City Composite")
print(f"  Point Estimate: {pred:.2f}%")
print(f"  Std Deviation:  ±{confidence_std:.2f}%")
print(f"  80% CI:         [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"\nCurrent Index:  {latest_price:.2f}")
print(f"YoY Change:     {latest_yoy:.2f}%")
print("="*50)