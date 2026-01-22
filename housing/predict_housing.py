# train_predict_hpa_20city_composite.py
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
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
    s = fred.get_series(sid, observation_start="1991-01-01")
    df_series = s.to_frame(name)
    df_series.index.name = "date"
    macro_dfs.append(df_series)

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
# 5) Compute 5-year forward returns
# -----------------------------
df_all = macro_df.join(cs20_df).reset_index()
df_all["5y_return"] = (df_all["cs20_index"].shift(-60) - df_all["cs20_index"]) / df_all["cs20_index"] * 100
df_all = df_all.dropna(subset=["5y_return"])
print("Number of training samples:", df_all.shape[0])

# -----------------------------
# 6) Features and target
# -----------------------------
feature_cols = list(macro_ids.values()) + ["cs20_index"]
X = df_all[feature_cols]
y = df_all["5y_return"]

# -----------------------------
# 7) Train RandomForest
# -----------------------------
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_hpa_20city_composite.pkl")

print("✅ Trained HPA model using Case-Shiller 20-City Composite + macro (1991+)")

# -----------------------------
# 8) Predict latest 5-year HPA
# -----------------------------
latest_macro = {name: fred.get_series(sid).iloc[-1] for sid, name in macro_ids.items()}
latest_price = cs20.iloc[-1]
X_pred = pd.DataFrame([{**latest_macro, "cs20_index": latest_price}])

pred = model.predict(X_pred)[0]
confidence = np.std([tree.predict(X_pred.to_numpy()) for tree in model.estimators_])

print("\n--- 5-Year Forward HPA Prediction ---")
print(f"Case-Shiller 20-City Composite: {pred:.2f}% ± {confidence:.2f}%")
