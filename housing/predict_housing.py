import os
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# 1) FRED Setup
# ---------------------------------------------
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# ---------------------------------------------
# 2) Macro Series Including Foreclosure Proxy
# ---------------------------------------------
macro_ids = {
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "fed_funds_rate",
    "GS10": "yield_10y",
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "DRSFRMACBS": "mortgage_serious_delinquency_rate"  # foreclosure proxy
}

macro_dfs = []
for sid, name in macro_ids.items():
    s = fred.get_series(sid)
    df_series = s.to_frame(name)
    df_series.index.name = "date"
    macro_dfs.append(df_series)

macro_df = pd.concat(macro_dfs, axis=1).sort_index()
macro_df = macro_df.resample("MS").ffill().bfill()

# ---------------------------------------------
# 3) Home Sales Features
# ---------------------------------------------
sales_ids = {
    "EXHOSLUSM495S": "existing_home_sales",
    "HSN1F": "new_one_family_homes_sold"
}

sales_dfs = []
for sid, name in sales_ids.items():
    s = fred.get_series(sid)
    df_series = s.to_frame(name)
    df_series.index.name = "date"
    sales_dfs.append(df_series)

sales_df = pd.concat(sales_dfs, axis=1).sort_index()
sales_df = sales_df.resample("MS").ffill().bfill()

# ---------------------------------------------
# 4) Case-Shiller Home Price Series
# ---------------------------------------------
price_ids = {
    "SPCS10RSA": "case_shiller_10_city",
    "SPCS20RSA": "case_shiller_20_city",
    "PHXRSA": "phoenix_index",
    "LXXRSA": "los_angeles_index",
    "DNXRSA": "denver_index",
    "SFXRSA": "san_francisco_index",
    "MIXRSA": "miami_index"
}

price_dfs = []
for sid, name in price_ids.items():
    s = fred.get_series(sid)
    df_series = s.to_frame(name)
    df_series.index.name = "date"
    price_dfs.append(df_series)

price_df = pd.concat(price_dfs, axis=1).sort_index()
price_df = price_df.resample("MS").last().ffill().bfill()

# ---------------------------------------------
# 5) Combine Macro, Sales, and Prices
# ---------------------------------------------
df_all = macro_df.join(sales_df, how="inner").join(price_df, how="inner")
df_all = df_all.reset_index()

# ---------------------------------------------
# 6) Melt Data for ML
# ---------------------------------------------
# Use the correct column names for sales (values, not keys)
df_long = df_all.melt(
    id_vars=["date"] + list(macro_ids.values()) + list(sales_ids.values()),  # <-- use values
    value_vars=list(price_ids.values()),
    var_name="metro",
    value_name="price"
)

# ---------------------------------------------
# 7) Compute 5-Year Forward Return per row
# ---------------------------------------------
def compute_5y_return(row):
    future_date = row["date"] + pd.DateOffset(months=60)
    try:
        future_price = df_all.loc[df_all["date"] == future_date, row["metro"]].values[0]
        return (future_price - row["price"]) / row["price"] * 100
    except IndexError:
        return np.nan

df_long["5y_return"] = df_long.apply(compute_5y_return, axis=1)
df_long = df_long.dropna(subset=["5y_return"])

# ---------------------------------------------
# 8) Prepare Features for ML
# ---------------------------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
metro_encoded = encoder.fit_transform(df_long[["metro"]])
metro_cols = encoder.get_feature_names_out(["metro"])
df_metro_enc = pd.DataFrame(metro_encoded, columns=metro_cols, index=df_long.index)

X_numeric = df_long[list(macro_ids.values()) + list(sales_ids.values())]
X = pd.concat([X_numeric.reset_index(drop=True), df_metro_enc.reset_index(drop=True)], axis=1)
y = df_long["5y_return"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---------------------------------------------
# 9) Train RandomForest
# ---------------------------------------------
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict 5-year HPA
preds = model.predict(X)
# Confidence = standard deviation across trees
pred_std = np.std([tree.predict(X) for tree in model.estimators_], axis=0)

df_long["pred_5y_return"] = preds
df_long["pred_confidence"] = pred_std

# ---------------------------------------------
# 10) Rank Latest Predictions by Metro
# ---------------------------------------------
latest_idx = df_long.groupby("metro")["date"].idxmax()
df_pred = df_long.loc[latest_idx, ["metro", "pred_5y_return", "pred_confidence"]]
df_pred = df_pred.sort_values(by="pred_5y_return", ascending=False)

# ---------------------------------------------
# 11) Print Ranked Metro Predictions
# ---------------------------------------------
print("\n--- 5-Year Forward Home Price Appreciation Predictions ---")
print(df_pred.to_string(index=False))
