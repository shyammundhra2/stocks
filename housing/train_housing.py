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
    "DRSFRMACBS": "mortgage_serious_delinquency_rate"  # foreclosure-related proxy
}

macro_dfs = []
for sid, name in macro_ids.items():
    try:
        series = fred.get_series(sid)
        df_series = series.to_frame(name)
        df_series.index.name = "date"
        macro_dfs.append(df_series)
    except Exception as e:
        print(f"Error fetching macro series {sid}: {e}")

macro_df = pd.concat(macro_dfs, axis=1).sort_index()
macro_df = macro_df.resample("MS").ffill().bfill()

# ---------------------------------------------
# 3) Home Sales Features from FRED
# ---------------------------------------------
sales_ids = {
    "EXHOSLUSM495S": "existing_home_sales",
    "HSN1F": "new_one_family_homes_sold"
}

sales_dfs = []
for sid, name in sales_ids.items():
    try:
        series = fred.get_series(sid)
        df_series = series.to_frame(name)
        df_series.index.name = "date"
        sales_dfs.append(df_series)
    except Exception as e:
        print(f"Error fetching sales series {sid}: {e}")

sales_df = pd.concat(sales_dfs, axis=1).sort_index()
sales_df = sales_df.resample("MS").ffill().bfill()

# ---------------------------------------------
# 4) House Price Series from FRED
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
    try:
        series = fred.get_series(sid)
        df_series = series.to_frame(name)
        df_series.index.name = "date"
        price_dfs.append(df_series)
    except Exception as e:
        print(f"Error fetching price series {sid}: {e}")

price_df = pd.concat(price_dfs, axis=1).sort_index()
price_df = price_df.resample("MS").last().ffill().bfill()

# ---------------------------------------------
# 5) Combine Macro, Sales, and Price Data
# ---------------------------------------------
df_all = macro_df.join(sales_df, how="inner").join(price_df, how="inner")
df_all = df_all.reset_index()

# ---------------------------------------------
# 6) Machine Learning Dataset Prep
# ---------------------------------------------
price_cols = list(price_ids.values())
df_long = df_all.melt(
    id_vars=["date"] + list(macro_ids.values()) + list(sales_ids.values()),
    value_vars=price_cols,
    var_name="metro",
    value_name="price"
)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
metro_encoded = encoder.fit_transform(df_long[["metro"]])
metro_cols = encoder.get_feature_names_out(["metro"])
df_metro_enc = pd.DataFrame(metro_encoded, columns=metro_cols, index=df_long.index)

X_numeric = df_long[list(macro_ids.values()) + list(sales_ids.values())]
X = pd.concat(
    [X_numeric.reset_index(drop=True),
     df_metro_enc.reset_index(drop=True)],
    axis=1
)

y = df_long["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importances Including Home Sales and Foreclosure Proxy:")
print(importances.head(15))
