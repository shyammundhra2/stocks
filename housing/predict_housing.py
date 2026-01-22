# predict_rank_all_markets.py
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
# 2) Case-Shiller Metro Areas
# -----------------------------
# All available Case-Shiller MSA series
cs_markets = {
    # 20-City Composite metros
    "ATXRNSA": "Atlanta",
    "BOXRNSA": "Boston",
    "CHXRNSA": "Chicago",
    "CXXRNSA": "Charlotte",
    "SDXRNSA": "San Diego",
    "DAXRNSA": "Dallas",
    "DNXRNSA": "Denver",
    "DEXRNSA": "Detroit",
    "LXXRNSA": "Los Angeles",
    "MIXRNSA": "Miami",
    "MNXRNSA": "Minneapolis",
    "NYXRNSA": "New York",
    "PHXRNSA": "Phoenix",
    "POXRNSA": "Portland",
    "LVXRNSA": "Las Vegas",
    "SFXRNSA": "San Francisco",
    "SEXRNSA": "Seattle",
    "TPXRNSA": "Tampa",
    "WDXRNSA": "Washington DC",
    "CEXRNSA": "Cleveland",

    # Additional markets
    "SEXRSA": "Seattle (SA)",
    "SFXRSA": "San Francisco (SA)",
    "LXXRSA": "Los Angeles (SA)",
    "SDXRSA": "San Diego (SA)",
}

# Use the NSA (non-seasonally adjusted) versions primarily
priority_markets = {
    "ATXRNSA": "Atlanta",
    "BOXRNSA": "Boston",
    "CHXRNSA": "Chicago",
    "CXXRNSA": "Charlotte",
    "SDXRNSA": "San Diego",
    "DAXRNSA": "Dallas",
    "DNXRNSA": "Denver",
    "DEXRNSA": "Detroit",
    "LXXRNSA": "Los Angeles",
    "MIXRNSA": "Miami",
    "MNXRNSA": "Minneapolis",
    "NYXRNSA": "New York",
    "PHXRNSA": "Phoenix",
    "POXRNSA": "Portland",
    "LVXRNSA": "Las Vegas",
    "SFXRNSA": "San Francisco",
    "SEXRNSA": "Seattle",
    "TPXRNSA": "Tampa",
    "WDXRNSA": "Washington DC",
    "CEXRNSA": "Cleveland",
}

# -----------------------------
# 3) Macro series to fetch
# -----------------------------
macro_ids = {
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "fed_funds_rate",
    "GS10": "yield_10y",
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "DRSFRMACBS": "mortgage_serious_delinquency_rate"
}

print("Fetching macro indicators...")
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

# Fetch latest macro values
latest_macro = {}
for sid, name in macro_ids.items():
    try:
        latest_macro[name] = fred.get_series(sid).iloc[-1]
    except:
        latest_macro[name] = np.nan

latest_yield_curve = latest_macro.get("yield_10y", 0) - latest_macro.get("fed_funds_rate", 0)
latest_permits_ma6 = fred.get_series("PERMIT").iloc[-6:].mean()

# -----------------------------
# 4) Train models and predict for each market
# -----------------------------
predictions = []

print(f"\nTraining models for {len(priority_markets)} markets...\n")

for series_id, market_name in priority_markets.items():
    try:
        # Fetch market data
        market_series = fred.get_series(series_id, observation_start="1991-01-01")

        if len(market_series) < 100:
            print(f"⚠️  {market_name}: Insufficient data")
            continue

        market_df = market_series.to_frame("price_index")
        market_df.index.name = "date"
        market_df = market_df.resample("MS").ffill().bfill()

        # Merge with macro data
        df_all = macro_df.join(market_df).reset_index()

        # Feature engineering
        df_all["yoy_price_change"] = df_all["price_index"].pct_change(12) * 100
        df_all["price_momentum_3m"] = df_all["price_index"].pct_change(3) * 100
        df_all["yield_curve"] = df_all["yield_10y"] - df_all["fed_funds_rate"]
        df_all["housing_permits_ma6"] = df_all["building_permits"].rolling(6).mean()

        # Target: 5-year forward return
        df_all["5y_return"] = (
                (df_all["price_index"].shift(-60) - df_all["price_index"])
                / df_all["price_index"] * 100
        )

        # Drop NaN
        df_all = df_all.dropna()

        if len(df_all) < 50:
            print(f"⚠️  {market_name}: Insufficient training samples")
            continue

        # Features and target
        feature_cols = [
            "unemployment_rate", "fed_funds_rate", "yield_10y",
            "housing_starts", "building_permits", "mortgage_serious_delinquency_rate",
            "price_index", "yoy_price_change", "price_momentum_3m",
            "yield_curve", "housing_permits_ma6"
        ]

        X = df_all[feature_cols]
        y = df_all["5y_return"]

        # Train model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

        # Get latest values for prediction
        latest_price = market_series.iloc[-1]
        latest_yoy = (market_series.iloc[-1] - market_series.iloc[-13]) / market_series.iloc[-13] * 100
        latest_mom3 = (market_series.iloc[-1] - market_series.iloc[-4]) / market_series.iloc[-4] * 100

        X_pred = pd.DataFrame([{
            **latest_macro,
            "price_index": latest_price,
            "yoy_price_change": latest_yoy,
            "price_momentum_3m": latest_mom3,
            "yield_curve": latest_yield_curve,
            "housing_permits_ma6": latest_permits_ma6
        }])

        # Prediction with confidence
        pred = model.predict(X_pred)[0]
        tree_preds = np.array([tree.predict(X_pred.to_numpy())[0] for tree in model.estimators_])
        confidence_std = np.std(tree_preds)
        ci_lower = np.percentile(tree_preds, 10)
        ci_upper = np.percentile(tree_preds, 90)

        predictions.append({
            "Market": market_name,
            "5Y_HPA_Forecast": pred,
            "Std_Dev": confidence_std,
            "CI_Lower_10pct": ci_lower,
            "CI_Upper_90pct": ci_upper,
            "Current_Index": latest_price,
            "YoY_Change": latest_yoy,
            "Training_Samples": len(df_all)
        })

        print(f"✅ {market_name:20s} | Forecast: {pred:6.2f}% | YoY: {latest_yoy:6.2f}%")

    except Exception as e:
        print(f"❌ {market_name:20s} | Error: {str(e)[:50]}")

# -----------------------------
# 5) Create rankings
# -----------------------------
if predictions:
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values("5Y_HPA_Forecast", ascending=False).reset_index(drop=True)
    results_df["Rank"] = range(1, len(results_df) + 1)

    # Reorder columns
    results_df = results_df[[
        "Rank", "Market", "5Y_HPA_Forecast", "Std_Dev",
        "CI_Lower_10pct", "CI_Upper_90pct", "YoY_Change",
        "Current_Index", "Training_Samples"
    ]]

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/market_rankings.csv", index=False)

    # Display rankings
    print("\n" + "=" * 100)
    print("📊 5-YEAR HOME PRICE APPRECIATION FORECAST - MARKET RANKINGS")
    print("=" * 100)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 100)

    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Average 5Y HPA Forecast: {results_df['5Y_HPA_Forecast'].mean():.2f}%")
    print(f"   Median:                  {results_df['5Y_HPA_Forecast'].median():.2f}%")
    print(f"   Std Deviation:           {results_df['5Y_HPA_Forecast'].std():.2f}%")
    print(
        f"   Range:                   {results_df['5Y_HPA_Forecast'].min():.2f}% to {results_df['5Y_HPA_Forecast'].max():.2f}%")

    # Top and bottom 5
    print(f"\n🏆 Top 5 Markets:")
    for idx, row in results_df.head(5).iterrows():
        print(f"   {row['Rank']}. {row['Market']:20s} {row['5Y_HPA_Forecast']:6.2f}% (±{row['Std_Dev']:.2f}%)")

    print(f"\n⚠️  Bottom 5 Markets:")
    for idx, row in results_df.tail(5).iterrows():
        print(f"   {row['Rank']}. {row['Market']:20s} {row['5Y_HPA_Forecast']:6.2f}% (±{row['Std_Dev']:.2f}%)")

    print(f"\n✅ Results saved to: results/market_rankings.csv")
else:
    print("\n❌ No predictions generated")