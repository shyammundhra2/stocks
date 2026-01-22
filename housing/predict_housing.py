# predict_rank_all_markets_improved.py
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# Case-Shiller Metro Areas (using seasonally adjusted where available)
MARKETS = {
    "ATXRNSA": "Atlanta",
    "BOXRNSA": "Boston",
    "CHXRNSA": "Chicago",
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

# Macro indicators
MACRO_SERIES = {
    "UNRATE": "unemployment_rate",
    "FEDFUNDS": "fed_funds_rate",
    "MORTGAGE30US": "mortgage_rate_30y",
    "GS10": "yield_10y",
    "HOUST": "housing_starts",
    "PERMIT": "building_permits",
    "UMCSENT": "consumer_sentiment",
}


# -----------------------------
# Helper Functions
# -----------------------------
def fetch_macro_data(start_date="2000-01-01"):
    """Fetch and process macro indicators"""
    macro_dfs = []
    for sid, name in MACRO_SERIES.items():
        try:
            s = fred.get_series(sid, observation_start=start_date)
            df = s.to_frame(name)
            df.index.name = "date"
            macro_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Failed to fetch {sid}: {e}")

    if not macro_dfs:
        raise ValueError("No macro data fetched")

    macro_df = pd.concat(macro_dfs, axis=1).sort_index()
    macro_df = macro_df.resample("MS").ffill().bfill()
    return macro_df


def create_features(df, price_col="price_index"):
    """Create engineered features"""
    df = df.copy()

    # Price-based features
    df["yoy_return"] = df[price_col].pct_change(12) * 100
    df["mom3_return"] = df[price_col].pct_change(3) * 100
    df["mom6_return"] = df[price_col].pct_change(6) * 100
    df["price_ma12"] = df[price_col].rolling(12).mean()
    df["price_vs_ma12"] = (df[price_col] / df["price_ma12"] - 1) * 100

    # Macro features
    if "yield_10y" in df.columns and "fed_funds_rate" in df.columns:
        df["yield_curve"] = df["yield_10y"] - df["fed_funds_rate"]

    if "mortgage_rate_30y" in df.columns and "fed_funds_rate" in df.columns:
        df["mortgage_spread"] = df["mortgage_rate_30y"] - df["fed_funds_rate"]

    if "building_permits" in df.columns:
        df["permits_ma6"] = df["building_permits"].rolling(6).mean()
        df["permits_yoy"] = df["building_permits"].pct_change(12) * 100

    # Volatility measure
    df["price_volatility_12m"] = df[price_col].pct_change().rolling(12).std() * 100

    return df


def train_ensemble_model(X_train, y_train):
    """Train ensemble of models for better stability"""
    models = {
        'rf': RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gbm': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def get_ensemble_prediction(models, X_pred):
    """Get weighted ensemble prediction"""
    predictions = []
    for model in models.values():
        pred = model.predict(X_pred)[0]
        predictions.append(pred)

    # Use median for robustness
    return np.median(predictions), np.std(predictions)


# -----------------------------
# Main Analysis
# -----------------------------
def main():
    print("Fetching macro indicators...")
    macro_df = fetch_macro_data(start_date="2000-01-01")

    # Get latest macro values
    latest_macro = {name: macro_df[name].iloc[-1] for name in MACRO_SERIES.values() if name in macro_df.columns}

    predictions = []

    print(f"\nTraining models for {len(MARKETS)} markets...\n")

    for series_id, market_name in MARKETS.items():
        try:
            # Fetch market data
            market_series = fred.get_series(series_id, observation_start="2000-01-01")

            if len(market_series) < 120:
                print(f"⚠️  {market_name}: Insufficient data")
                continue

            market_df = market_series.to_frame("price_index")
            market_df.index.name = "date"
            market_df = market_df.resample("MS").ffill().bfill()

            # Merge with macro
            df_all = macro_df.join(market_df).dropna(subset=["price_index"])

            # Create features
            df_all = create_features(df_all)

            # Target: 5-year forward return
            df_all["target_5y"] = (
                    (df_all["price_index"].shift(-60) - df_all["price_index"])
                    / df_all["price_index"] * 100
            )

            # Drop NaN - must drop before selecting features
            df_all = df_all.dropna()

            # Replace any remaining inf values
            df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna()

            if len(df_all) < 60:
                print(f"⚠️  {market_name}: Insufficient training samples ({len(df_all)})")
                continue

            # Define features
            feature_cols = [col for col in df_all.columns
                            if col not in ["date", "target_5y", "price_index", "price_ma12"]]

            X = df_all[feature_cols]
            y = df_all["target_5y"]

            # Final NaN check
            if X.isnull().any().any():
                nan_cols = X.columns[X.isnull().any()].tolist()
                print(f"⚠️  {market_name}: NaN values in features: {nan_cols}")
                continue

            # Check for extreme values in target
            y_mean, y_std = y.mean(), y.std()
            if abs(y_mean) > 100 or y_std > 100:
                print(f"⚠️  {market_name}: Extreme target values (mean={y_mean:.1f}, std={y_std:.1f})")
                continue

            # Train ensemble
            models = train_ensemble_model(X, y)

            # Prepare latest data for prediction
            latest_row = df_all.iloc[[-1]][feature_cols].copy()

            # Ensure no NaN in prediction data
            if latest_row.isnull().any().any():
                print(f"⚠️  {market_name}: NaN in latest data")
                continue

            X_pred = latest_row

            # Get prediction
            pred, pred_std = get_ensemble_prediction(models, X_pred)

            # Calculate confidence intervals using historical residuals
            rf_preds = [tree.predict(X.to_numpy()) for tree in models['rf'].estimators_[:100]]
            residuals = [y.values - pred for pred in rf_preds]
            residual_std = np.mean([np.std(res) for res in residuals])

            ci_lower = pred - 1.28 * residual_std  # 80% CI
            ci_upper = pred + 1.28 * residual_std

            # Get current market stats
            current_price = market_series.iloc[-1]
            yoy_change = (market_series.iloc[-1] / market_series.iloc[-13] - 1) * 100

            predictions.append({
                "Market": market_name,
                "5Y_Forecast_Pct": pred,
                "Uncertainty_Pct": residual_std,
                "CI_Lower_80": ci_lower,
                "CI_Upper_80": ci_upper,
                "Current_Index": current_price,
                "YoY_Change_Pct": yoy_change,
                "Training_N": len(df_all),
                "Historical_Mean_5Y": y_mean,
                "Historical_Std_5Y": y_std
            })

            print(f"✅ {market_name:20s} | Forecast: {pred:6.2f}% | Hist Avg: {y_mean:6.2f}% | YoY: {yoy_change:5.2f}%")

        except Exception as e:
            print(f"❌ {market_name:20s} | Error: {str(e)[:60]}")

    # Create rankings
    if not predictions:
        print("\n❌ No predictions generated")
        return

    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values("5Y_Forecast_Pct", ascending=False).reset_index(drop=True)
    results_df["Rank"] = range(1, len(results_df) + 1)

    # Reorder columns
    results_df = results_df[[
        "Rank", "Market", "5Y_Forecast_Pct", "Uncertainty_Pct",
        "CI_Lower_80", "CI_Upper_80", "YoY_Change_Pct",
        "Historical_Mean_5Y", "Historical_Std_5Y", "Training_N"
    ]]

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/market_rankings_improved.csv", index=False)

    # Display
    print("\n" + "=" * 120)
    print("📊 5-YEAR HOME PRICE APPRECIATION FORECAST - MARKET RANKINGS (IMPROVED)")
    print("=" * 120)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("=" * 120)

    # Summary
    print(f"\n📈 Summary Statistics:")
    print(f"   Mean Forecast:     {results_df['5Y_Forecast_Pct'].mean():6.2f}%")
    print(f"   Median Forecast:   {results_df['5Y_Forecast_Pct'].median():6.2f}%")
    print(f"   Std Dev:           {results_df['5Y_Forecast_Pct'].std():6.2f}%")
    print(
        f"   Range:             {results_df['5Y_Forecast_Pct'].min():6.2f}% to {results_df['5Y_Forecast_Pct'].max():6.2f}%")

    # Top/Bottom 5
    print(f"\n🏆 Top 5 Markets:")
    for _, row in results_df.head(5).iterrows():
        print(f"   {row['Rank']:2.0f}. {row['Market']:20s} {row['5Y_Forecast_Pct']:6.2f}% "
              f"[{row['CI_Lower_80']:6.2f}%, {row['CI_Upper_80']:6.2f}%]")

    print(f"\n⚠️  Bottom 5 Markets:")
    for _, row in results_df.tail(5).iterrows():
        print(f"   {row['Rank']:2.0f}. {row['Market']:20s} {row['5Y_Forecast_Pct']:6.2f}% "
              f"[{row['CI_Lower_80']:6.2f}%, {row['CI_Upper_80']:6.2f}%]")

    # Risk-adjusted ranking
    results_df["Sharpe_Ratio"] = results_df["5Y_Forecast_Pct"] / results_df["Uncertainty_Pct"]
    results_df_sharpe = results_df.sort_values("Sharpe_Ratio", ascending=False).reset_index(drop=True)

    print(f"\n📊 Best Risk-Adjusted Returns (Top 5 by Sharpe Ratio):")
    for idx, row in results_df_sharpe.head(5).iterrows():
        print(f"   {idx + 1}. {row['Market']:20s} Sharpe: {row['Sharpe_Ratio']:5.2f} "
              f"(Return: {row['5Y_Forecast_Pct']:6.2f}%, Risk: {row['Uncertainty_Pct']:5.2f}%)")

    print(f"\n✅ Results saved to: results/market_rankings_improved.csv")


if __name__ == "__main__":
    main()