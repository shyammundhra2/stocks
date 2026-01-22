# predict_rank_all_markets_with_rent.py
import os
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# Case-Shiller Metro Areas (20 major metros)
CS_MARKETS = {
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

# FHFA House Price Index - Additional metros (Quarterly data)
# Format: ATNHPIUS##### where ##### is the MSA code
FHFA_MARKETS = {
    "ATNHPIUS39580Q": "Raleigh",  # Raleigh-Cary, NC
    "ATNHPIUS34980Q": "Nashville",  # Nashville-Davidson-Murfreesboro, TN
    "ATNHPIUS12420Q": "Austin",  # Austin-Round Rock, TX
    "ATNHPIUS40900Q": "Sacramento",  # Sacramento-Roseville-Arden-Arcade, CA
    "ATNHPIUS26620Q": "Huntsville",  # Huntsville, AL
    "ATNHPIUS14260Q": "Boise",  # Boise City, ID
    "ATNHPIUS40140Q": "Riverside",  # Riverside-San Bernardino-Ontario, CA
    "ATNHPIUS37980Q": "Philadelphia",  # Philadelphia-Camden-Wilmington, PA-NJ
    "ATNHPIUS26420Q": "Houston",  # Houston-The Woodlands-Sugar Land, TX
    "ATNHPIUS17140Q": "Cincinnati",  # Cincinnati, OH-KY-IN
    "ATNHPIUS19820Q": "Detroit-FHFA",  # Detroit-Warren-Dearborn, MI (for comparison)
    "ATNHPIUS41740Q": "San Antonio",  # San Antonio-New Braunfels, TX
    "ATNHPIUS18140Q": "Columbus",  # Columbus, OH
    "ATNHPIUS17460Q": "Cleveland-FHFA",  # Cleveland-Elyria, OH
    "ATNHPIUS36740Q": "Orlando",  # Orlando-Kissimmee-Sanford, FL
    "ATNHPIUS19740Q": "Denver-FHFA",  # Denver-Aurora-Lakewood, CO
    "ATNHPIUS41620Q": "Salt Lake City",  # Salt Lake City, UT
    "ATNHPIUS32820Q": "Memphis",  # Memphis, TN-MS-AR
    "ATNHPIUS38900Q": "Portland-FHFA",  # Portland-Vancouver-Hillsboro, OR-WA
}

# Combine all markets
MARKETS = {**CS_MARKETS, **FHFA_MARKETS}

# CPI Rent Index by Metro (BLS - Rent of Primary Residence)
# Format: CUURA### or CUUSA### where ### is the metro code
# Note: Some series are monthly (CUURA), some are annual/semi-annual (CUUSA)
RENT_SERIES = {
    "Atlanta": "CUURA319SEHA",  # Atlanta - Monthly
    "Boston": "CUURA103SEHA",  # Boston - Monthly
    "Chicago": "CUURA207SEHA",  # Chicago - Monthly
    "San Diego": "CUURA424SEHA",  # San Diego - Monthly
    "Dallas": "CUURA316SEHA",  # Dallas - Monthly
    "Denver": "CUUSA433SEHA",  # Denver - Annual
    "Detroit": "CUURA208SEHA",  # Detroit - Monthly
    "Los Angeles": "CUURA421SEHA",  # Los Angeles - Monthly
    "Miami": "CUURA320SEHA",  # Miami - Monthly
    "Minneapolis": "CUUSA211SEHA",  # Minneapolis - Annual
    "New York": "CUURA101SEHA",  # New York - Monthly
    "Phoenix": "CUUSA429SEHA",  # Phoenix - Annual
    "Portland": "CUUSA438SEHA",  # Portland - Annual
    "Las Vegas": "CUUSA437SEHA",  # Las Vegas - Annual
    "San Francisco": "CUURA422SEHA",  # San Francisco - Monthly
    "Seattle": "CUURA423SEHA",  # Seattle - Monthly
    "Tampa": "CUURA321SEHA",  # Tampa - Monthly
    "Washington DC": "CUURA311SEHA",  # Washington DC - Monthly
    "Cleveland": "CUURA210SEHA",  # Cleveland - Monthly
    # FHFA markets - many don't have separate CPI rent series
    "Raleigh": None,
    "Nashville": "CUUSA427SEHA",  # Nashville - Annual
    "Austin": "CUUSA426SEHA",  # Austin - Annual
    "Sacramento": None,
    "Huntsville": None,
    "Boise": None,
    "Riverside": None,
    "Philadelphia": "CUURA102SEHA",  # Philadelphia - Monthly
    "Houston": "CUURA318SEHA",  # Houston - Monthly
    "Cincinnati": None,
    "Detroit-FHFA": "CUURA208SEHA",  # Same as Detroit
    "San Antonio": None,
    "Columbus": None,
    "Cleveland-FHFA": "CUURA210SEHA",  # Same as Cleveland
    "Orlando": None,
    "Denver-FHFA": "CUUSA433SEHA",  # Same as Denver
    "Salt Lake City": None,
    "Memphis": None,
    "Portland-FHFA": "CUUSA438SEHA",  # Same as Portland
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

    # Rent-based features (if available)
    if "rent_index" in df.columns:
        df["rent_yoy"] = df["rent_index"].pct_change(12) * 100  # Monthly CPI data
        df["rent_mom3"] = df["rent_index"].pct_change(3) * 100
        df["price_to_rent"] = df[price_col] / df["rent_index"] * 100
        df["price_to_rent_ma12"] = df["price_to_rent"].rolling(12).mean()
        df["price_to_rent_deviation"] = (df["price_to_rent"] / df["price_to_rent_ma12"] - 1) * 100

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
    print(f"{'Market':<25} {'Source':<15} {'HPA':<10} {'Rent':<10} {'YoY%'}")
    print("=" * 75)

    for series_id, market_name in MARKETS.items():
        try:
            # Fetch house price data
            market_series = fred.get_series(series_id, observation_start="2000-01-01")

            if len(market_series) < 120:
                print(f"⚠️  {market_name}: Insufficient price data")
                continue

            market_df = market_series.to_frame("price_index")
            market_df.index.name = "date"

            # Handle quarterly FHFA data - resample to monthly
            if series_id.startswith("ATNHPIUS"):
                market_df = market_df.resample("MS").ffill()
            else:
                market_df = market_df.resample("MS").ffill().bfill()

            # Fetch rent data if available
            rent_data = None
            rent_yoy_latest = np.nan
            price_to_rent_latest = np.nan

            if market_name in RENT_SERIES and RENT_SERIES[market_name] is not None:
                try:
                    rent_series = fred.get_series(RENT_SERIES[market_name], observation_start="2000-01-01")
                    if len(rent_series) > 20:
                        rent_df = rent_series.to_frame("rent_index")
                        rent_df.index.name = "date"
                        # Handle both monthly and annual/semi-annual data
                        # Forward fill to monthly if needed
                        rent_df = rent_df.resample("MS").ffill()
                        market_df = market_df.join(rent_df)

                        # Calculate latest rent metrics
                        if len(rent_series) >= 2:
                            # For annual data, calculate YoY differently
                            if len(rent_series) >= 13:
                                rent_yoy_latest = (rent_series.iloc[-1] / rent_series.iloc[-13] - 1) * 100
                            else:
                                rent_yoy_latest = (rent_series.iloc[-1] / rent_series.iloc[-2] - 1) * 100
                        price_to_rent_latest = market_series.iloc[-1] / rent_series.iloc[-1] * 100

                except Exception as e:
                    pass  # Silently skip rent data if not available

            # Merge with macro
            df_all = macro_df.join(market_df).dropna(subset=["price_index"])

            # Create features
            df_all = create_features(df_all)

            # Target: 5-year forward return
            df_all["target_5y"] = (
                    (df_all["price_index"].shift(-60) - df_all["price_index"])
                    / df_all["price_index"] * 100
            )

            # Also create rent target if available
            has_rent = "rent_index" in df_all.columns
            if has_rent:
                df_all["rent_target_5y"] = (
                        (df_all["rent_index"].shift(-60) - df_all["rent_index"])
                        / df_all["rent_index"] * 100
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
                            if col not in ["date", "target_5y", "rent_target_5y", "price_index",
                                           "price_ma12", "price_to_rent_ma12", "rent_index"]]

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

            # Train ensemble for house prices
            models = train_ensemble_model(X, y)

            # Train rent model if data available
            rent_pred = np.nan
            rent_uncertainty = np.nan
            rent_hist_mean = np.nan

            if has_rent and "rent_target_5y" in df_all.columns:
                y_rent = df_all["rent_target_5y"].dropna()
                X_rent = X.loc[y_rent.index]

                if len(y_rent) >= 30:
                    rent_models = train_ensemble_model(X_rent, y_rent)
                    rent_hist_mean = y_rent.mean()

                    # Predict rent
                    latest_row = df_all.iloc[[-1]][feature_cols].copy()
                    if not latest_row.isnull().any().any():
                        rent_pred, _ = get_ensemble_prediction(rent_models, latest_row)

                        # Rent uncertainty
                        rent_preds = [tree.predict(X_rent.to_numpy()) for tree in rent_models['rf'].estimators_[:100]]
                        rent_residuals = [y_rent.values - pred for pred in rent_preds]
                        rent_uncertainty = np.mean([np.std(res) for res in rent_residuals])

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

            ci_lower = pred - 1.28 * residual_std
            ci_upper = pred + 1.28 * residual_std

            # Rent CIs
            rent_ci_lower = rent_pred - 1.28 * rent_uncertainty if not np.isnan(rent_uncertainty) else np.nan
            rent_ci_upper = rent_pred + 1.28 * rent_uncertainty if not np.isnan(rent_uncertainty) else np.nan

            # Get current market stats
            current_price = market_series.iloc[-1]
            yoy_change = (market_series.iloc[-1] / market_series.iloc[-13] - 1) * 100

            predictions.append({
                "Market": market_name,
                "5Y_HPA_Forecast": pred,
                "HPA_Uncertainty": residual_std,
                "HPA_CI_Lower": ci_lower,
                "HPA_CI_Upper": ci_upper,
                "5Y_Rent_Forecast": rent_pred,
                "Rent_Uncertainty": rent_uncertainty,
                "Rent_CI_Lower": rent_ci_lower,
                "Rent_CI_Upper": rent_ci_upper,
                "Current_Index": current_price,
                "YoY_Price_Change": yoy_change,
                "YoY_Rent_Change": rent_yoy_latest,
                "Price_to_Rent": price_to_rent_latest,
                "Training_N": len(df_all),
                "Hist_Mean_HPA": y_mean,
                "Hist_Mean_Rent": rent_hist_mean,
            })

            rent_str = f"{rent_pred:6.2f}%" if not np.isnan(rent_pred) else "N/A"
            source = "Case-Shiller" if not series_id.startswith("ATNHPIUS") else "FHFA"
            print(f"✅ {market_name:<25} {source:<15} {pred:6.2f}% {rent_str:<10} {yoy_change:5.2f}%")

        except Exception as e:
            print(f"❌ {market_name:<25} {'Error':<15} {str(e)[:40]}")

    # Create rankings
    if not predictions:
        print("\n❌ No predictions generated")
        return

    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values("5Y_HPA_Forecast", ascending=False).reset_index(drop=True)
    results_df["Rank_HPA"] = range(1, len(results_df) + 1)

    # Rank by rent appreciation
    results_df_rent = results_df.dropna(subset=["5Y_Rent_Forecast"]).copy()
    results_df_rent = results_df_rent.sort_values("5Y_Rent_Forecast", ascending=False).reset_index(drop=True)
    results_df_rent["Rank_Rent"] = range(1, len(results_df_rent) + 1)

    # Calculate total return (cap appreciation + rent yield approximation)
    # For markets with rent data, estimate 5-year rental income
    # For markets without rent data, use the mean of markets with data
    mean_rent_forecast = results_df['5Y_Rent_Forecast'].mean()
    results_df["Rent_Forecast_Filled"] = results_df["5Y_Rent_Forecast"].fillna(mean_rent_forecast)
    results_df["Total_Return_5Y"] = results_df["5Y_HPA_Forecast"] + results_df["Rent_Forecast_Filled"]

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/market_rankings_with_rent.csv", index=False)

    # Display HPA rankings
    print("\n" + "=" * 140)
    print("📊 5-YEAR HOME PRICE APPRECIATION FORECAST")
    print("=" * 140)
    display_cols = ["Rank_HPA", "Market", "5Y_HPA_Forecast", "HPA_Uncertainty",
                    "5Y_Rent_Forecast", "Rent_Uncertainty", "YoY_Price_Change", "YoY_Rent_Change"]
    print(results_df[display_cols].to_string(index=False,
                                             float_format=lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"))
    print("=" * 140)

    # Display rent rankings
    if len(results_df_rent) > 0:
        print("\n" + "=" * 100)
        print("📊 5-YEAR RENT APPRECIATION FORECAST - TOP MARKETS")
        print("=" * 100)
        rent_display = results_df_rent[["Rank_Rent", "Market", "5Y_Rent_Forecast", "Rent_Uncertainty",
                                        "YoY_Rent_Change", "Price_to_Rent"]].head(10)
        print(rent_display.to_string(index=False, float_format=lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A"))
        print("=" * 100)

    # Summary
    print(f"\n📈 Summary Statistics:")
    print(f"   HPA Forecast - Mean:   {results_df['5Y_HPA_Forecast'].mean():6.2f}%")
    print(f"   HPA Forecast - Median: {results_df['5Y_HPA_Forecast'].median():6.2f}%")
    print(f"   Rent Forecast - Mean:  {results_df['5Y_Rent_Forecast'].mean():6.2f}%")
    print(f"   Rent Forecast - Median:{results_df['5Y_Rent_Forecast'].median():6.2f}%")

    # Top markets by total return
    results_df_total = results_df.sort_values("Total_Return_5Y", ascending=False).reset_index(drop=True)
    print(f"\n🏆 Top 10 Markets by Total Return (Price + Rent):")
    for idx, row in results_df_total.head(10).iterrows():
        hpa = row['5Y_HPA_Forecast']
        rent = row['5Y_Rent_Forecast'] if not np.isnan(row['5Y_Rent_Forecast']) else row['Rent_Forecast_Filled']
        rent_note = "" if not np.isnan(row['5Y_Rent_Forecast']) else "*"
        total = row['Total_Return_5Y']
        print(
            f"   {idx + 1:2d}. {row['Market']:20s} Total: {total:6.2f}% (HPA: {hpa:6.2f}% + Rent: {rent:5.2f}%{rent_note})")

    # Risk-adjusted
    results_df["Sharpe_Ratio"] = results_df["5Y_HPA_Forecast"] / results_df["HPA_Uncertainty"]
    results_df_sharpe = results_df.sort_values("Sharpe_Ratio", ascending=False).reset_index(drop=True)

    print(f"\n📊 Best Risk-Adjusted Returns (Top 5 by Sharpe Ratio):")
    for idx, row in results_df_sharpe.head(5).iterrows():
        print(f"   {idx + 1}. {row['Market']:20s} Sharpe: {row['Sharpe_Ratio']:5.2f} "
              f"(HPA: {row['5Y_HPA_Forecast']:6.2f}%, Risk: {row['HPA_Uncertainty']:5.2f}%)")

    # Investment insights
    print(f"\n💡 Investment Insights:")
    print(f"   * Markets with asterisk (*) use average rent forecast of {mean_rent_forecast:.2f}%")

    # Find markets with low price-to-rent (potentially undervalued)
    results_with_ptr = results_df.dropna(subset=["Price_to_Rent"]).copy()
    if len(results_with_ptr) > 0:
        results_with_ptr = results_with_ptr.sort_values("Price_to_Rent")
        print(f"\n🔍 Potentially Undervalued Markets (Low Price-to-Rent Ratio):")
        for idx, row in results_with_ptr.head(5).iterrows():
            print(f"   - {row['Market']:20s} P/R: {row['Price_to_Rent']:.1f} | "
                  f"HPA: {row['5Y_HPA_Forecast']:5.2f}% | Rent: {row['5Y_Rent_Forecast']:5.2f}%")

    # Find markets with high rent growth relative to price growth
    results_rent_ratio = results_df.dropna(subset=["5Y_Rent_Forecast"]).copy()
    if len(results_rent_ratio) > 0:
        results_rent_ratio["Rent_to_HPA_Ratio"] = results_rent_ratio["5Y_Rent_Forecast"] / results_rent_ratio[
            "5Y_HPA_Forecast"]
        results_rent_ratio = results_rent_ratio.sort_values("Rent_to_HPA_Ratio", ascending=False)
        print(f"\n🏡 Best Markets for Rental Income Investors (High Rent Growth vs Price):")
        for idx, row in results_rent_ratio.head(5).iterrows():
            ratio = row['Rent_to_HPA_Ratio'] * 100
            print(f"   - {row['Market']:20s} Rent/HPA: {ratio:.1f}% | "
                  f"HPA: {row['5Y_HPA_Forecast']:5.2f}% | Rent: {row['5Y_Rent_Forecast']:5.2f}%")

    print(f"\n✅ Results saved to: results/market_rankings_with_rent.csv")


if __name__ == "__main__":
    main()