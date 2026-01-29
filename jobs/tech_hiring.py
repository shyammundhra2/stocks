import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Suppress future warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================
# FREE DATA SOURCES
# =============================================

def load_fred_data(start="2005-01-01"):
    """Load comprehensive labor market data from FRED (FREE)"""
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")

    fred = Fred(api_key=api_key)

    # Updated series IDs - validated to work
    series = {
        'unrate': 'UNRATE',  # Overall unemployment rate
        'initial_claims': 'ICSA',  # Weekly jobless claims
        'cont_claims': 'CCSA',  # Continued claims
        'fed_rate': 'DFF',  # Federal funds rate
        'nfp': 'PAYEMS',  # Nonfarm payrolls (total employment)
        'jolts_total': 'JTSJOL',  # Total job openings (JOLTS)
        'avg_hours': 'AWHAETP',  # Average weekly hours
        'labor_force': 'CLF16OV',  # Civilian labor force
    }

    data = {}
    failed = []

    for name, series_id in series.items():
        try:
            series_data = fred.get_series(series_id, observation_start=start)
            if series_data is not None and len(series_data) > 0:
                data[name] = series_data
            else:
                failed.append(name)
        except Exception as e:
            print(f"⚠️  Failed to load {name} ({series_id}): {e}")
            failed.append(name)

    if not data:
        raise ValueError("No FRED data loaded successfully")

    print(f"✓ Loaded {len(data)}/{len(series)} FRED series")
    if failed:
        print(f"  Skipped: {', '.join(failed)}")

    df = pd.DataFrame(data)
    df = df.resample('ME').mean()

    # Create change features only for columns that exist
    if 'unrate' in df.columns:
        df['unrate_chg'] = df['unrate'].pct_change(fill_method=None)
    if 'jolts_total' in df.columns:
        df['jolts_chg'] = df['jolts_total'].pct_change(fill_method=None)
    if 'initial_claims' in df.columns:
        df['claims_chg'] = df['initial_claims'].pct_change(fill_method=None)
    if 'nfp' in df.columns:
        df['nfp_chg'] = df['nfp'].pct_change(fill_method=None)

    return df.dropna()


def load_yf_market_data(start="2005-01-01"):
    """Load market data from Yahoo Finance (FREE)"""
    tickers = ['SPY', '^VIX', 'QQQ', 'HYG']
    ticker_names = ['spy', 'vix', 'qqq', 'hyg']

    data = {}
    failed = []

    for name, ticker in zip(ticker_names, tickers):
        try:
            # Use Ticker object for more reliable access
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, auto_adjust=False)

            if not hist.empty:
                # Use Close price
                if 'Close' in hist.columns:
                    # Remove timezone information to match FRED data
                    series = hist['Close']
                    series.index = series.index.tz_localize(None)
                    data[name] = series
                else:
                    print(f"⚠️  No Close column for {name}")
                    failed.append(name)
            else:
                print(f"⚠️  No data for {name}")
                failed.append(name)
        except Exception as e:
            print(f"⚠️  Failed to load {name}: {e}")
            failed.append(name)

    if not data:
        raise ValueError("No market data loaded successfully")

    print(f"✓ Loaded {len(data)}/{len(tickers)} market series")
    if failed:
        print(f"  Skipped: {', '.join(failed)}")

    # Create DataFrame from dict of Series
    df = pd.DataFrame(data)
    df = df.resample('ME').last()

    # Create return features only for columns that exist
    if 'spy' in df.columns:
        df['spy_ret'] = df['spy'].pct_change(fill_method=None)
    if 'qqq' in df.columns:
        df['qqq_ret'] = df['qqq'].pct_change(fill_method=None)
        if 'spy' in df.columns:
            df['qqq_spy_ratio'] = df['qqq'] / df['spy']  # Tech relative strength
    if 'vix' in df.columns:
        df['vix_chg'] = df['vix'].pct_change(fill_method=None)
        df['vix_level'] = df['vix']
    if 'hyg' in df.columns:
        df['credit_spread'] = df['hyg'].pct_change(fill_method=None)

    return df.dropna()


# =============================================
# BUILD DATASET WITH FREE DATA
# =============================================

def build_free_dataset():
    """Combine all free data sources"""

    print("\n📊 Loading data sources...")

    # Load macro/labor data
    fred_data = load_fred_data()
    market_data = load_yf_market_data()

    # Combine
    X = fred_data.join(market_data, how='inner')

    print(f"\n✓ Combined dataset: {len(X)} months")
    print(f"  Date range: {X.index.min().strftime('%Y-%m')} to {X.index.max().strftime('%Y-%m')}")
    print(f"  Features: {X.shape[1]}")

    # Build target: forward-looking unemployment improvement
    # Using general unemployment as proxy for tech hiring
    if 'unrate_chg' in X.columns:
        y = -X['unrate_chg'].shift(-3)  # Negative = improvement (lower unemp)
    elif 'unrate' in X.columns:
        y = -X['unrate'].pct_change(fill_method=None).shift(-3)
    else:
        raise ValueError("No unemployment data available for target")

    # Convert to probability
    y = 1 / (1 + np.exp(-10 * y))
    y = y.rename('hiring_prob')

    # Clean
    data = X.join(y, how='inner').dropna()

    if len(data) < 50:
        raise ValueError(f"Insufficient data: only {len(data)} months available")

    return data.drop(columns='hiring_prob'), data['hiring_prob']


# =============================================
# TRAIN MODEL
# =============================================

def train_model(X, y):
    """Train with walk-forward validation"""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    # Walk-forward validation
    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []

    print("\n🔄 Training with walk-forward validation...")

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        p = model.predict(X_test)

        preds.extend(p)
        actuals.extend(y_test)

        fold_corr = np.corrcoef(y_test, p)[0, 1]
        print(f"  Fold {i}: corr={fold_corr:.3f}, test_size={len(y_test)}")

    corr = np.corrcoef(actuals, preds)[0, 1]
    print(f"\n✓ Overall walk-forward correlation: {corr:.3f}")

    # Fit final model on all data
    model.fit(X, y)

    # Feature importance
    feature_imp = pd.Series(
        model.named_steps['gbm'].feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print(f"\n📊 Top 5 Features:")
    for feat, imp in feature_imp.head(5).items():
        print(f"  {feat}: {imp:.3f}")

    return model


def get_live_signal(model, X):
    """Generate current hiring signal"""
    latest_prob = model.predict(X.tail(1))[0]

    print(f"\n📊 Current tech hiring probability: {latest_prob:.1%}")

    if latest_prob > 0.6:
        return "🟢 Positive hiring environment"
    elif latest_prob < 0.4:
        return "🔴 Challenging hiring environment"
    else:
        return "🟡 Neutral hiring conditions"


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('FRED_API_KEY'):
        print("❌ Error: FRED_API_KEY environment variable not set")
        print("\nGet your free API key from: https://fredaccount.stlouisfed.org/apikeys")
        print("\nThen set it using:")
        print("  export FRED_API_KEY='your_key_here'  # Linux/Mac")
        print("  set FRED_API_KEY=your_key_here       # Windows CMD")
        print("  $env:FRED_API_KEY='your_key_here'    # Windows PowerShell")
        exit(1)

    try:
        X, y = build_free_dataset()

        # Show feature preview
        print(f"\nFeatures ({len(X.columns)}): {', '.join(X.columns[:10])}{'...' if len(X.columns) > 10 else ''}")

        # Train model
        model = train_model(X, y)

        # Generate signal
        signal = get_live_signal(model, X)
        print(f"\n{signal}")

        # Optional: save model and data
        import joblib

        joblib.dump(model, 'hiring_forecast_model.pkl')
        X.to_csv('latest_features.csv')

        # Save predictions for visualization
        y_pred = model.predict(X)
        pred_df = pd.DataFrame({
            'date': X.index,
            'actual': y,
            'predicted': y_pred
        })
        pred_df.to_csv('predictions.csv', index=False)

        print("\n✓ Model saved to hiring_forecast_model.pkl")
        print("✓ Features saved to latest_features.csv")
        print("✓ Predictions saved to predictions.csv")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()