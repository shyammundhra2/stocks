import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Assuming these are imported correctly from your local structure
# from macro.constants import COMMODITIES, ML_MACRO_TICKERS

# Mocking constants for the sake of the complete code example
COMMODITIES = {'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil', 'HG=F': 'Copper'}
ML_MACRO_TICKERS = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD']

commodity_tickers = list(COMMODITIES.keys())
macro_tickers = ML_MACRO_TICKERS

def train_quarterly_commodity_model():
    print("🚀 Training Quarterly Commodity Model with constants...")

    # Fetch 25 years of historical data
    data = yf.download(commodity_tickers + macro_tickers, start="2000-01-01")['Close'].ffill()

    # 1. Feature Engineering
    X = pd.DataFrame(index=data.index)

    # Macro features
    X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63)
    X['VIX_level'] = data['^VIX'].rolling(21).mean()
    X['Yield_Curve'] = data['^TYX'] - data['^TNX']
    X['Credit_Spread'] = data['LQD'] / data['HYG']

    # Commodity-specific features
    for ticker in commodity_tickers:
        X[f'{ticker}_mom'] = data[ticker].pct_change(63)           # quarterly momentum
        X[f'{ticker}_vol'] = data[ticker].rolling(63).std()        # quarterly volatility
        X[f'{ticker}_mom_1m'] = data[ticker].pct_change(21)       # 1-month momentum
        X[f'{ticker}_vol_1m'] = data[ticker].rolling(21).std()    # 1-month volatility

    # 2. Target: best-performing commodity over next quarter
    horizon = 63
    returns = data[commodity_tickers].pct_change(horizon).shift(-horizon)
    y = returns.idxmax(axis=1)

    # 3. Clean and scale
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_train = valid.drop(columns=['target'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 4. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, valid['target'])

    # --- NEW: FEATURE RANKING SECTION ---
    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()
    
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 Commodity Driver Hierarchy (Top 15 Drivers):")
    print("-" * 60)
    # Displaying top 15 to keep the console clean
    for _, row in feature_imp_df.head(15).iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"{row['Feature']:<20} | {bar} {row['Importance']:.2%}")
    print("-" * 60)
    # ------------------------------------

    # 5. Save model
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': X_train.columns.tolist()
    }, 'commodity_model.joblib')

    print("✅ Commodity model trained and saved!")

if __name__ == "__main__":
    train_quarterly_commodity_model()
