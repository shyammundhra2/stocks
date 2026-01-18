import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Configuration
sector_tickers = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']
macro_tickers = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD'] 

def train_quarterly_model_with_live_names():
    print("🚀 Training Quarterly Model using Live Dashboard feature names...")
    # Fetch 25 years of history
    data = yf.download(sector_tickers + macro_tickers, start="2000-01-01")['Close'].ffill()

    # 2. Feature Engineering - NAMES MATCH YOUR CURRENT INDICATORS.PY
    X = pd.DataFrame(index=data.index)
    
    # We calculate these over a 63-day period to match a Quarterly horizon
    X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63) 
    X['VIX_level'] = data['^VIX'].rolling(21).mean() # Monthly smoothed average
    X['MOVE_level'] = data['^MOVE'].ffill().fillna(data['^MOVE'].mean())
    X['Yield_Curve'] = data['^TYX'] - data['^TNX']
    X['Credit_Spread'] = data['LQD'] / data['HYG']
    X['TNX_vol'] = data['^TNX'].rolling(63).std() # Quarterly yield volatility

    # 3. Target: Best performer over the NEXT 63 trading days (1 Quarter)
    horizon = 63 
    sector_returns = data[sector_tickers].pct_change(horizon).shift(-horizon)
    y = sector_returns.idxmax(axis=1)

    # 4. Clean and Scale
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_train = valid.drop(columns=['target'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 5. Model: Built for structural patterns
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7, 
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_scaled, valid['target'])

    # 6. Save with the names your app expects
    joblib.dump({
        'model': model, 
        'scaler': scaler, 
        'features': X_train.columns.tolist()
    }, 'sector_model.joblib')
    
    print("✅ Model trained on Quarterly horizon using original feature names.")
    print(f"Features mapped: {X_train.columns.tolist()}")

if __name__ == "__main__":
    train_quarterly_model_with_live_names()
