import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------
# Configuration
# -----------------------
# Using your ML_MACRO_TICKERS logic plus the Breadth proxies
MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']

def train_risk_regime_model():
    print("🚀 Training Macro Risk Model (RSP/SPY Breadth Logic)...")

    # 1. Download Data
    # Downloading back to 2010 to capture multiple Volatility regimes
    data = yf.download(MACRO_TICKERS, start="2010-01-01", progress=False)['Close'].ffill()
    
    # 2. Feature Engineering (Mirroring indicators.py)
    X = pd.DataFrame(index=data.index)
    
    # Breadth Logic (RSP/SPY Ratio)
    X['Breadth_Ratio'] = data['RSP'] / data['SPY']
    X['Breadth_MA_Diff'] = X['Breadth_Ratio'] - X['Breadth_Ratio'].rolling(50).mean()
    
    # Risk & Volatility
    X['VIX_Level'] = data['^VIX']
    X['MOVE_Level'] = data['^MOVE']
    X['Credit_Spread'] = data['LQD'] / data['HYG']
    
    # Monetary & Trend
    X['Yield_Curve'] = data['^TYX'] - data['^TNX']
    X['DXY_Mom'] = data['DX-Y.NYB'].pct_change(21)
    X['SPY_Trend'] = data['SPY'] / data['SPY'].rolling(200).mean()

    # 3. Sahm Rule "Danger Zone" Proxy
    # Since we avoid Wikipedia/External Scraping, we use a price-action proxy:
    # Sustained Breadth decay + Yield Curve inversion often lead labor market shifts.
    X['Labor_Stress_Proxy'] = ((X['Yield_Curve'] < 0) & (X['Breadth_MA_Diff'] < 0)).astype(int)

    # 4. Target Variable: Forward 1-Month Returns
    # 1 = Risk-On (Positive return), 0 = Risk-Off (Negative/Flat)
    horizon = 21 
    future_return = data['SPY'].pct_change(horizon).shift(-horizon)
    y = (future_return > 0).astype(int)

    # 5. Clean and Split
    # Drop NaNs from rolling windows and the shifted target
    valid_data = pd.concat([X, y.rename('target')], axis=1).dropna()
    
    X_clean = valid_data.drop(columns=['target'])
    y_clean = valid_data['target']

    # Time-series split (Last 15% for testing)
    split = int(len(X_clean) * 0.85)
    X_train, X_test = X_clean.iloc[:split], X_clean.iloc[split:]
    y_train, y_test = y_clean.iloc[:split], y_clean.iloc[split:]

    # 6. Scaling & Training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Balanced class weights handle the "Risk-Off" rarity in bull markets
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        class_weight='balanced_subsample',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 7. Feature Importance Report
    importance = pd.DataFrame({
        'Feature': X_clean.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n📊 Model Driver Analysis:")
    print(importance)

    # 8. Save Model Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': X_clean.columns.tolist(),
        'target_horizon': horizon
    }
    joblib.dump(bundle, 'risk_model_v1.joblib')
    print("\n✅ Risk training complete. 'risk_model_v1.joblib' is ready.")

if __name__ == "__main__":
    train_risk_regime_model()
