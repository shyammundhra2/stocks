import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------
# Configuration & Tickers
# -----------------------
# Core Macro + Sector ETFs for Breadth/Rotation Analysis
MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
ALL_TICKERS = MACRO_TICKERS + SECTOR_TICKERS

def train_risk_regime_model():
    print("🚀 Training Macro Risk Model v2 (Sector-Aware)...")

    # 1. Download Data
    # 2010 start allows for multiple credit and hiking cycles
    data = yf.download(ALL_TICKERS, start="2000-01-01", progress=False)['Close'].ffill()
    
    # 2. Feature Engineering
    X = pd.DataFrame(index=data.index)
    
    # --- A. Market Breadth & Structure ---
    X['Breadth_Ratio'] = data['RSP'] / data['SPY']
    X['Breadth_MA_Diff'] = X['Breadth_Ratio'] - X['Breadth_Ratio'].rolling(50).mean()
    X['SPY_Trend'] = data['SPY'] / data['SPY'].rolling(200).mean()

    # --- B. Sector Strength (The "Rotational" Edge) ---
    # Defensive vs Cyclical Ratio: (Utilities + Staples) / (Tech + Discretionary)
    # A spike here indicates "Smart Money" moving to safety
    X['Defensive_Rotation'] = (data['XLU'] + data['XLP']) / (data['XLK'] + data['XLY'])
    
    # Financials Relative Strength (Lead indicator for credit/liquidity)
    X['XLF_Relative_Strength'] = data['XLF'] / data['SPY']
    
    # Sector Momentum (3-Month lookback as per instructions)
    for s in SECTOR_TICKERS:
        X[f'{s}_3M_Mom'] = data[s].pct_change(63)

    # --- C. Risk & Volatility ---
    X['VIX_Level'] = data['^VIX']
    X['MOVE_Level'] = data['^MOVE']
    X['Credit_Spread_Proxy'] = data['LQD'] / data['HYG'] # Higher = Stress (Flight to Quality)
    
    # --- D. Monetary & FX ---
    X['Yield_Curve'] = data['^TYX'] - data['^TNX'] # 30Y - 10Y (Steepening/Flattening)
    X['DXY_Mom'] = data['DX-Y.NYB'].pct_change(21)

    # --- E. Sahm Rule "Danger Zone" Proxy ---
    # We flag stress when Yield Curve is flat AND Defensive sectors are outperforming
    X['Labor_Stress_Proxy'] = ((X['Yield_Curve'] < 0.1) & (X['Defensive_Rotation'] > X['Defensive_Rotation'].rolling(126).mean())).astype(int)

    # 3. Target Variable: Forward 1-Month Returns
    # 1 = Risk-On (SPY > 0), 0 = Risk-Off (SPY <= 0)
    horizon = 21 
    future_return = data['SPY'].pct_change(horizon).shift(-horizon)
    y = (future_return > 0).astype(int)

    # 4. Data Cleaning
    valid_data = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_clean = valid_data.drop(columns=['target'])
    y_clean = valid_data['target']

    # 5. Time-Series Split (Last 15% for Validation)
    split = int(len(X_clean) * 0.85)
    X_train, X_test = X_clean.iloc[:split], X_clean.iloc[split:]
    y_train, y_test = y_clean.iloc[:split], y_clean.iloc[split:]

    # 6. Scaling & Training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        class_weight='balanced_subsample', # Critical for spotting rare Risk-Off regimes
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 7. Model Driver Analysis
    importance = pd.DataFrame({
        'Feature': X_clean.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n📊 Macro Model Driver Analysis (Top 10):")
    print(importance.head(10))

    # 8. Save Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': X_clean.columns.tolist(),
        'target_horizon': horizon
    }
    joblib.dump(bundle, 'risk_model.joblib')
    print("\n✅ Training complete. 'macro_risk_model_v2.joblib' includes Sector Intelligence.")

if __name__ == "__main__":
    train_risk_regime_model()
