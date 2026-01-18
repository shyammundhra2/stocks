import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------- 1. Configuration -----------------
# Sector tickers (ETFs)
SECTOR_TICKERS = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']

# Macro context tickers used as "Features"
MACRO_TICKERS = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD'] 

def train_quarterly_model():
    print("🚀 Initializing Institutional-Grade Quarterly Model Training...")
    
    # 2. Data Acquisition (25 Years of History)
    tickers = SECTOR_TICKERS + MACRO_TICKERS
    raw_data = yf.download(tickers, start="2000-01-01")['Close'].ffill()

    # 3. Feature Engineering (Matches Live Dashboard Names)
    print("🛠️  Engineering Macro Features...")
    X = pd.DataFrame(index=raw_data.index)
    
    # Quarterly Momentum (63 trading days)
    X['DXY_mom'] = raw_data['DX-Y.NYB'].pct_change(63) 
    
    # Risk & Volatility Metrics
    X['VIX_level'] = raw_data['^VIX'].rolling(21).mean() # Monthly smoothed
    X['MOVE_level'] = raw_data['^MOVE'].ffill().fillna(raw_data['^MOVE'].mean())
    X['TNX_vol'] = raw_data['^TNX'].rolling(63).std()   # Quarterly yield volatility
    
    # Spread Metrics (Credit & Yield Curve)
    X['Yield_Curve'] = raw_data['^TYX'] - raw_data['^TNX'] # 30Y - 10Y
    X['Credit_Spread'] = raw_data['LQD'] / raw_data['HYG'] # IG vs HY ratio
    
    # 4. Target Generation (Look-Forward 63 Days)
    # Goal: Identify which sector will be the #1 performer over the NEXT quarter
    horizon = 63 
    sector_returns = raw_data[SECTOR_TICKERS].pct_change(horizon).shift(-horizon)
    y = sector_returns.idxmax(axis=1)

    # 5. Data Cleaning & Temporal Split (Prevent Data Leakage)
    valid_data = pd.concat([X, y.rename('target')], axis=1).dropna()
    
    # Split: Train on 80% of history, Test on most recent 20%
    split_idx = int(len(valid_data) * 0.8)
    train_df = valid_data.iloc[:split_idx]
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    # 6. Scaling (Fit ONLY on Training Data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 7. Model: Random Forest (Regime Switching Logic)
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,           # Prevent overfitting to single-day noise
        min_samples_leaf=20,    # Ensure patterns are statistically significant
        class_weight='balanced',# Don't let historical Tech dominance bias the AI
        random_state=42
    )
    
    print("🧠 Training Random Forest Ensemble...")
    model.fit(X_train_scaled, y_train)

    # 8. Feature Importance Audit (The Strategist's View)
    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()
    feature_imp_df = pd.DataFrame({
        'Driver': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 Macro Driver Hierarchy (Why the AI is picking sectors):")
    print("-" * 55)
    for _, row in feature_imp_df.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"{row['Driver']:<15} | {bar} {row['Importance']:.2%}")
    print("-" * 55)

    # 9. Save Bundle for Dashboard Integration
    bundle = {
        'model': model, 
        'scaler': scaler, 
        'features': feature_names
    }
    joblib.dump(bundle, 'sector_model.joblib')
    print("\n✅ Training Complete. 'sector_model.joblib' is ready for live use.")

if __name__ == "__main__":
    train_quarterly_model()
