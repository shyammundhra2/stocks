import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro.paths import model_path

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# -----------------------
# Configuration & Tickers
# -----------------------
MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
ALL_TICKERS = MACRO_TICKERS + SECTOR_TICKERS

def train_risk_regime_model():
    print("🚀 Training Macro Risk Model v2 (Sector-Aware)...")

    # 1. Download Data — ffill deferred to after split (prevents leakage)
    data = yf.download(ALL_TICKERS, start="2000-01-01", progress=False, auto_adjust=True)['Close']

    # 2. Feature Engineering
    X = pd.DataFrame(index=data.index)

    # --- A. Market Breadth & Structure ---
    X['Breadth_Ratio'] = data['RSP'] / data['SPY']
    X['Breadth_MA_Diff'] = X['Breadth_Ratio'] - X['Breadth_Ratio'].rolling(50).mean()
    X['SPY_Trend'] = data['SPY'] / data['SPY'].rolling(200).mean()

    # --- B. Sector Strength (The "Rotational" Edge) ---
    X['Defensive_Rotation'] = (data['XLU'] + data['XLP']) / (data['XLK'] + data['XLY'])
    X['XLF_Relative_Strength'] = data['XLF'] / data['SPY']

    for s in SECTOR_TICKERS:
        X[f'{s}_3M_Mom'] = data[s].pct_change(63, fill_method=None)

    # --- C. Risk & Volatility ---
    X['VIX_Level'] = data['^VIX']
    X['MOVE_Level'] = data['^MOVE']
    X['Credit_Spread_Proxy'] = data['LQD'] / data['HYG']

    # --- D. Monetary & FX ---
    X['Yield_Curve'] = data['^TYX'] - data['^TNX']
    X['DXY_Mom'] = data['DX-Y.NYB'].pct_change(21, fill_method=None)

    # --- E. Sahm Rule "Danger Zone" Proxy ---
    X['Labor_Stress_Proxy'] = ((X['Yield_Curve'] < 0.1) & (X['Defensive_Rotation'] > X['Defensive_Rotation'].rolling(126).mean())).astype(int)

    # 3. Target Variable: Forward 1-Month Returns
    # 1 = Risk-On (SPY > 0), 0 = Risk-Off (SPY <= 0)
    horizon = 21
    future_return = data['SPY'].pct_change(horizon, fill_method=None).shift(-horizon)
    y = (future_return > 0).astype(int)

    # 4. Data Cleaning
    valid_data = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_clean = valid_data.drop(columns=['target'])
    y_clean = valid_data['target']

    # 5. Time-Series Split (Last 15% for Validation)
    split = int(len(X_clean) * 0.85)
    X_train, X_test = X_clean.iloc[:split], X_clean.iloc[split:]
    y_train, y_test = y_clean.iloc[:split], y_clean.iloc[split:]

    # Apply ffill within each split independently to prevent leakage
    X_train = X_train.ffill()
    X_test = X_test.ffill()

    print(f"Class distribution (train): {y_train.value_counts(normalize=True).round(2).to_dict()}")

    # 6. Scaling & Training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=3,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate on held-out test set
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Probability < threshold = Risk-Off; >= threshold = Risk-On
    # Using 0.55 as threshold since model probabilities are compressed (min~0.44, max~0.76)
    # Tune between 0.55-0.60: lower = more Risk-Off signals, higher = more conservative
    threshold = 0.55
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n🔍 Predicted probability stats (Risk-On):")
    print(f"   min={y_prob.min():.3f}  max={y_prob.max():.3f}  "
          f"mean={y_prob.mean():.3f}  "
          f"pct<0.55={(y_prob < 0.55).mean():.2%}  "
          f"pct<0.60={(y_prob < 0.60).mean():.2%}")

    print(f"\n📈 Out-of-Sample Evaluation (threshold={threshold}):")
    print(classification_report(y_test, y_pred,
                                target_names=['Risk-Off', 'Risk-On'],
                                zero_division=0))

    # 8. Model Driver Analysis
    importance = pd.DataFrame({
        'Feature': X_clean.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 Macro Model Driver Analysis (Top 10):")
    print(importance.head(10))

    # 9. Save Bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': X_clean.columns.tolist(),
        'target_horizon': horizon,
        'threshold': threshold
    }
    joblib.dump(bundle, model_path('risk_model.joblib'))
    print("\n✅ Training complete. 'risk_model.joblib' includes Sector Intelligence.")

if __name__ == "__main__":
    train_risk_regime_model()
