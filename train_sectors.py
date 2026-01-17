import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Tickers - Expanded for better Macro coverage
sector_tickers = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']
# Added TYX for Curve Slope and HYG/LQD for Credit Spread
macro_tickers = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD'] 

# 2. Data Fetching
data = yf.download(sector_tickers + macro_tickers, start="2015-01-01")['Close'].ffill()

# 3. Feature Engineering (The "Strategist" Layer)
X_raw = pd.DataFrame(index=data.index)
X_raw['DXY_mom'] = data['DX-Y.NYB'].pct_change(10) # 2-week momentum
X_raw['VIX_level'] = data['^VIX']
X_raw['MOVE_level'] = data['^MOVE']
X_raw['Yield_Curve'] = data['^TYX'] - data['^TNX'] # 30Y - 10Y
X_raw['Credit_Spread'] = data['LQD'] / data['HYG'] # IG vs HY proxy
X_raw['TNX_vol'] = data['^TNX'].rolling(10).std() # Yield volatility

# 4. Target Labeling (The "Portfolio Manager" Layer)
# Find the best performing sector over the NEXT 5 days
sector_returns = data[sector_tickers].pct_change(5).shift(-5)
y = sector_returns.idxmax(axis=1)

# 5. Final Alignment & Cleaning
features_and_target = pd.concat([X_raw, y.rename('target')], axis=1).dropna()
X = features_and_target.drop(columns=['target'])
y = features_and_target['target']

# 6. Scaling (Crucial for interpreting 'importance' across different units)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 7. Time-Series Walk-Forward Validation
tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(
    n_estimators=500, # Increased for stability
    max_depth=4,      # Kept shallow to prevent overfitting macro noise
    class_weight='balanced',
    random_state=42
)

# 
# This diagram would illustrate how we train on early dates and test on later dates 
# to ensure no data leakage.

print("Executing Walk-Forward Analysis...")
scores = []
for train_index, test_index in tscv.split(X_scaled):
    model.fit(X_scaled.iloc[train_index], y.iloc[train_index])
    scores.append(model.score(X_scaled.iloc[test_index], y.iloc[test_index]))

print(f"Mean Predictive Accuracy: {np.mean(scores):.2%}")

# 8. Final Fit and Export
model.fit(X_scaled, y)
joblib.dump({'model': model, 'scaler': scaler}, 'enhanced_sector_model.joblib')
