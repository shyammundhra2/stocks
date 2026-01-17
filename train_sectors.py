import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib

# 1. Tickers for Sectors and Macro Metrics
sector_tickers = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']
macro_tickers = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE'] 

# 2. Download and Prep Sector Data
print("Downloading sector data...")
sector_data = yf.download(sector_tickers, start="2018-01-01")['Close']
returns = sector_data.pct_change(periods=5).shift(-5)
returns = returns.dropna()
y = returns.idxmax(axis=1)

# 3. Download Macro Data
print("Downloading macro features...")
X = yf.download(macro_tickers, start="2018-01-01")['Close'].ffill()

# 4. Align Datasets
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

# 5. Time-Series Cross-Validation
# We use 5 splits to simulate training on the past to predict the "future"
tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')

print("Executing Time-Series Cross-Validation...")
cv_results = cross_val_score(model, X, y, cv=tscv)

print(f"Mean CV Accuracy: {cv_results.mean():.2%}")
print(f"Stability (Std Dev): {cv_results.std():.2%}")

# 6. Final Fit & Save
# We train on the full dataset now that we've verified the logic
model.fit(X, y)

# Save feature importance for dashboard insights
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

joblib.dump(model, 'sector_model.joblib')
print("\nSuccess: 'sector_model.joblib' updated with CV-verified logic.")
