# predict_commodities.py

import yfinance as yf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from macro.constants import COMMODITIES, ML_MACRO_TICKERS

# Load model
model_bundle = joblib.load('commodity_model.joblib')
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_names = model_bundle['features']

commodity_tickers = list(COMMODITIES.keys())
macro_tickers = ML_MACRO_TICKERS

# Fetch recent data
data = yf.download(commodity_tickers + macro_tickers, start="2000-01-01")['Close'].ffill()

# Feature engineering (match training)
X = pd.DataFrame(index=data.index)

# Macro
X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63)
X['VIX_level'] = data['^VIX'].rolling(21).mean()
X['Yield_Curve'] = data['^TYX'] - data['^TNX']
X['Credit_Spread'] = data['LQD'] / data['HYG']

# Commodity-specific
for ticker in commodity_tickers:
    X[f'{ticker}_mom'] = data[ticker].pct_change(63)
    X[f'{ticker}_vol'] = data[ticker].rolling(63).std()
    X[f'{ticker}_mom_1m'] = data[ticker].pct_change(21)
    X[f'{ticker}_vol_1m'] = data[ticker].rolling(21).std()

# Last row for prediction
X_pred = X.tail(1)[feature_names]
X_scaled = scaler.transform(X_pred)

# Predict probabilities
proba = model.predict_proba(X_scaled)[0]
classes = model.classes_

# Map to friendly names
proba_dict = {COMMODITIES[c]: p for c, p in zip(classes, proba)}
sorted_proba = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

# Best and worst
best = list(sorted_proba.keys())[0]
worst = list(sorted_proba.keys())[-1]

print("📈 Predicted best commodity for next 3 months:", best)
print("📉 Predicted worst commodity for next 3 months:", worst)
print("\nAll probabilities:")
for name, p in sorted_proba.items():
    print(f"{name}: {p:.2%}")

