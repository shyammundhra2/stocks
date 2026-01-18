import yfinance as yf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from macro.constants import COUNTRIES, ML_MACRO_TICKERS, CURRENCIES

# Load trained model
model_bundle = joblib.load('country_model_with_fx.joblib')
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_names = model_bundle['features']

country_tickers = list(COUNTRIES.keys())
macro_tickers = ML_MACRO_TICKERS
currency_tickers = list(CURRENCIES.keys())

# Fetch historical data
data = yf.download(country_tickers + macro_tickers + currency_tickers, start="2000-01-01")['Close'].ffill()

# Feature engineering
X = pd.DataFrame(index=data.index)

# Macro
X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63)
X['VIX_level'] = data['^VIX'].rolling(21).mean()
X['Yield_Curve'] = data['^TYX'] - data['^TNX']
X['Credit_Spread'] = data['LQD'] / data['HYG']

# Country-specific + FX trends
for ticker in country_tickers:
    X[f'{ticker}_mom'] = data[ticker].pct_change(63)
    X[f'{ticker}_vol'] = data[ticker].rolling(63).std()
    X[f'{ticker}_mom_1m'] = data[ticker].pct_change(21)
    X[f'{ticker}_vol_1m'] = data[ticker].rolling(21).std()

    # Add FX momentum features
    country_name = COUNTRIES[ticker]
    fx_pair = None
    for k, v in CURRENCIES.items():
        if v == country_name:
            fx_pair = k
            break
    if fx_pair and fx_pair in data.columns:
        X[f'{ticker}_FX_mom'] = data[fx_pair].pct_change(63)
        X[f'{ticker}_FX_mom_1m'] = data[fx_pair].pct_change(21)

# Last row for prediction
X_pred = X.tail(1)[feature_names]
X_scaled = scaler.transform(X_pred)

# Predict probabilities
proba = model.predict_proba(X_scaled)[0]
classes = model.classes_

# Map to friendly country names
proba_dict = {COUNTRIES[c]: p for c, p in zip(classes, proba)}
sorted_proba = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

# Best & worst countries
best = list(sorted_proba.keys())[0]
worst = list(sorted_proba.keys())[-1]

print("📈 Predicted best country ETF for next 3 months:", best)
print("📉 Predicted worst country ETF for next 3 months:", worst)
print("\nAll probabilities:")
for name, p in sorted_proba.items():
    print(f"{name}: {p:.2%}")

