import yfinance as yf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# --- Tickers ---
sector_tickers = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']
macro_tickers = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD']

# --- Friendly sector names (optional for display) ---
SECTOR_NAMES = {
    'VDE':'Energy', 'XLB':'Materials', 'VIS':'Industrials', 
    'XLY':'Discr', 'XLF':'Financials', 'XLC':'Comm Serv', 
    'VGT':'Tech', 'XLV':'Health', 'XLP':'Staples', 
    'XLU':'Utilities', 'XLRE':'Real Estate'
}

# --- Load model ---
model_bundle = joblib.load('sector_model.joblib')
model = model_bundle['model']
scaler = model_bundle['scaler']
feature_names = model_bundle['features']

# --- Fetch recent data ---
data = yf.download(sector_tickers + macro_tickers, start="2000-01-01")['Close'].ffill()

# --- Feature engineering ---
X = pd.DataFrame(index=data.index)
horizon = 63

X['DXY_mom'] = data['DX-Y.NYB'].pct_change(horizon)
X['VIX_level'] = data['^VIX'].rolling(21).mean()
X['MOVE_level'] = data['^MOVE'].ffill().fillna(data['^MOVE'].mean())
X['Yield_Curve'] = data['^TYX'] - data['^TNX']
X['Credit_Spread'] = data['LQD'] / data['HYG']
X['TNX_vol'] = data['^TNX'].rolling(horizon).std()

# --- Last row for prediction ---
X_pred = X.tail(1)[feature_names]
X_scaled = scaler.transform(X_pred)

# --- Predict probabilities ---
proba = model.predict_proba(X_scaled)[0]
classes = model.classes_

# Map to friendly sector names
proba_dict = {SECTOR_NAMES.get(c, c): p for c, p in zip(classes, proba)}

# Sort sectors by probability
sorted_proba = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

# Top 3 best & bottom 3 worst
top3 = list(sorted_proba.items())[:3]
bottom3 = list(sorted_proba.items())[-3:]

print("📈 Top 3 sectors for next 3 months:")
for name, p in top3:
    print(f"{name}: {p:.2%}")

print("\n📉 Bottom 3 sectors for next 3 months:")
for name, p in bottom3:
    print(f"{name}: {p:.2%}")

print("\nAll sector probabilities:")
for name, p in sorted_proba.items():
    print(f"{name}: {p:.2%}")

