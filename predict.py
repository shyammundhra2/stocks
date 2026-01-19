import yfinance as yf
import pandas as pd
import joblib
import argparse
from datetime import timedelta, datetime
from macro.constants import SECTOR_NAMES, COMMODITIES, COUNTRIES, CURRENCIES, ML_MACRO_TICKERS

# ----------------- Helper Functions -----------------
def compute_features(data, model_type):
    """Compute features depending on model type with MultiIndex safety"""
    horizon_q = 63
    horizon_1m = 21
    X = pd.DataFrame(index=data.index)
    cols = data.columns

    # Forward fill once
    data = data.ffill().fillna(method='bfill')

    # --- Macro features ---
    if 'DX-Y.NYB' in cols:
        X['DXY_mom'] = data['DX-Y.NYB'].pct_change(horizon_q)
        X['DXY_Mom'] = data['DX-Y.NYB'].pct_change(horizon_1m)
    if '^VIX' in cols:
        X['VIX_level'] = data['^VIX'].rolling(21).mean()
        X['VIX_Level'] = data['^VIX']
    if '^MOVE' in cols:
        X['MOVE_level'] = data['^MOVE']
        X['MOVE_Level'] = data['^MOVE']
    if '^TYX' in cols and '^TNX' in cols:
        X['Yield_Curve'] = data['^TYX'] - data['^TNX']
        X['TNX_vol'] = data['^TNX'].rolling(horizon_q).std()
    if 'LQD' in cols and 'HYG' in cols:
        X['Credit_Spread'] = data['LQD'] / data['HYG']
        X['Credit_Spread_Proxy'] = X['Credit_Spread']

    # --- Risk-specific features ---
    if model_type == 'risk':
        if 'RSP' in cols and 'SPY' in cols:
            X['Breadth_Ratio'] = data['RSP'] / data['SPY']
            X['Breadth_MA_Diff'] = X['Breadth_Ratio'] - X['Breadth_Ratio'].rolling(50).mean()
            X['SPY_Trend'] = data['SPY'] / data['SPY'].rolling(200).mean()

        required_rot = ['XLU', 'XLP', 'XLK', 'XLY']
        if all(s in cols for s in required_rot):
            X['Defensive_Rotation'] = (data['XLU'] + data['XLP']) / (data['XLK'] + data['XLY'])

        if 'XLF' in cols and 'SPY' in cols:
            X['XLF_Relative_Strength'] = data['XLF'] / data['SPY']

        for s in ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']:
            if s in cols:
                X[f'{s}_3M_Mom'] = data[s].pct_change(63)

        if 'Yield_Curve' in X.columns and 'Defensive_Rotation' in X.columns:
            X['Labor_Stress_Proxy'] = ((X['Yield_Curve'] < 0.1) &
                                       (X['Defensive_Rotation'] > X['Defensive_Rotation'].rolling(126).mean())).astype(int)

    # --- Commodity and Country specific features ---
    if model_type in ['commodity', 'country']:
        target_dict = COMMODITIES if model_type == 'commodity' else COUNTRIES
        tickers = [c for c in cols if c in target_dict.keys()]
        for c in tickers:
            X[f'{c}_mom'] = data[c].pct_change(horizon_q)
            X[f'{c}_vol'] = data[c].rolling(horizon_q).std()
            X[f'{c}_mom_1m'] = data[c].pct_change(horizon_1m)
            X[f'{c}_vol_1m'] = data[c].rolling(horizon_1m).std()

    return X

def predict_assets(model_path, tickers, friendly_names, model_type, as_of_date=None):
    """Fetch data, compute features, and run prediction bundle"""
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    # Load model bundle
    bundle = joblib.load(model_path)
    model, scaler, feature_names = bundle['model'], bundle['scaler'], bundle['features']

    # Determine minimal start date for rolling windows
    max_rolling = 200  # largest rolling window used in compute_features
    buffer_days = 10   # safety buffer
    start_date = as_of_date - pd.Timedelta(days=int(max_rolling * 1.5 + buffer_days))

    # yfinance end is exclusive
    fetch_end = as_of_date + timedelta(days=1)

    # Download data in parallel
    raw_data = yf.download(tickers, start=start_date, end=fetch_end, progress=False, threads=True)
    data = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data

    X = compute_features(data, model_type)
    for feat in feature_names:
        if feat not in X.columns:
            X[feat] = 0

    X_scaled = scaler.transform(X.tail(1)[feature_names])
    proba = model.predict_proba(X_scaled)[0]

    if model_type == 'risk':
        proba_dict = {f"Class {c}": p for c, p in zip(model.classes_, proba)}
    else:
        proba_dict = {friendly_names.get(c, c): p for c, p in zip(model.classes_, proba)}

    return {
        'date': as_of_date.strftime('%Y-%m-%d'),
        'probabilities': dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
    }

def print_prediction(title, result):
    """Pretty-print every asset and its associated probability"""
    print(f"\n📊 {title.upper()} FULL PROBABILITIES ({result['date']})")
    print("-" * 60)
    for name, p in result['probabilities'].items():
        if "Class" in name:
            label = "Risk-On (Positive Outcome)" if "1" in name else "Risk-Off (Negative Outcome)"
            print(f"  {label:<35} {round(float(p) * 100, 2):>6}%")
        else:
            print(f"  {name:<35} {round(float(p) * 100, 2):>6}%")

# ----------------- Main Execution -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Prediction Dashboard")
    parser.add_argument("--date", type=str, default=None, help="Date to predict (YYYY-MM-DD)")
    args = parser.parse_args()

    model_configs = [
        ("Risk Regime", "risk_model.joblib", list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys()))), {}, 'risk'),
        ("Sector Rotation", "sector_model.joblib", list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS, SECTOR_NAMES, 'sector'),
        ("Commodities", "commodity_model.joblib", list(COMMODITIES.keys()) + ML_MACRO_TICKERS, COMMODITIES, 'commodity'),
        ("Countries", "country_model.joblib", list(COUNTRIES.keys()) + ML_MACRO_TICKERS + list(CURRENCIES.keys()), COUNTRIES, 'country')
    ]

    for title, path, tickers, names, m_type in model_configs:
        try:
            res = predict_assets(path, tickers, names, m_type, args.date)
            print_prediction(title, res)
        except FileNotFoundError:
            print(f"\n⚠️ File '{path}' not found. Skipping {title}.")
        except Exception as e:
            print(f"\n❌ Error processing {title}: {e}")
