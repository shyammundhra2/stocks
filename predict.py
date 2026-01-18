import yfinance as yf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from macro.constants import SECTOR_NAMES, COMMODITIES, COUNTRIES, CURRENCIES, ML_MACRO_TICKERS
import argparse

# ----------------- Helper Functions -----------------
def compute_features(data, model_type):
    """Compute features depending on model type"""
    horizon_q = 63
    horizon_1m = 21
    X = pd.DataFrame(index=data.index)

    # --- Macro features (for all models) ---
    if 'DX-Y.NYB' in data.columns:
        X['DXY_mom'] = data['DX-Y.NYB'].pct_change(horizon_q)
    if '^VIX' in data.columns:
        X['VIX_level'] = data['^VIX'].rolling(21).mean()
    if '^MOVE' in data.columns:
        X['MOVE_level'] = data['^MOVE'].ffill().fillna(data['^MOVE'].mean())
    if '^TYX' in data.columns and '^TNX' in data.columns:
        X['Yield_Curve'] = data['^TYX'] - data['^TNX']
        X['TNX_vol'] = data['^TNX'].rolling(horizon_q).std()
    if 'LQD' in data.columns and 'HYG' in data.columns:
        X['Credit_Spread'] = data['LQD'] / data['HYG']

    # --- Commodity-specific features ---
    if model_type == 'commodity':
        commodity_tickers = [c for c in data.columns if c in COMMODITIES.keys()]
        for c in commodity_tickers:
            X[f'{c}_mom'] = data[c].pct_change(horizon_q)
            X[f'{c}_vol'] = data[c].rolling(horizon_q).std()
            X[f'{c}_mom_1m'] = data[c].pct_change(horizon_1m)
            X[f'{c}_vol_1m'] = data[c].rolling(horizon_1m).std()

    # --- Country-specific features ---
    if model_type == 'country':
        country_tickers = [c for c in data.columns if c in COUNTRIES.keys()]
        for c in country_tickers:
            X[f'{c}_mom'] = data[c].pct_change(horizon_q)
            X[f'{c}_vol'] = data[c].rolling(horizon_q).std()
            X[f'{c}_mom_1m'] = data[c].pct_change(horizon_1m)
            X[f'{c}_vol_1m'] = data[c].rolling(horizon_1m).std()

    return X

def predict_assets(model_path, tickers, friendly_names, model_type, as_of_date=None, start_date="2000-01-01", top_n=3):
    """Predict top/bottom assets as of a specific date"""
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    # --- Load model ---
    model_bundle = joblib.load(model_path)
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['features']

    # --- Download data ---
    data = yf.download(tickers, start=start_date, end=as_of_date)['Close'].ffill()

    # --- Compute features ---
    X = compute_features(data, model_type)

    # --- Match features expected by model ---
    X_pred = X.tail(1)[feature_names]
    X_scaled = scaler.transform(X_pred)

    # --- Predict probabilities ---
    proba = model.predict_proba(X_scaled)[0]
    classes = model.classes_
    proba_dict = {friendly_names.get(c, c): p for c, p in zip(classes, proba)}
    sorted_proba = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

    top = list(sorted_proba.keys())[:top_n]
    bottom = list(sorted_proba.keys())[-top_n:]

    return {
        'date': as_of_date.strftime('%Y-%m-%d'),
        'probabilities': sorted_proba,
        'top': top,
        'bottom': bottom
    }

def print_prediction(title, result):
    """Pretty-print top/bottom assets and all probabilities"""
    print(f"\n📆 {title} prediction for {result['date']}\n")
    
    # Top N
    print("📈 Top assets:")
    for i, name in enumerate(result['top'], 1):
        prob = round(float(result['probabilities'][name]) * 100, 2)
        print(f"  {i}. {name} ({prob}%)")
    
    # Bottom N
    print("\n📉 Bottom assets:")
    for i, name in enumerate(result['bottom'], 1):
        prob = round(float(result['probabilities'][name]) * 100, 2)
        print(f"  {i}. {name} ({prob}%)")
    
    # Full probability list
    print("\nAll probabilities:")
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for name, p in sorted_probs:
        print(f"  {name}: {round(float(p) * 100, 2)}%")

# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict top/bottom sectors, commodities, or countries for a given date.")
    parser.add_argument("--date", type=str, default=None, help="Date to predict (YYYY-MM-DD), default is today")
    parser.add_argument("--top", type=int, default=3, help="Number of top/bottom assets to return")
    args = parser.parse_args()

    as_of_date = args.date
    top_n = args.top

    # --- Sectors ---
    sector_model_path = "sector_model.joblib"
    sector_result = predict_assets(
        model_path=sector_model_path,
        tickers=list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS,
        friendly_names=SECTOR_NAMES,
        model_type='sector',
        as_of_date=as_of_date,
        top_n=top_n
    )
    print_prediction("Sector", sector_result)

    # --- Commodities ---
    commodity_model_path = "commodity_model.joblib"
    commodity_result = predict_assets(
        model_path=commodity_model_path,
        tickers=list(COMMODITIES.keys()) + ML_MACRO_TICKERS,
        friendly_names=COMMODITIES,
        model_type='commodity',
        as_of_date=as_of_date,
        top_n=top_n
    )
    print_prediction("Commodity", commodity_result)

    # --- Countries ---
    country_model_path = "country_model.joblib"
    currency_tickers = list(CURRENCIES.keys())
    country_result = predict_assets(
        model_path=country_model_path,
        tickers=list(COUNTRIES.keys()) + ML_MACRO_TICKERS + currency_tickers,
        friendly_names=COUNTRIES,
        model_type='country',
        as_of_date=as_of_date,
        top_n=top_n
    )
    print_prediction("Country", country_result)

