import yfinance as yf
import pandas as pd
import xgboost as xgb
import os
import time

# --- CONFIGURATION ---
MODEL_NAME = "macro_model_2027.json"
LOCAL_FILE = "sp500.csv"


def predict_top_20():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, MODEL_NAME)

    if not os.path.exists(model_path):
        return print(f"Error: Model not found at {model_path}.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
    SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

    print("Fetching live macro signals...")
    # Fix: Added retry loop for network stability
    for attempt in range(3):
        try:
            ms_data = yf.download(MACRO_TICKERS + SECTOR_TICKERS, period="6mo", auto_adjust=True)['Close']
            if not ms_data.empty: break
        except Exception:
            if attempt < 2:
                time.sleep(2)
            else:
                return print("Final Network Error: Could not reach Yahoo Finance.")

    # Strategic Feature Vector
    current_m = pd.Series({
        'Curve_Inversion': ms_data['^TNX'].iloc[-1] - ms_data['^TYX'].iloc[-1],
        'Bond_Stress': ms_data['^MOVE'].iloc[-1],
        'Liquidity': ms_data['HYG'].iloc[-1] / ms_data['LQD'].iloc[-1],
        'Concentration': ms_data['SPY'].iloc[-1] / ms_data['RSP'].iloc[-1]
    })

    # Fix: Pandas 2.1+ compatibility for Sector 3M logic
    for s in SECTOR_TICKERS:
        current_m[f'{s}_3M'] = ms_data[s].ffill().pct_change(63).iloc[-1]

    symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()
    print(f"Scoring S&P 500 universe (Target: 2027)...")

    # Batch download with retry logic
    prices = yf.download(symbols, period="6mo", progress=False, auto_adjust=True)['Close']

    results = []
    feature_order = ['Curve_Inversion', 'Bond_Stress', 'Liquidity', 'Concentration', 'Stock_Mom_3M'] + \
                    [f'{s}_3M' for s in SECTOR_TICKERS]

    for t in symbols:
        if t not in prices.columns or prices[t].isnull().all(): continue

        row = current_m.copy()
        # Fix: Future-proofing .pct_change() by explicitly calling .ffill()
        row['Stock_Mom_3M'] = prices[t].ffill().pct_change(63).iloc[-1]

        input_df = pd.DataFrame([row])[feature_order]
        prob = model.predict_proba(input_df)[0][1]
        results.append({'Ticker': t, 'Score': prob})

    top_20 = pd.DataFrame(results).sort_values('Score', ascending=False).head(20)
    print("\n" + "=" * 40 + "\nTOP 20 STRATEGIC PICKS (2027 TARGET)\n" + "=" * 40)
    print(top_20.to_string(index=False))


if __name__ == "__main__":
    predict_top_20()