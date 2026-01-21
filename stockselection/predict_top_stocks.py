import yfinance as yf
import pandas as pd
import xgboost as xgb
import os
import time

# --- CONFIGURATION ---
MODEL_NAME = "macro_model_2027.json"
LOCAL_FILE = "sp500.csv"


def predict_top():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, MODEL_NAME)

    if not os.path.exists(model_path):
        return print(f"Error: Model not found at {model_path}. Run training first.")

    # Load Model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load Tickers and Names from local CSV
    try:
        ticker_df = pd.read_csv(LOCAL_FILE)
        # Create a mapping dictionary for quick name lookup
        name_map = dict(zip(ticker_df['Symbol'], ticker_df['Company']))
        symbols = ticker_df['Symbol'].tolist()
    except Exception as e:
        return print(f"Error loading {LOCAL_FILE}: {e}")

    MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
    SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

    print("Fetching live macro signals...")
    for attempt in range(3):
        try:
            ms_data = yf.download(MACRO_TICKERS + SECTOR_TICKERS, period="6mo", auto_adjust=True)['Close']
            if not ms_data.empty: break
        except Exception:
            if attempt < 2:
                time.sleep(2)
            else:
                return print("Final Network Error: Could not reach Yahoo Finance.")

    # Build Strategic Feature Vector
    current_m = pd.Series({
        'Curve_Inversion': ms_data['^TNX'].iloc[-1] - ms_data['^TYX'].iloc[-1],
        'Bond_Stress': ms_data['^MOVE'].iloc[-1],
        'Liquidity': ms_data['HYG'].iloc[-1] / ms_data['LQD'].iloc[-1],
        'Concentration': ms_data['SPY'].iloc[-1] / ms_data['RSP'].iloc[-1]
    })

    for s in SECTOR_TICKERS:
        current_m[f'{s}_3M'] = ms_data[s].ffill().pct_change(63).iloc[-1]

    print(f"Scoring S&P 500 universe (Target: 2027)...")

    # Batch download stock prices
    prices = yf.download(symbols, period="6mo", progress=False, auto_adjust=True)['Close']

    results = []
    feature_order = ['Curve_Inversion', 'Bond_Stress', 'Liquidity', 'Concentration', 'Stock_Mom_3M'] + \
                    [f'{s}_3M' for s in SECTOR_TICKERS]

    for t in symbols:
        # Check if ticker exists in downloaded data and has sufficient history
        if t not in prices.columns or len(prices[t].dropna()) < 64:
            continue

        try:
            row = current_m.copy()
            row['Stock_Mom_3M'] = prices[t].ffill().pct_change(63).iloc[-1]

            # Predict
            input_df = pd.DataFrame([row])[feature_order]
            prob = model.predict_proba(input_df)[0][1]

            results.append({
                'Ticker': t,
                'Name': name_map.get(t, "N/A"),  # Fetch name from local CSV
                'Score': prob
            })
        except Exception:
            continue

    # Create DataFrame and sort
    top = pd.DataFrame(results).sort_values('Score', ascending=False).head(50)

    # Format the output for a clean terminal look
    print("\n" + "=" * 75)
    print(f"{'TICKER':<10} {'NAME':<45} {'PROBABILITY'}")
    print("=" * 75)
    for _, row in top.iterrows():
        print(f"{row['Ticker']:<10} {row['Name'][:43]:<45} {row['Score']:.2%}")
    print("=" * 75)


if __name__ == "__main__":
    predict_top()