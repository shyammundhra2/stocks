import yfinance as yf
import pandas as pd
import xgboost as xgb
import os
import argparse
from datetime import datetime, timedelta

# --- CONFIGURATION ---
MODEL_NAME = "macro_model_2027.json"
LOCAL_FILE = "sp500.csv"


def valid_date(s):
    """Check if the provided string is a valid YYYY-MM-DD date."""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'. Expected YYYY-MM-DD.")


def predict_top(target_dt=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, MODEL_NAME)

    if not os.path.exists(model_path):
        return print(f"Error: Model not found at {model_path}.")

    # If no date is passed from CLI, use today's date
    if target_dt is None:
        target_dt = datetime.now()

    start_dt = target_dt - timedelta(days=200)
    print(f"\n[Strategy Snapshot: {target_dt.strftime('%Y-%m-%d')}]")

    # Load Model and Data
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    ticker_df = pd.read_csv(LOCAL_FILE)
    name_map = dict(zip(ticker_df['Symbol'], ticker_df['Company']))
    symbols = ticker_df['Symbol'].tolist()

    MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
    SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

    # Fetch Data for that specific window
    ms_data = yf.download(MACRO_TICKERS + SECTOR_TICKERS, start=start_dt, end=target_dt, auto_adjust=True)['Close']

    # Build Features
    current_m = pd.Series({
        'Curve_Inversion': ms_data['^TNX'].iloc[-1] - ms_data['^TYX'].iloc[-1],
        'Bond_Stress': ms_data['^MOVE'].iloc[-1],
        'Liquidity': ms_data['HYG'].iloc[-1] / ms_data['LQD'].iloc[-1],
        'Concentration': ms_data['SPY'].iloc[-1] / ms_data['RSP'].iloc[-1]
    })
    for s in SECTOR_TICKERS:
        current_m[f'{s}_3M'] = ms_data[s].ffill().pct_change(63).iloc[-1]

    # Predict
    prices = yf.download(symbols, start=start_dt, end=target_dt, progress=False, auto_adjust=True)['Close']
    results = []
    feature_order = ['Curve_Inversion', 'Bond_Stress', 'Liquidity', 'Concentration', 'Stock_Mom_3M'] + \
                    [f'{s}_3M' for s in SECTOR_TICKERS]

    for t in symbols:
        if t not in prices.columns or len(prices[t].dropna()) < 64: continue
        try:
            row = current_m.copy()
            row['Stock_Mom_3M'] = prices[t].ffill().pct_change(63).iloc[-1]
            prob = model.predict_proba(pd.DataFrame([row])[feature_order])[0][1]
            results.append({'Ticker': t, 'Name': name_map.get(t, "N/A"), 'Score': prob})
        except:
            continue

    top = pd.DataFrame(results).sort_values('Score', ascending=False).head(50)

    # Print Output Table
    print("\n" + "=" * 75)
    print(f"{'TICKER':<10} {'NAME':<45} {'PROBABILITY'}")
    print("=" * 75)
    for _, r in top.iterrows():
        print(f"{r['Ticker']:<10} {r['Name'][:43]:<45} {r['Score']:.2%}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict top macro picks for a specific date.")
    parser.add_argument("--date", type=valid_date, help="The target date (YYYY-MM-DD)")
    args = parser.parse_args()

    predict_top(target_dt=args.date)