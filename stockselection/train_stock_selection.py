import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime
import os

# --- CONFIGURATION ---
START, END = "2001-01-01", datetime.now().strftime('%Y-%m-%d')
LOCAL_FILE = "sp500.csv"
MODEL_NAME = "macro_model_2027.json"  # <--- Added for persistence

# User-Defined Tickers for 2027 Prediction
MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB', '^TNX', '^TYX', 'HYG', 'LQD']
SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

def run_2027_strategy():
    # 1. Load Local Data
    try:
        symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()
    except FileNotFoundError:
        return print(f"Error: {LOCAL_FILE} not found. Ensure you created your Wikipedia-free CSV first.")

    # 2. Macro & Sector Pulse
    print("Syncing Macro Signals (MOVE, VIX, Spreads)...")
    ms_data = yf.download(MACRO_TICKERS + SECTOR_TICKERS, start=START, end=END, auto_adjust=True)['Close']

    m = pd.DataFrame(index=ms_data.index)
    m['Curve_Inversion'] = ms_data['^TNX'] - ms_data['^TYX']
    m['Bond_Stress'] = ms_data['^MOVE']
    m['Liquidity'] = ms_data['HYG'] / ms_data['LQD']
    m['Concentration'] = ms_data['SPY'] / ms_data['RSP']
    for s in SECTOR_TICKERS:
        m[f'{s}_3M'] = ms_data[s].pct_change(63)  # Sector Strength (Last 3 Months)

    # 3. Model Training (Predicting 2027)
    print("Training XGBoost on 12-Month Forward Returns...")
    train_pool = symbols[:100]  # Training on top 100 for efficiency
    all_features = []

    for t in train_pool:
        s_raw = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
        if s_raw.empty: continue

        s = pd.DataFrame(index=s_raw.index)
        s['Ticker'], s['Stock_Mom_3M'] = t, s_raw['Close'].pct_change(63)
        # TARGET: 252 trading days forward (approx 1 year)
        s['Target_2027'] = s_raw['Close'].pct_change(252).shift(-252)
        all_features.append(s.join(m).dropna())

    df_ml = pd.concat(all_features)
    feats = ['Curve_Inversion', 'Bond_Stress', 'Liquidity', 'Concentration', 'Stock_Mom_3M'] + [f'{s}_3M' for s in SECTOR_TICKERS]

    X, y = df_ml[feats], (df_ml['Target_2027'] > df_ml['Target_2027'].median()).astype(int)
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05)
    model.fit(X, y)

    # 4. PERSISTENCE LAYER: Save the Model
    # We save it in the same directory as the script to avoid pathing errors
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, MODEL_NAME)
    model.save_model(full_path)
    print(f"\nSUCCESS: Model logic saved to {full_path}")

    # 5. RANKING OUTPUT
    print("\n" + "=" * 60 + "\nFEATURE RANKING: Drivers for 2027 Strategic Picks\n" + "=" * 60)
    imp = pd.DataFrame({'Feature': feats, 'Imp': model.feature_importances_}).sort_values('Imp', ascending=False)
    for _, row in imp.head(15).iterrows():
        print(f"{row['Feature']:<20} | {'█' * int(row['Imp'] * 100)} {row['Imp']:.2%}")
    print("-" * 60)

if __name__ == "__main__":
    run_2027_strategy()