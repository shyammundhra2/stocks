import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
LOCAL_FILE = "sp500.csv"
MODEL_NAME = "macro_model_2027.json"

MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB',
                 '^IRX', '^TNX', 'HYG', 'LQD']

SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV',
                  'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

LOOKBACK_3M = 63
BUFFER_DAYS = LOOKBACK_3M * 2

# Performance settings
MAX_WORKERS = 10  # For parallel name fetching
BATCH_SIZE = 100  # Bulk download batch size (yfinance supports up to ~100)


def download_stocks_batch(tickers, start, end, batch_size=50):
    """Download stocks in bulk batches for maximum speed."""
    all_results = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  📦 Batch {batch_num}/{total_batches}: downloading {len(batch)} stocks...")

        try:
            # Bulk download entire batch at once
            data = yf.download(batch, start=start, end=end,
                               auto_adjust=True, progress=False,
                               group_by='ticker', threads=True)

            # Process each stock in the batch
            for ticker in batch:
                try:
                    # Handle single vs multi-ticker dataframe structure
                    if len(batch) == 1:
                        stock_data = data
                    else:
                        stock_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

                    if stock_data.empty or len(stock_data) < LOOKBACK_3M:
                        continue

                    mom_3m = stock_data['Close'].pct_change(LOOKBACK_3M, fill_method=None).iloc[-1]
                    if pd.isna(mom_3m):
                        continue

                    all_results.append({
                        'ticker': ticker,
                        'mom_3m': float(mom_3m)
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"  ⚠️  Batch {batch_num} failed, skipping...")
            continue

    return all_results


def get_company_names_batch(tickers):
    """Fetch company names in parallel."""
    names = {}

    def fetch_name(t):
        try:
            return t, yf.Ticker(t).info.get('shortName', t)
        except Exception:
            return t, t

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_name, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, name = future.result()
            names[ticker] = name

    return names


def infer_today(top_n=15, avoid_n=15):
    print("🚀 Starting optimized inference pipeline...")

    # ---------------- LOAD MODEL ----------------
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_NAME)
        print(f"✅ Model loaded: {MODEL_NAME}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Model file '{MODEL_NAME}' not found!")

    # ---------------- DATE RANGE ----------------
    end = datetime.now()
    start = end - timedelta(days=BUFFER_DAYS)
    print(f"📅 Date range: {start.date()} to {end.date()}")

    # ---------------- MACRO & SECTOR (Single batch download) ----------------
    print("📊 Downloading macro & sector data...")
    ms = yf.download(MACRO_TICKERS + SECTOR_TICKERS,
                     start=start, end=end,
                     auto_adjust=True, progress=False)['Close']

    m = pd.DataFrame(index=ms.index)
    m['Curve_Inversion'] = ms['^IRX'] - ms['^TNX']
    m['Bond_Stress'] = ms['^MOVE']
    m['Liquidity'] = ms['HYG'] / ms['LQD']
    m['Concentration'] = ms['SPY'] / ms['RSP']

    for s in SECTOR_TICKERS:
        m[f'{s}_3M'] = ms[s].pct_change(LOOKBACK_3M, fill_method=None)

    m = m.shift(1).dropna()
    macro_latest = m.iloc[-1].astype(float)
    print("✅ Macro features computed")

    # ---------------- STOCK FEATURES (Bulk Download) ----------------
    print("📈 Bulk downloading stock data...")
    symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()

    stock_results = download_stocks_batch(symbols, start, end, batch_size=BATCH_SIZE)

    print(f"✅ Downloaded {len(stock_results)}/{len(symbols)} valid stocks")

    # ---------------- GET COMPANY NAMES (Parallel) ----------------
    print("🏢 Fetching company names...")
    valid_tickers = [r['ticker'] for r in stock_results]
    ticker_to_name = get_company_names_batch(valid_tickers)
    print("✅ Company names retrieved")

    # ---------------- BUILD DATAFRAME ----------------
    print("🔧 Building feature matrix...")
    rows = []
    for result in stock_results:
        row = macro_latest.copy()
        row['Stock_Mom_3M'] = result['mom_3m']
        row['Ticker'] = result['ticker']
        row['Name'] = ticker_to_name[result['ticker']]
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------- FEATURES ----------------
    feats = ['Curve_Inversion', 'Bond_Stress', 'Liquidity',
             'Concentration', 'Stock_Mom_3M'] + \
            [f'{s}_3M' for s in SECTOR_TICKERS]

    df[feats] = df[feats].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=feats)

    if df.empty:
        raise ValueError("❌ No valid rows for inference. Check data sources.")

    print(f"✅ Feature matrix: {len(df)} stocks × {len(feats)} features")

    # ---------------- PREDICTIONS ----------------
    print("🤖 Running model predictions...")
    df['Prob'] = model.predict_proba(df[feats])[:, 1]
    df['Margin'] = model.predict(df[feats], output_margin=True)

    df['Confidence'] = pd.qcut(df['Prob'], q=[0, 0.33, 0.66, 1],
                               labels=['Avoid', 'Medium', 'Strong'])

    df = df.sort_values(['Prob', 'Margin'], ascending=False)

    # ---------------- OUTPUT ----------------
    print("\n" + "=" * 80)
    print("TOP 2027 STRATEGIC HOLDS")
    print("=" * 80)
    print(df[['Ticker', 'Name', 'Prob', 'Confidence']].head(top_n).to_string(index=False))

    print("\n" + "=" * 80)
    print("AVOID / UNDERWEIGHT")
    print("=" * 80)
    print(df[['Ticker', 'Name', 'Prob', 'Confidence']].tail(avoid_n).to_string(index=False))

    # Save output
    output_file = 'macro_2027_ranking_fast_with_names.csv'
    df[['Ticker', 'Name', 'Prob', 'Confidence']].to_csv(output_file, index=False)
    print(f"\n✅ Ranking CSV saved → {output_file}")
    print(f"⏱️  Pipeline complete!")

    return df


if __name__ == "__main__":
    infer_today()