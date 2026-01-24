import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
LOCAL_FILE = "sp500.csv"
MODEL_NAME = "macro_model_2027.json"

MACRO_TICKERS = ['SPY', 'RSP', '^VIX', '^MOVE', 'DX-Y.NYB',
                 '^IRX', '^TNX', 'HYG', 'LQD']

SECTOR_TICKERS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV',
                  'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

LOOKBACK_3M = 63
LOOKBACK_STOP = 50  # Look back 50 days for 50-day high
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5
BUFFER_DAYS = LOOKBACK_3M * 3  # give more data for smoothing

# Entry timing parameters
RSI_PERIOD = 14
DMA_200 = 200

# Performance settings
BATCH_SIZE = 50
RETRY_DELAY = 2


# ---------------- INDICATORS ----------------
def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ---------------- SIGNAL LOGIC ----------------
def evaluate_signal(stock_data, prob):
    """
    Simplified trading signal:
    - BUY: probability > 50% and RSI < 70
    - SELL: price < 50-day high - 2.5*ATR OR < 200-day SMA (optional)
    - HOLD: everything else
    """
    close = stock_data['Close']
    high = stock_data['High']
    low = stock_data['Low']
    current_price = close.iloc[-1]

    # Indicators
    rsi = calculate_rsi(close, RSI_PERIOD).iloc[-1]
    atr = calculate_atr(high, low, close, ATR_PERIOD).iloc[-1]
    max_high_50d = high.tail(LOOKBACK_STOP).max()
    sma_200 = close.rolling(DMA_200).mean().iloc[-1]

    # Handle missing SMA
    sma_200_check = False
    if not pd.isna(sma_200):
        sma_200_check = current_price < sma_200

    # BUY
    if prob > 0.5 and rsi < 70:
        signal = "BUY"
    # SELL
    elif current_price < (max_high_50d - ATR_MULTIPLIER * atr) or sma_200_check:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        'RSI': rsi,
        'ATR': atr,
        'Max_High_50D': max_high_50d,
        'SMA_200': sma_200,
        'Signal': signal
    }


# ---------------- DATA DOWNLOAD ----------------
def download_stocks_batch(tickers, start, end, batch_size=50):
    """Download stocks in batches with retries."""
    all_results = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"📦 Downloading batch {batch_num}/{total_batches}...")

        for attempt in range(3):
            try:
                if i > 0:
                    time.sleep(RETRY_DELAY)

                data = yf.download(batch, start=start, end=end,
                                   auto_adjust=True, progress=False,
                                   group_by='ticker', threads=True)

                for ticker in batch:
                    try:
                        stock_data = data[ticker] if len(batch) > 1 else data
                        if stock_data.empty:
                            print(f"⚠️ Skipping {ticker}: no data")
                            continue
                        all_results.append({'ticker': ticker, 'data': stock_data})
                    except Exception as e:
                        print(f"⚠️ Error processing {ticker}: {str(e)[:50]}")
                        continue

                break
            except Exception as e:
                print(f"⚠️ Batch download error: {str(e)[:50]}, retrying...")
                time.sleep(RETRY_DELAY)

    return all_results


def load_company_names_from_csv():
    try:
        df = pd.read_csv(LOCAL_FILE)
        return dict(zip(df['Symbol'], df['Company']))
    except Exception as e:
        print(f"⚠️ Could not load names from CSV: {e}")
        return {}


# ---------------- INFERENCE PIPELINE ----------------
def infer_today(top_n=25, avoid_n=25):
    print("🚀 Starting inference pipeline with simplified signals...")

    # Load model
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_NAME)
        print(f"✅ Model loaded: {MODEL_NAME}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Model file '{MODEL_NAME}' not found!")

    # Date range
    end = datetime.now()
    start = end - timedelta(days=BUFFER_DAYS)
    print(f"📅 Date range: {start.date()} to {end.date()}")

    # Macro & sector data
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

    # Stock data
    print("📈 Downloading stock data...")
    symbols_df = pd.read_csv(LOCAL_FILE)
    symbols = symbols_df['Symbol'].tolist()
    ticker_to_name = load_company_names_from_csv()
    stock_results = download_stocks_batch(symbols, start, end, batch_size=BATCH_SIZE)
    print(f"✅ Downloaded {len(stock_results)}/{len(symbols)} stocks")

    # Build dataframe
    feats = ['Curve_Inversion', 'Bond_Stress', 'Liquidity',
             'Concentration', 'Stock_Mom_3M'] + [f'{s}_3M' for s in SECTOR_TICKERS]

    rows = []
    for result in stock_results:
        ticker = result['ticker']
        data = result['data']

        # Minimum data check
        min_len = max(LOOKBACK_3M, ATR_PERIOD, LOOKBACK_STOP)
        if len(data) < min_len:
            print(f"⚠️ Skipping {ticker}: not enough data ({len(data)} rows)")
            continue

        # Stock momentum
        mom_3m = data['Close'].pct_change(LOOKBACK_3M, fill_method=None).iloc[-1]
        if pd.isna(mom_3m):
            print(f"⚠️ Skipping {ticker}: 3M momentum is NaN")
            continue

        macro_features = macro_latest.copy()
        macro_features['Stock_Mom_3M'] = mom_3m

        X = pd.DataFrame([macro_features])
        X = X[feats].apply(pd.to_numeric, errors='coerce')
        if X.isna().any().any():
            print(f"⚠️ Skipping {ticker}: features contain NaN")
            continue

        # Model probability
        prob = model.predict_proba(X)[:, 1][0]

        # Evaluate simplified signal
        sig = evaluate_signal(data, prob)

        row = macro_features.copy()
        row['Ticker'] = ticker
        row['Name'] = ticker_to_name.get(ticker, ticker)
        row['Current_Price'] = data['Close'].iloc[-1]
        row['RSI'] = sig['RSI']
        row['ATR'] = sig['ATR']
        row['SMA_200'] = sig['SMA_200']
        row['Signal'] = sig['Signal']
        row['Prob'] = prob

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("❌ No valid stocks to process. Check CSV and data availability.")
        return df

    df = df.sort_values('Prob', ascending=False)
    df['Prob'] = df['Prob'].round(2)
    df['Current_Price'] = df['Current_Price'].round(1)
    df['RSI'] = df['RSI'].round(1)
    df['ATR'] = df['ATR'].round(1)
    df['SMA_200'] = df['SMA_200'].round(1)

    output_cols = ['Ticker', 'Name', 'Prob', 'Signal', 'RSI', 'Current_Price', 'ATR', 'SMA_200']
    print("\n" + "=" * 120)
    print("TOP STRATEGIC HOLDS (with BUY/HOLD/SELL Signals)")
    print("=" * 120)
    print(df[output_cols].head(top_n).to_string(index=False))

    # Save CSV
    df[output_cols].to_csv('macro_2027_ranking_simplified.csv', index=False)
    print(f"\n✅ Ranking CSV saved → macro_2027_ranking_simplified.csv")

    # Signal summary
    total_buy = len(df[df['Signal'] == 'BUY'])
    total_hold = len(df[df['Signal'] == 'HOLD'])
    total_sell = len(df[df['Signal'] == 'SELL'])
    print(f"\n📊 Trading Signal Summary:")
    print(f"   • Total stocks analyzed: {len(df)}")
    print(f"   • 🟢 BUY: {total_buy} ({total_buy / len(df) * 100:.1f}%)")
    print(f"   • 🟡 HOLD: {total_hold} ({total_hold / len(df) * 100:.1f}%)")
    print(f"   • 🔴 SELL: {total_sell} ({total_sell / len(df) * 100:.1f}%)")

    return df


if __name__ == "__main__":
    infer_today()
