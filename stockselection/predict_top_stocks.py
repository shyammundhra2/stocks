import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
LOOKBACK_STOP = 50  # Look back 50 days for max price stop loss
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5
BUFFER_DAYS = LOOKBACK_3M * 2

# Entry timing parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 45
RSI_OVERBOUGHT = 65
SMA_SHORT = 20
MIN_MOMENTUM = 0.10  # 10% minimum 3-month momentum

# Performance settings
MAX_WORKERS = 10
BATCH_SIZE = 50
RETRY_DELAY = 2


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def evaluate_entry_timing(stock_data, mom_3m):
    """
    Evaluate trading signal for stock.

    Signal Rules:
    - BUY: Strong momentum (>10%), RSI 45-65, Price > 20D SMA
    - HOLD: Moderate momentum (0-10%), or price near SMA, not extreme RSI
    - SELL: Negative momentum (<0%), or RSI extreme (>70 or <30), or price < SMA

    Returns: 'BUY', 'HOLD', or 'SELL' and component signals
    """
    close = stock_data['Close']

    # Calculate indicators
    rsi = calculate_rsi(close, RSI_PERIOD)
    sma_20 = close.rolling(SMA_SHORT).mean()

    current_price = close.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_sma = sma_20.iloc[-1]

    price_vs_sma = ((current_price - current_sma) / current_sma) * 100

    # Determine signal
    signal = "HOLD"  # Default

    # SELL conditions (highest priority)
    if (mom_3m < 0 or  # Negative momentum
            current_rsi > 70 or  # Overbought
            current_rsi < 30 or  # Oversold
            current_price < current_sma):  # Below trend
        signal = "SELL"

    # BUY conditions (all must be true)
    elif (mom_3m > MIN_MOMENTUM and  # Strong momentum
          RSI_OVERSOLD <= current_rsi <= RSI_OVERBOUGHT and  # Healthy RSI
          current_price > current_sma):  # Above trend
        signal = "BUY"

    # Otherwise HOLD (moderate conditions)

    return {
        'rsi': current_rsi,
        'sma_20': current_sma,
        'price_vs_sma': price_vs_sma,
        'signal': signal,
        'momentum_ok': mom_3m > MIN_MOMENTUM,
        'rsi_ok': RSI_OVERSOLD <= current_rsi <= RSI_OVERBOUGHT,
        'trend_ok': current_price > current_sma
    }


def download_stocks_batch(tickers, start, end, batch_size=50):
    """Download stocks in bulk batches with retry logic and signal evaluation."""
    all_results = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  📦 Batch {batch_num}/{total_batches}: downloading {len(batch)} stocks...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if i > 0:
                    time.sleep(RETRY_DELAY)

                # Download with full OHLC data
                data = yf.download(batch, start=start, end=end,
                                   auto_adjust=True, progress=False,
                                   group_by='ticker', threads=True)

                for ticker in batch:
                    try:
                        # Handle single vs multi-ticker dataframe structure
                        if len(batch) == 1:
                            stock_data = data
                        else:
                            stock_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

                        if stock_data.empty or len(stock_data) < max(LOOKBACK_3M, ATR_PERIOD, SMA_SHORT, LOOKBACK_STOP):
                            continue

                        # Calculate 3-month momentum
                        mom_3m = stock_data['Close'].pct_change(LOOKBACK_3M, fill_method=None).iloc[-1]
                        if pd.isna(mom_3m):
                            continue

                        # Calculate ATR
                        atr = calculate_atr(stock_data['High'],
                                            stock_data['Low'],
                                            stock_data['Close'],
                                            period=ATR_PERIOD)

                        current_price = stock_data['Close'].iloc[-1]
                        current_atr = atr.iloc[-1]

                        if pd.isna(current_atr) or pd.isna(current_price):
                            continue

                        # Calculate stop loss as: Max Price (last 50 days) - 2.5 * ATR
                        max_price_50d = stock_data['High'].tail(LOOKBACK_STOP).max()
                        stop_loss = max_price_50d - (ATR_MULTIPLIER * current_atr)
                        stop_pct = ((stop_loss - current_price) / current_price) * 100

                        # Evaluate trading signal
                        entry_signals = evaluate_entry_timing(stock_data, mom_3m)

                        all_results.append({
                            'ticker': ticker,
                            'mom_3m': float(mom_3m),
                            'current_price': float(current_price),
                            'atr': float(current_atr),
                            'stop_loss': float(stop_loss),
                            'stop_pct': float(stop_pct),
                            'rsi': float(entry_signals['rsi']),
                            'price_vs_sma': float(entry_signals['price_vs_sma']),
                            'signal': entry_signals['signal']
                        })
                    except Exception:
                        continue

                break

            except Exception as e:
                if '401' in str(e) or 'Unauthorized' in str(e):
                    if attempt < max_retries - 1:
                        wait_time = RETRY_DELAY * (attempt + 1)
                        print(
                            f"  ⚠️  401 error on batch {batch_num}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"  ❌ Batch {batch_num} failed after {max_retries} attempts, skipping...")
                else:
                    print(f"  ⚠️  Batch {batch_num} error: {str(e)[:50]}, skipping...")
                    break

    return all_results


def load_company_names_from_csv():
    """Load company names directly from CSV."""
    try:
        df = pd.read_csv(LOCAL_FILE)
        return dict(zip(df['Symbol'], df['Company']))
    except Exception as e:
        print(f"⚠️  Could not load names from CSV: {e}")
        return {}


def infer_today(top_n=25, avoid_n=25):
    print("🚀 Starting optimized inference pipeline with BUY/HOLD/SELL signals...")

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

    # ---------------- MACRO & SECTOR ----------------
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

    # ---------------- STOCK FEATURES ----------------
    print("📈 Bulk downloading stock data with signal evaluation...")
    symbols_df = pd.read_csv(LOCAL_FILE)
    symbols = symbols_df['Symbol'].tolist()

    ticker_to_name = load_company_names_from_csv()
    print(f"✅ Loaded {len(ticker_to_name)} company names from CSV")

    stock_results = download_stocks_batch(symbols, start, end, batch_size=BATCH_SIZE)
    print(f"✅ Downloaded {len(stock_results)}/{len(symbols)} valid stocks with signals")

    # ---------------- BUILD DATAFRAME ----------------
    print("🔧 Building feature matrix...")
    rows = []
    for result in stock_results:
        row = macro_latest.copy()
        row['Stock_Mom_3M'] = result['mom_3m']
        row['Ticker'] = result['ticker']
        row['Name'] = ticker_to_name.get(result['ticker'], result['ticker'])
        row['Current_Price'] = result['current_price']
        row['ATR'] = result['atr']
        row['Stop_Loss'] = result['stop_loss']
        row['Stop_Pct'] = result['stop_pct']
        row['RSI'] = result['rsi']
        row['Price_vs_SMA'] = result['price_vs_sma']
        row['Signal'] = result['signal']
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

    # ---------------- ENTRY TIMING EVALUATION ----------------
    print("⏱️  Evaluating entry timing with probability threshold...")

    # Evaluate entry criteria for each stock (now with probability)
    entry_signals = []
    for idx, row in df.iterrows():
        prob_ok = row['Prob'] > 0.5
        momentum_ok = row['Stock_Mom_3M'] > MIN_MOMENTUM
        rsi_ok = RSI_OVERSOLD <= row['RSI'] <= RSI_OVERBOUGHT
        trend_ok = row['Price_vs_SMA'] > 0  # Price above SMA

        # ALL conditions must be met
        entry_now = "BUY" if (prob_ok and momentum_ok and rsi_ok and trend_ok) else "WAIT"
        entry_signals.append(entry_now)

    df['Entry_Now'] = entry_signals

    df = df.sort_values(['Prob', 'Margin'], ascending=False)

    # ---------------- OUTPUT ----------------
    # Round numeric columns
    df['Prob'] = df['Prob'].round(2)
    df['Current_Price'] = df['Current_Price'].round(1)
    df['Stop_Loss'] = df['Stop_Loss'].round(1)
    df['Stop_Pct'] = df['Stop_Pct'].round(1)
    df['ATR'] = df['ATR'].round(1)
    df['RSI'] = df['RSI'].round(1)
    df['Price_vs_SMA'] = df['Price_vs_SMA'].round(1)

    print("\n" + "=" * 120)
    print("TOP 2027 STRATEGIC HOLDS (with BUY/HOLD/SELL Signals)")
    print("=" * 120)
    output_cols = ['Ticker', 'Name', 'Prob', 'Signal', 'RSI', 'Price_vs_SMA',
                   'Current_Price', 'Stop_Loss', 'Stop_Pct']
    print(df[output_cols].head(top_n).to_string(index=False))

    # Show buy signals
    buy_signals = df[df['Signal'] == 'BUY'].head(15)
    if not buy_signals.empty:
        print("\n" + "=" * 120)
        print(f"🟢 BUY SIGNALS ({len(df[df['Signal'] == 'BUY'])} total stocks)")
        print("=" * 120)
        print(buy_signals[output_cols].to_string(index=False))

    # Show hold signals
    hold_signals = df[df['Signal'] == 'HOLD'].head(10)
    if not hold_signals.empty:
        print("\n" + "=" * 120)
        print(f"🟡 HOLD SIGNALS ({len(df[df['Signal'] == 'HOLD'])} total stocks)")
        print("=" * 120)
        print(hold_signals[output_cols].to_string(index=False))

    # Show sell signals
    sell_signals = df[df['Signal'] == 'SELL'].head(15)
    if not sell_signals.empty:
        print("\n" + "=" * 120)
        print(f"🔴 SELL SIGNALS ({len(df[df['Signal'] == 'SELL'])} total stocks)")
        print("=" * 120)
        print(sell_signals[output_cols].to_string(index=False))

    print("\n" + "=" * 120)
    print("LOWEST RANKED STOCKS")
    print("=" * 120)
    print(df[output_cols].tail(avoid_n).to_string(index=False))

    # Save output
    output_file = 'macro_2027_ranking_with_signals.csv'
    df[output_cols].to_csv(output_file, index=False)
    print(f"\n✅ Ranking CSV saved → {output_file}")

    # Print signal summary
    total_buy = len(df[df['Signal'] == 'BUY'])
    total_hold = len(df[df['Signal'] == 'HOLD'])
    total_sell = len(df[df['Signal'] == 'SELL'])

    print(f"\n📊 Trading Signal Summary:")
    print(f"   • Total stocks analyzed: {len(df)}")
    print(f"   • 🟢 BUY signals: {total_buy} ({total_buy / len(df) * 100:.1f}%)")
    print(f"   • 🟡 HOLD signals: {total_hold} ({total_hold / len(df) * 100:.1f}%)")
    print(f"   • 🔴 SELL signals: {total_sell} ({total_sell / len(df) * 100:.1f}%)")
    print(f"\n   Signal Logic:")
    print(
        f"   • BUY: Momentum >{MIN_MOMENTUM * 100:.0f}%, RSI {RSI_OVERSOLD}-{RSI_OVERBOUGHT}, Price > {SMA_SHORT}D SMA")
    print(f"   • SELL: Momentum <0%, RSI >70 or <30, OR Price < {SMA_SHORT}D SMA")
    print(f"   • HOLD: Everything else (moderate conditions)")
    print(f"   • Stop Loss: Max Price (last {LOOKBACK_STOP} days) - ({ATR_MULTIPLIER} × {ATR_PERIOD}-day ATR)")
    print(f"⏱️  Pipeline complete!")

    return df


if __name__ == "__main__":
    infer_today()