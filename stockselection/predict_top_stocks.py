import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from scipy import stats
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
LOOKBACK_1M = 21  # ~1 month for slope/R²
LOOKBACK_STOP = 50
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.5
BUFFER_DAYS = LOOKBACK_3M * 3

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
    if len(close) < period + 1:
        return pd.Series([np.nan] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_slope_r2(close, period=21):
    """
    Calculate linear regression slope and R² for the last 'period' days
    Returns: (slope, r2, p_value)
    """
    if len(close) < period:
        return np.nan, np.nan, np.nan

    y = close.tail(period).values
    x = np.arange(len(y))

    if len(y) < 2 or np.all(np.isnan(y)):
        return np.nan, np.nan, np.nan

    # Remove NaN values
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan

    x_clean = x[mask]
    y_clean = y[mask]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    r2 = r_value ** 2

    # Normalize slope as percentage change per day
    slope_pct = (slope / y_clean.mean()) * 100 if y_clean.mean() != 0 else 0

    return slope_pct, r2, p_value


def calculate_rolling_slopes(close, period=21, n_months=6):
    """
    Calculate slopes for the last n_months (rolling)
    Returns list of slopes for zscore calculation

    Default changed to 6 months for more stable Z-score calculation
    """
    slopes = []

    # Calculate slope for each of the last n_months
    for i in range(n_months):
        start_idx = -(period * (i + 1))
        end_idx = -(period * i) if i > 0 else None

        window = close.iloc[start_idx:end_idx]
        if len(window) < period:
            continue

        slope, _, _ = calculate_slope_r2(window, period=period)
        if not np.isnan(slope):
            slopes.append(slope)

    return slopes


def calculate_zscore(value, historical_values):
    """Calculate zscore of current value vs historical values"""
    if len(historical_values) < 2:
        return np.nan

    mean = np.mean(historical_values)
    std = np.std(historical_values, ddof=1)

    if std == 0:
        return np.nan

    return (value - mean) / std


# ---------------- SIGNAL LOGIC ----------------
def evaluate_signal(stock_data, prob):
    """
    Enhanced trading signal with slope/R² analysis:
    - BUY: probability > 50% AND RSI < 70 AND slope_3m > 0 AND R² > 0.4
    - TRIM: slope zscore > 2.5 (momentum overextension - uses 6-month lookback)
    - SELL: price < 50-day high - 2.5*ATR OR < 200-day SMA
    - HOLD: everything else
    """
    close = stock_data['Close']
    high = stock_data['High']
    low = stock_data['Low']
    current_price = close.iloc[-1]

    # Basic indicators
    rsi = calculate_rsi(close, RSI_PERIOD).iloc[-1]
    atr = calculate_atr(high, low, close, ATR_PERIOD).iloc[-1]
    max_high_50d = high.tail(LOOKBACK_STOP).max()
    sma_200 = close.rolling(DMA_200).mean().iloc[-1]

    # Slope and R² analysis (1 month)
    slope_1m, r2_1m, p_value = calculate_slope_r2(close, period=LOOKBACK_1M)

    # Calculate last 6 months of slopes for zscore (more stable)
    historical_slopes = calculate_rolling_slopes(close, period=LOOKBACK_1M, n_months=6)

    # Current slope zscore
    slope_zscore = np.nan
    if not np.isnan(slope_1m) and len(historical_slopes) >= 3:  # Need at least 3 months
        slope_zscore = calculate_zscore(slope_1m, historical_slopes)

    # Average slope over last 3 months (for trend direction)
    slope_3m_avg = np.mean(historical_slopes[:3]) if len(historical_slopes) >= 3 else np.nan

    # Handle missing SMA
    sma_200_check = False
    if not pd.isna(sma_200):
        sma_200_check = current_price < sma_200

    # SIGNAL LOGIC
    signal = "HOLD"

    # TRIM: Momentum overextension (increased threshold to 2.5)
    if not np.isnan(slope_zscore) and slope_zscore > 2.5:
        signal = "TRIM"

    # SELL: Stop loss or trend break
    elif current_price < (max_high_50d - ATR_MULTIPLIER * atr) or sma_200_check:
        signal = "SELL"

    # BUY: Model signal + momentum confirmation
    elif (prob > 0.5 and
          rsi < 70 and
          not np.isnan(slope_3m_avg) and slope_3m_avg > 0 and
          not np.isnan(r2_1m) and r2_1m > 0.4):
        signal = "BUY"

    return {
        'RSI': rsi,
        'ATR': atr,
        'Max_High_50D': max_high_50d,
        'SMA_200': sma_200,
        'Slope_1M': slope_1m,
        'R2_1M': r2_1m,
        'Slope_3M_Avg': slope_3m_avg,
        'Slope_ZScore': slope_zscore,
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
    print("🚀 Starting inference pipeline with slope/R² analysis...")

    # Load model
    try:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_NAME)
        print(f"✅ Model loaded: {MODEL_NAME}")
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Model file '{MODEL_NAME}' not found!")

    # Date range - need more history for slope calculations
    # Request ~500 calendar days to ensure we get 252+ trading days (1 year)
    end = datetime.now()
    start = end - timedelta(days=500)
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

        # Minimum data check - need 6 months for slope zscore analysis
        min_len = max(LOOKBACK_3M, ATR_PERIOD, LOOKBACK_STOP, LOOKBACK_1M * 6)
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

        # Evaluate enhanced signal
        sig = evaluate_signal(data, prob)

        row = macro_features.copy()
        row['Ticker'] = ticker
        row['Name'] = ticker_to_name.get(ticker, ticker)
        row['Current_Price'] = data['Close'].iloc[-1]
        row['RSI'] = sig['RSI']
        row['ATR'] = sig['ATR']
        row['SMA_200'] = sig['SMA_200']
        row['Slope_1M'] = sig['Slope_1M']
        row['R2_1M'] = sig['R2_1M']
        row['Slope_3M_Avg'] = sig['Slope_3M_Avg']
        row['Slope_ZScore'] = sig['Slope_ZScore']
        row['Signal'] = sig['Signal']
        row['Prob'] = prob

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("❌ No valid stocks to process. Check CSV and data availability.")
        return df

    df = df.sort_values('Prob', ascending=False)
    df['Prob'] = df['Prob'].round(3)
    df['Current_Price'] = df['Current_Price'].round(2)
    df['RSI'] = df['RSI'].round(1)
    df['ATR'] = df['ATR'].round(2)
    df['SMA_200'] = df['SMA_200'].round(2)
    df['Slope_1M'] = df['Slope_1M'].round(3)
    df['R2_1M'] = df['R2_1M'].round(3)
    df['Slope_3M_Avg'] = df['Slope_3M_Avg'].round(3)
    df['Slope_ZScore'] = df['Slope_ZScore'].round(2)

    output_cols = ['Ticker', 'Name', 'Prob', 'Signal', 'Slope_1M', 'R2_1M',
                   'Slope_3M_Avg', 'Slope_ZScore', 'RSI', 'Current_Price']

    print("\n" + "=" * 140)
    print("TOP STRATEGIC HOLDS (Enhanced with Slope/R² Analysis)")
    print("=" * 140)
    print(df[output_cols].head(top_n).to_string(index=False))

    # Save CSV
    all_output_cols = output_cols + ['ATR', 'SMA_200']
    df[all_output_cols].to_csv('macro_2027_ranking_enhanced.csv', index=False)
    print(f"\n✅ Ranking CSV saved → macro_2027_ranking_enhanced.csv")

    # Signal summary
    total_buy = len(df[df['Signal'] == 'BUY'])
    total_hold = len(df[df['Signal'] == 'HOLD'])
    total_sell = len(df[df['Signal'] == 'SELL'])
    total_trim = len(df[df['Signal'] == 'TRIM'])

    print(f"\n📊 Trading Signal Summary:")
    print(f"   • Total stocks analyzed: {len(df)}")
    print(f"   • 🟢 BUY: {total_buy} ({total_buy / len(df) * 100:.1f}%)")
    print(f"   • 🟡 HOLD: {total_hold} ({total_hold / len(df) * 100:.1f}%)")
    print(f"   • 🟠 TRIM: {total_trim} ({total_trim / len(df) * 100:.1f}%)")
    print(f"   • 🔴 SELL: {total_sell} ({total_sell / len(df) * 100:.1f}%)")

    # Additional insights
    print(f"\n📈 Momentum Insights:")
    buy_stocks = df[df['Signal'] == 'BUY']
    if len(buy_stocks) > 0:
        print(f"   • BUY stocks avg R²: {buy_stocks['R2_1M'].mean():.3f}")
        print(f"   • BUY stocks avg 3M slope: {buy_stocks['Slope_3M_Avg'].mean():.3f}%")

    trim_stocks = df[df['Signal'] == 'TRIM']
    if len(trim_stocks) > 0:
        print(f"   • TRIM stocks avg zscore: {trim_stocks['Slope_ZScore'].mean():.2f}")
        print(f"   • TRIM stocks (overextended momentum): {list(trim_stocks['Ticker'].head(5))}")

    return df


if __name__ == "__main__":
    infer_today()