import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from macro.helpers import compute_RSI, compute_ATR

# 1️⃣ Setup
trend_tickers = ['VGT','VDE','VIS','XME','GLD','IBIT','TLT', 'SPY', 'QQQ', 'XLV', 'XLF']
FEATURE_COLS = ['RSI14', 'ATR14', 'SMA50_dist', 'SMA200_dist', 'mom_5d', 'mom_10d', 'vol_z']

all_features = []
all_targets_fast = []
all_targets_slow = []

for ticker in trend_tickers:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    if df.empty or len(df) < 250: continue

    close = df['Close'].squeeze()
    vol = df['Volume'].squeeze()

    # --- Feature Engineering ---
    ticker_df = pd.DataFrame(index=df.index)
    ticker_df['RSI14'] = compute_RSI(close, 14)
    ticker_df['ATR14'] = compute_ATR(df, 14)
    
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ticker_df['SMA50_dist'] = (close - sma50) / sma50
    ticker_df['SMA200_dist'] = (close - sma200) / sma200
    
    ticker_df['mom_5d'] = close.pct_change(5)
    ticker_df['mom_10d'] = close.pct_change(10)
    ticker_df['vol_z'] = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-6)

    # --- Dual Targets ---
    target_fast = (close.shift(-5) > close).astype(int)   # 1-Week
    target_slow = (close.shift(-21) > close).astype(int)  # 1-Month

    ticker_df['target_f'] = target_fast
    ticker_df['target_s'] = target_slow
    ticker_df = ticker_df.dropna()

    all_features.append(ticker_df[FEATURE_COLS])
    all_targets_fast.append(ticker_df['target_f'])
    all_targets_slow.append(ticker_df['target_s'])

# 2️⃣ Combine & Scale
X = pd.concat(all_features)
y_f = pd.concat(all_targets_fast)
y_s = pd.concat(all_targets_slow)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3️⃣ Train Dual Models
print("Training Fast Model (5d)...")
model_fast = RandomForestClassifier(n_estimators=500, max_depth=4, class_weight='balanced', random_state=42)
model_fast.fit(X_scaled, y_f)

print("Training Slow Model (21d)...")
model_slow = RandomForestClassifier(n_estimators=500, max_depth=5, class_weight='balanced', random_state=42)
model_slow.fit(X_scaled, y_s)

# 4️⃣ Save Bundle
joblib.dump({
    'model_fast': model_fast,
    'model_slow': model_slow,
    'scaler': scaler,
    'features': FEATURE_COLS
}, 'trend_model.joblib')

print("✅ Dual-Horizon Models Saved.")
