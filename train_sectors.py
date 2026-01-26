import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
import warnings
from functools import lru_cache
import hashlib
import os

# ---------------- CONFIG ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
MODEL_FILE = "sector_model.joblib"
START_DATE = "2000-01-01"
CACHE_DIR = ".cache"
warnings.filterwarnings('ignore')


# ---------------- CACHING UTILITIES ----------------
def _get_cache_key(tickers, start_date):
    """Generate cache key from tickers and start date"""
    ticker_str = ','.join(sorted(tickers))
    key_str = f"{ticker_str}_{start_date}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_key):
    """Get cache file path"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"data_{cache_key}.pkl")


def _load_from_cache(cache_key):
    """Load data from cache if it exists and is valid"""
    cache_path = _get_cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            print(f"📦 Loading data from cache...")
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"⚠️ Cache read failed: {e}")
            return None
    return None


def _save_to_cache(data, cache_key):
    """Save data to cache"""
    cache_path = _get_cache_path(cache_key)
    try:
        data.to_pickle(cache_path)
        print(f"💾 Data cached for future use")
    except Exception as e:
        print(f"⚠️ Cache write failed: {e}")


# ---------------- DATA PREP (OPTIMIZED) ----------------
def get_macro_data():
    """Download all tickers in single batch call with caching"""
    print("📥 Gathering Macro & Sector Data...")

    # All tickers in one list for batch download
    tickers = SECTORS + ['SPY', 'QQQ', '^VIX', '^MOVE', '^TNX', '^TYX', 'HYG', 'DX-Y.NYB']

    # Check cache first
    cache_key = _get_cache_key(tickers, START_DATE)
    cached_data = _load_from_cache(cache_key)
    if cached_data is not None:
        return cached_data

    # Download all tickers at once (single API call)
    print(f"🌐 Downloading {len(tickers)} tickers in batch...")
    data = yf.download(
        tickers,
        start=START_DATE,
        progress=False,
        auto_adjust=True,
        group_by='column',  # Efficient grouping
        threads=True  # Parallel downloads
    )['Close']

    # Fill missing values once
    data = data.ffill().bfill()

    # Cache for next time
    _save_to_cache(data, cache_key)

    return data


# ---------------- FEATURE COMPUTATION (OPTIMIZED) ----------------
def compute_features(df):
    """Vectorized feature computation where possible"""
    X = pd.DataFrame(index=df.index)
    window = 21
    x_axis = np.arange(window)

    # Pre-compute log prices for all sectors at once
    log_prices = np.log(df[SECTORS])

    # Vectorized slope and R² calculation
    def calc_linregress_metrics(series):
        """Calculate slope and R² for a series"""
        slope = series.rolling(window).apply(
            lambda x: stats.linregress(x_axis, x)[0] if not np.isnan(x).any() else 0,
            raw=False
        )
        r2 = series.rolling(window).apply(
            lambda x: stats.linregress(x_axis, x)[2] ** 2 if not np.isnan(x).any() else 0,
            raw=False
        )
        return slope, r2

    # Process all sectors
    for s in SECTORS:
        # Linear regression features
        X[f'{s}_Slope'], X[f'{s}_R2'] = calc_linregress_metrics(log_prices[s])

        # Risk-adjusted momentum (vectorized)
        pct_change = df[s].pct_change(window)
        vol = df[s].pct_change().rolling(window).std()
        X[f'{s}_Risk_Adj_Mom'] = pct_change / (vol + 1e-6)

    # Macro features (vectorized operations)
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['VIX_Level'] = df['^VIX']
    X['MOVE_Level'] = df['^MOVE']
    X['DXY_Mom'] = df['DX-Y.NYB'].pct_change(window)
    X['Sector'] = 0

    return X.fillna(0)


# ---------------- REGIME-DEPENDENT TARGET (OPTIMIZED) ----------------
def get_regime_target(df):
    """Vectorized regime target calculation"""
    # Calculate relative returns for all sectors at once
    sector_returns = df[SECTORS].pct_change(21)
    spy_returns = df['SPY'].pct_change(21)

    # Vectorized relative returns
    rel_returns = sector_returns.sub(spy_returns, axis=0).shift(-21)

    # Vectorized regime detection
    is_stress = (df['^MOVE'] > 110) | (df['^VIX'] > 20)

    # Vectorized target assignment
    target = pd.Series(index=df.index, dtype='object')

    # Use numpy for faster indexing
    stress_mask = is_stress.values
    for i in range(len(df)):
        if stress_mask[i]:
            target.iloc[i] = rel_returns.iloc[i].idxmin()  # Stress: Mean Reversion
        else:
            target.iloc[i] = rel_returns.iloc[i].idxmax()  # Calm: Trend Continuation

    return target


# ---------------- EXECUTION ----------------
print("🚀 Starting optimized training pipeline...")

# Step 1: Load data (cached or downloaded)
df = get_macro_data()
print(f"✅ Data loaded: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

# Step 2: Compute features
print("⚙️ Computing features...")
X_full = compute_features(df)
print(f"✅ Features computed: {len(X_full.columns)} features")

# Step 3: Generate targets
print("🎯 Generating regime-dependent targets...")
y = get_regime_target(df)
print(f"✅ Targets generated: {len(y)} samples")

# Step 4: Train/test split
X = X_full.iloc[252:-21]
y = y.iloc[252:-21]
split = int(len(X) * 0.8)

print(f"📊 Training samples: {split}, Test samples: {len(X) - split}")

# Step 5: Scale and train
print("🔧 Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X.iloc[:split])
X_test = scaler.transform(X.iloc[split:])
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("🤖 Training GradientBoosting model...")
model = GradientBoostingClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.01,
    subsample=0.7,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model training complete")

# ---------------- DASHBOARD ----------------
latest = X_full.tail(1).copy()
latest_scaled = scaler.transform(latest)
probs = model.predict_proba(latest_scaled)[0]
top_idx = np.argsort(probs)[-3:][::-1]

# Identify current regime for printout
current_move = df['^MOVE'].iloc[-1]
current_vix = df['^VIX'].iloc[-1]
regime_mode = "STRESS (Mean Reversion)" if (current_move > 110 or current_vix > 30) else "NORMAL (Trend Continuation)"

print(f"\n{'=' * 60}")
print(f"### Strategic Analysis & Sector Picks [Regime: {regime_mode}]")
print(f"{'=' * 60}")
for i, idx in enumerate(top_idx):
    print(f"{i + 1}. **{model.classes_[idx]}** ({probs[idx] * 100:.1f}% confidence)")

# ---------------- HISTORICAL ODDS ----------------
hist_top_counts = y.value_counts()
hist_odds = hist_top_counts / len(y)

print(f"\n📈 Historical Sector Selection Frequency:")
print(hist_top_counts)

print(f"\n{'=' * 60}")
print("📊 Historical vs Predicted Odds:")
print(f"{'=' * 60}")
for s in SECTORS:
    pred_prob = probs[list(model.classes_).index(s)] if s in model.classes_ else 0.0
    hist_prob = hist_odds.get(s, 0.0)
    diff = (pred_prob - hist_prob) * 100
    arrow = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
    print(f"{s:<6} Predicted: {pred_prob * 100:5.1f}%  | Historical: {hist_prob * 100:5.1f}%  {arrow} ({diff:+.1f}%)")


# ---------------- EVALUATION ----------------
def top_n_accuracy(model, X_test, y_test, n=3):
    """Vectorized accuracy calculation"""
    probas = model.predict_proba(X_test)
    top_n_preds = np.argsort(probas, axis=1)[:, -n:]

    classes = list(model.classes_)
    y_test_vals = y_test.values

    # Vectorized top-1 accuracy
    top1_preds = [classes[top_n_preds[i, -1]] for i in range(len(y_test))]
    top1_acc = np.mean([top1_preds[i] == y_test_vals[i] for i in range(len(y_test))])

    # Vectorized top-2 accuracy
    top2_acc = np.mean([y_test_vals[i] in [classes[idx] for idx in top_n_preds[i, -2:]]
                        for i in range(len(y_test))])

    # Vectorized top-3 accuracy
    top3_acc = np.mean([y_test_vals[i] in [classes[idx] for idx in top_n_preds[i]]
                        for i in range(len(y_test))])

    return top1_acc, top2_acc, top3_acc


print(f"\n{'=' * 60}")
print("📈 MODEL PERFORMANCE METRICS")
print(f"{'=' * 60}")
top1, top2, top3 = top_n_accuracy(model, X_test, y_test)
print(f"✅ Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"✅ Top-2 Accuracy: {top2 * 100:.2f}%")
print(f"✅ Top-3 Accuracy: {top3 * 100:.2f}%")

# ---------------- FEATURE IMPORTANCE ----------------
feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\n{'=' * 60}")
print("🔑 Top 10 Feature Rankings:")
print(f"{'=' * 60}")
for f, v in feat_importance.head(10).items():
    print(f"{f:<30} {v:.4f}")

# ---------------- SAVE MODEL ----------------
print(f"\n{'=' * 60}")
print("💾 Saving model bundle...")
joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': X.columns.tolist()
}, MODEL_FILE)

print(f"✅ Model saved: {MODEL_FILE}")
print(f"✅ Features included: {len(X.columns)}")
print(f"✅ Classes: {len(model.classes_)}")
print(f"{'=' * 60}\n")