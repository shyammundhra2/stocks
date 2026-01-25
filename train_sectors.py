import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import warnings

# ---------------- CONFIG & SAVED PREFERENCES ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
MODEL_FILE = "sector_model.joblib"  # Per [2026-01-25] preference
START_DATE = "2000-01-01"
warnings.filterwarnings('ignore')


# ---------------- DATA PREP ----------------
def get_macro_data():
    print("📥 Gathering Macro & Sector Data...")
    # Added QQQ to ensure compatibility with predict.py Tech Regime logic
    tickers = SECTORS + ['SPY', 'QQQ', '^VIX', '^MOVE', '^TNX', '^TYX', 'HYG', 'DX-Y.NYB']
    data = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=True)['Close']
    return data.ffill().bfill()


def compute_features(df):
    X = pd.DataFrame(index=df.index)

    # 1. Macro & Risk (Aligned with predict.py naming)
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['Curve_Momentum'] = X['Yield_Curve'].diff(21)
    X['VIX_Level'] = df['^VIX']
    X['MOVE_Level'] = df['^MOVE']
    X['DXY_Mom'] = df['DX-Y.NYB'].pct_change(21)
    X['Market_Regime_3M'] = df['SPY'].pct_change(63)
    X['Tech_Regime_1M'] = df['QQQ'].pct_change(21) - df['SPY'].pct_change(21)

    # 2. Sector Strength (Last 3 Months - Per [2026-01-12] instruction)
    for s in SECTORS:
        X[f'{s}_Rel_Mom_1M'] = df[s].pct_change(21) - df['SPY'].pct_change(21)
        X[f'{s}_Rel_Mom_3M'] = df[s].pct_change(63) - df['SPY'].pct_change(63)

        # Risk Adjusted Momentum
        vol = df[s].pct_change().rolling(21).std()
        X[f'{s}_Risk_Adj_Mom'] = df[s].pct_change(21) / (vol + 1e-6)

        # New v2 Indicators
        from macro.helpers import compute_RSI  # Ensure this is accessible
        X[f'{s}_RSI'] = compute_RSI(df[s], 14)
        X[f'{s}_Above_MA50'] = (df[s] > df[s].rolling(50).mean()).astype(int)

        high_52w = df[s].rolling(252).max()
        X[f'{s}_Drawdown_High'] = (df[s] - high_52w) / high_52w

    return X.fillna(0)


# ---------------- EXECUTION ----------------
df = get_macro_data()
X_full = compute_features(df)

# Target for Training
returns = df[SECTORS].pct_change(21).shift(-21)
y = returns.idxmax(axis=1)

# Align and Split
X = X_full.iloc[252:-21]  # Increased warm-up for 52W high
y = y.iloc[252:-21]
split = int(len(X) * 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X.iloc[:split])
X_test = scaler.transform(X.iloc[split:])
y_train, y_test = y.iloc[:split], y.iloc[split:]

# The Optimized Model
model = GradientBoostingClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.01,
    subsample=0.7, max_features='sqrt', random_state=42
)
model.fit(X_train, y_train)

# ---------------- GENERATE DASHBOARD ----------------
latest_scaled = scaler.transform(X_full.tail(1))
probs = model.predict_proba(latest_scaled)[0]
top_idx = np.argsort(probs)[-3:][::-1]

# Sector Strength (Based on 3M per your rule)
sector_3m_perf = df[SECTORS].pct_change(63).iloc[-1].sort_values(ascending=False)

print(f"\n### Strategic Analysis & Sector Picks")
print(f"**Top 3 Predicted Sectors (High Confidence):**")
for i, idx in enumerate(top_idx):
    print(f"{i + 1}. **{model.classes_[idx]}** ({probs[idx] * 100:.1f}% confidence)")

# ---------------- SAVE (CRITICAL FIX) ----------------
# Added 'features' key so predict.py can load definitions
joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': X.columns.tolist()
}, MODEL_FILE)

print(f"\n💾 Model updated with feature definitions: {MODEL_FILE}")