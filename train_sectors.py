import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# ---------------- CONFIG ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
START_DATE = "2000-01-01"  # Extended for better regime training
MODEL_FILE = "sector_model.joblib"

warnings.filterwarnings('ignore')

# ---------------- DATA DOWNLOAD ----------------
print("📥 Downloading sector & macro data...")
# Added SPY for relative strength and TLT for bond market regime
tickers = SECTORS + ['SPY', 'TLT', '^VIX', '^TNX', '^TYX']
raw = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=True)['Close']
raw = raw.ffill().bfill()


# ---------------- FEATURE ENGINEERING ----------------
def compute_enhanced_features(df):
    X = pd.DataFrame(index=df.index)
    h_short = 21  # 1 month
    h_med = 63  # 3 months

    # 1. Macro & Risk Regimes
    X['VIX_Level'] = df['^VIX'].rolling(h_short).mean()
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['Bond_Equity_Corr'] = df['SPY'].pct_change(5).rolling(h_med).corr(df['TLT'].pct_change(5))

    # 2. Sector Relative Strength (The Alpha Signal)
    spy_ret_med = df['SPY'].pct_change(h_med)
    spy_ret_short = df['SPY'].pct_change(h_short)

    for s in SECTORS:
        # Relative Momentum (Over/Underperformance vs SPY)
        X[f'{s}_Rel_Mom_3M'] = df[s].pct_change(h_med) - spy_ret_med
        X[f'{s}_Rel_Mom_1M'] = df[s].pct_change(h_short) - spy_ret_short

        # Volatility-Adjusted Momentum
        vol = df[s].pct_change().rolling(h_short).std() * np.sqrt(252)
        X[f'{s}_Sharpe'] = df[s].pct_change(h_med) / vol

        # Mean Reversion (Distance from 200DMA)
        ma200 = df[s].rolling(200).mean()
        X[f'{s}_DMA_200_Dist'] = (df[s] - ma200) / ma200

    return X.fillna(0)


print("🛠️  Engineering features...")
X = compute_enhanced_features(raw)

# ---------------- TARGET GENERATION ----------------
horizon_target = 21
returns = raw[SECTORS].pct_change(horizon_target).shift(-horizon_target)
y_class = returns.idxmax(axis=1)

# Alignment
X = X.iloc[200:-horizon_target]  # Drop warm-up and look-ahead
y_class = y_class.iloc[200:-horizon_target]

# ---------------- TRAIN-TEST SPLIT ----------------
# Using a fixed date to respect time-series integrity
split_date = pd.Timestamp("2021-01-01")
X_train, X_test = X[X.index <= split_date], X[X.index > split_date]
y_train, y_test = y_class[X_train.index], y_class[X_test.index]

# ---------------- MODEL TRAINING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduced max_depth and added subsampling to reduce overfitting
model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=3,  # Simpler trees are better for noisy financial data
    learning_rate=0.03,
    subsample=0.8,  # Stochastically select 80% of data for each tree
    random_state=42
)

print("🏋️  Training model...")
model.fit(X_train_scaled, y_train)


# ---------------- TOP 3 EVALUATION ----------------
def evaluate_top_n(model, X_val, y_val, n=3):
    probs = model.predict_proba(X_val)
    classes = model.classes_
    top_n_preds = [classes[np.argsort(p)[-n:]] for p in probs]

    hits = 0
    for actual, preds in zip(y_val, top_n_preds):
        if actual in preds:
            hits += 1
    return hits / len(y_val)


top3_acc = evaluate_top_n(model, X_test_scaled, y_test, n=3)
hit_acc = accuracy_score(y_test, model.predict(X_test_scaled))

print("\n================ PERFORMANCE ANALYSIS ================")
print(f"🎯  Direct Accuracy (Top 1): {hit_acc * 100:.2f}%")
print(f"🏆  Top 3 Accuracy:          {top3_acc * 100:.2f}%")
print("======================================================")

# ---------------- SAVE & PREDICT ----------------
joblib.dump({'model': model, 'scaler': scaler, 'features': X_train.columns.tolist()}, MODEL_FILE)

# Get current top 3 picks for the portfolio
latest_data = X.tail(1)
latest_scaled = scaler.transform(latest_data)
latest_probs = model.predict_proba(latest_scaled)[0]
top_3_idx = np.argsort(latest_probs)[-3:][::-1]

print("\n🚀 CURRENT MACRO PICKS (TOP 3):")
for i, idx in enumerate(top_3_idx):
    print(f"{i + 1}. {model.classes_[idx]} (Confidence: {latest_probs[idx] * 100:.1f}%)")