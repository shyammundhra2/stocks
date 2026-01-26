import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from macro.helpers import compute_RSI

# ---------------- CONFIG ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
MODEL_FILE = "sector_model.joblib"
START_DATE = "2000-01-01"
warnings.filterwarnings('ignore')


# ---------------- DATA PREP ----------------
def get_macro_data():
    print("📥 Gathering Macro & Sector Data...")
    tickers = SECTORS + ['SPY', 'QQQ', '^VIX', '^MOVE', '^TNX', '^TYX', 'HYG', 'DX-Y.NYB']
    data = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=True)['Close']
    return data.ffill().bfill()


def compute_features(df):
    X = pd.DataFrame(index=df.index)

    # --- Macro & Risk Features ---
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['Curve_Momentum'] = X['Yield_Curve'].diff(21)
    X['VIX_Level'] = df['^VIX']
    X['MOVE_Level'] = df['^MOVE']
    X['DXY_Mom'] = df['DX-Y.NYB'].pct_change(21)
    X['Market_Regime_3M'] = df['SPY'].pct_change(63)
    X['Tech_Regime_1M'] = df['QQQ'].pct_change(21) - df['SPY'].pct_change(21)

    # --- Sector Features for Mean Reversion ---
    for s in SECTORS:
        X[f'{s}_Rel_Mom_1M'] = df[s].pct_change(21) - df['SPY'].pct_change(21)
        X[f'{s}_Rel_Mom_3M'] = df[s].pct_change(63) - df['SPY'].pct_change(63)

        vol = df[s].pct_change().rolling(21).std()
        X[f'{s}_Risk_Adj_Mom'] = df[s].pct_change(21) / (vol + 1e-6)
        X[f'{s}_RSI'] = compute_RSI(df[s], 14)
        X[f'{s}_Above_MA50'] = (df[s] > df[s].rolling(50).mean()).astype(int)
        high_52w = df[s].rolling(252).max()
        X[f'{s}_Drawdown_High'] = (df[s] - high_52w) / high_52w

    # Dummy column to match training features for dashboard
    X['Sector'] = 0

    return X.fillna(0)


# ---------------- EXECUTION ----------------
df = get_macro_data()
X_full = compute_features(df)

# Target: next 1-month **mean-reversion returns** (lowest relative to SPY)
returns = df[SECTORS].pct_change(21).shift(-21)
y = returns.sub(df['SPY'].pct_change(21), axis=0).idxmin(axis=1)  # min relative => mean-reversion

# Align for rolling windows
X = X_full.iloc[252:-21]
y = y.iloc[252:-21]
split = int(len(X) * 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X.iloc[:split])
X_test = scaler.transform(X.iloc[split:])
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ---------------- MODEL ----------------
model = GradientBoostingClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.01,
    subsample=0.7, max_features='sqrt', random_state=42
)
model.fit(X_train, y_train)

# ---------------- DASHBOARD ----------------
latest = X_full.tail(1).copy()
latest['Sector'] = 0  # ensure matching feature
latest_scaled = scaler.transform(latest)
probs = model.predict_proba(latest_scaled)[0]
top_idx = np.argsort(probs)[-3:][::-1]

print("\n### Strategic Analysis & Sector Picks")
for i, idx in enumerate(top_idx):
    print(f"{i + 1}. **{model.classes_[idx]}** ({probs[idx]*100:.1f}% confidence)")

# ---------------- HISTORICAL ODDS ----------------
hist_top_counts = y.value_counts()
hist_odds = hist_top_counts / len(y)
print(y.value_counts())

print("\n📊 Historical vs Predicted Odds:")
for s in SECTORS:
    pred_prob = probs[list(model.classes_).index(s)] if s in model.classes_ else 0.0
    hist_prob = hist_odds.get(s, 0.0)
    print(f"{s:<6} Predicted: {pred_prob*100:5.1f}%  | Historical: {hist_prob*100:5.1f}%")

# ---------------- EVALUATION ----------------
def top_n_accuracy(model, X_test, y_test, n=3):
    probas = model.predict_proba(X_test)
    top_n_preds = np.argsort(probas, axis=1)[:, -n:]
    top1_list = [list(model.classes_)[top_n_preds[i, -1]] == y_test.iloc[i] for i in range(len(y_test))]
    top2_list = [y_test.iloc[i] in [list(model.classes_)[idx] for idx in top_n_preds[i, -2:]] for i in range(len(y_test))]
    top3_list = [y_test.iloc[i] in [list(model.classes_)[idx] for idx in top_n_preds[i]] for i in range(len(y_test))]
    return np.mean(top1_list), np.mean(top2_list), np.mean(top3_list)

top1, top2, top3 = top_n_accuracy(model, X_test, y_test)
print(f"\n✅ Top-1 Accuracy: {top1*100:.2f}%")
print(f"✅ Top-2 Accuracy: {top2*100:.2f}%")
print(f"✅ Top-3 Accuracy: {top3*100:.2f}%")

# ---------------- FEATURE IMPORTANCE ----------------
feat_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🔑 Top 10 Feature Rankings:")
for f, v in feat_importance.head(10).items():
    print(f"{f:<30} {v:.4f}")

# ---------------- SAVE MODEL ----------------
joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': X.columns.tolist()
}, MODEL_FILE)

print(f"\n💾 Model updated with feature definitions: {MODEL_FILE}")
