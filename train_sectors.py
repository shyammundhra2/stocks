import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
import warnings

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
    window = 21
    x_axis = np.arange(window)

    for s in SECTORS:
        log_price = np.log(df[s])
        # Linear Regression Slope and R2 for all sectors
        X[f'{s}_Slope'] = log_price.rolling(window).apply(
            lambda x: stats.linregress(x_axis, x)[0] if not np.isnan(x).any() else 0
        )
        X[f'{s}_R2'] = log_price.rolling(window).apply(
            lambda x: stats.linregress(x_axis, x)[2] ** 2 if not np.isnan(x).any() else 0
        )
        vol = df[s].pct_change().rolling(21).std()
        X[f'{s}_Risk_Adj_Mom'] = df[s].pct_change(21) / (vol + 1e-6)

    # Macro Context
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['VIX_Level'] = df['^VIX']
    X['MOVE_Level'] = df['^MOVE']
    X['DXY_Mom'] = df['DX-Y.NYB'].pct_change(21)
    X['Sector'] = 0
    return X.fillna(0)


# ---------------- REGIME-DEPENDENT TARGET ----------------
def get_regime_target(df):
    rel_returns = df[SECTORS].pct_change(21).sub(df['SPY'].pct_change(21), axis=0).shift(-21)
    # Regime Trigger: Bond Vol (MOVE) > 110 or Equity Vol (VIX) > 20
    is_stress = (df['^MOVE'] > 110) | (df['^VIX'] > 20)

    target = pd.Series(index=df.index, dtype='object')
    for i in range(len(df)):
        if is_stress.iloc[i]:
            target.iloc[i] = rel_returns.iloc[i].idxmin()  # Stress: Predict Mean Reversion
        else:
            target.iloc[i] = rel_returns.iloc[i].idxmax()  # Calm: Predict Trend Continuation
    return target


# ---------------- EXECUTION ----------------
df = get_macro_data()
X_full = compute_features(df)
y = get_regime_target(df)

X = X_full.iloc[252:-21]
y = y.iloc[252:-21]
split = int(len(X) * 0.8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X.iloc[:split])
X_test = scaler.transform(X.iloc[split:])
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = GradientBoostingClassifier(
    n_estimators=250, max_depth=4, learning_rate=0.01,
    subsample=0.7, max_features='sqrt', random_state=42
)
model.fit(X_train, y_train)

# ---------------- DASHBOARD ----------------
latest = X_full.tail(1).copy()
latest_scaled = scaler.transform(latest)
probs = model.predict_proba(latest_scaled)[0]
top_idx = np.argsort(probs)[-3:][::-1]

# Identify current regime for printout
current_move = df['^MOVE'].iloc[-1]
current_vix = df['^VIX'].iloc[-1]
regime_mode = "STRESS (Mean Reversion)" if (current_move > 110 or current_vix > 20) else "NORMAL (Trend Continuation)"

print(f"\n### Strategic Analysis & Sector Picks [Regime: {regime_mode}]")
for i, idx in enumerate(top_idx):
    print(f"{i + 1}. **{model.classes_[idx]}** ({probs[idx] * 100:.1f}% confidence)")

# ---------------- HISTORICAL ODDS ----------------
hist_top_counts = y.value_counts()
hist_odds = hist_top_counts / len(y)
print(y.value_counts())

print("\n📊 Historical vs Predicted Odds:")
for s in SECTORS:
    pred_prob = probs[list(model.classes_).index(s)] if s in model.classes_ else 0.0
    hist_prob = hist_odds.get(s, 0.0)
    print(f"{s:<6} Predicted: {pred_prob * 100:5.1f}%  | Historical: {hist_prob * 100:5.1f}%")


# ---------------- EVALUATION ----------------
def top_n_accuracy(model, X_test, y_test, n=3):
    probas = model.predict_proba(X_test)
    top_n_preds = np.argsort(probas, axis=1)[:, -n:]
    top1_list = [list(model.classes_)[top_n_preds[i, -1]] == y_test.iloc[i] for i in range(len(y_test))]
    top2_list = [y_test.iloc[i] in [list(model.classes_)[idx] for idx in top_n_preds[i, -2:]] for i in
                 range(len(y_test))]
    top3_list = [y_test.iloc[i] in [list(model.classes_)[idx] for idx in top_n_preds[i]] for i in range(len(y_test))]
    return np.mean(top1_list), np.mean(top2_list), np.mean(top3_list)


top1, top2, top3 = top_n_accuracy(model, X_test, y_test)
print(f"\n✅ Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"✅ Top-2 Accuracy: {top2 * 100:.2f}%")
print(f"✅ Top-3 Accuracy: {top3 * 100:.2f}%")

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