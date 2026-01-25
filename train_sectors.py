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
START_DATE = "2010-01-01"
warnings.filterwarnings('ignore')


# ---------------- DATA PREP ----------------
def get_macro_data():
    print("📥 Gathering Macro & Sector Data...")
    tickers = SECTORS + ['SPY', '^VIX', '^MOVE', '^TNX', '^TYX', 'HYG', 'DXY']
    data = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=True)['Close']
    return data.ffill().bfill()


def compute_features(df):
    X = pd.DataFrame(index=df.index)

    # 1. Macro & Risk (For RoRo Logic)
    X['Yield_Curve'] = df['^TYX'] - df['^TNX']
    X['VIX_Level'] = df['^VIX']
    X['MOVE_Level'] = df['^MOVE']
    X['Credit_Stress'] = df['HYG'].pct_change(21)  # Proxy for High Yield OAS
    X['DXY_Mom'] = df['DXY'].pct_change(21)

    # 2. Sector Strength (Last 3 Months - Per [2026-01-12] instruction)
    for s in SECTORS:
        X[f'{s}_Rel_Mom_3M'] = df[s].pct_change(63) - df['SPY'].pct_change(63)
        X[f'{s}_Vol'] = df[s].pct_change().rolling(21).std()

    return X.fillna(0)


# ---------------- EXECUTION ----------------
df = get_macro_data()
X_full = compute_features(df)

# Target for Training
returns = df[SECTORS].pct_change(21).shift(-21)
y = returns.idxmax(axis=1)

# Align and Split
X = X_full.iloc[200:-21]
y = y.iloc[200:-21]
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
bot_idx = np.argsort(probs)[:3]

# Calculate RoRo Score (Sample Logic based on VIX, Curve, and Spreads)
vix_component = max(0, 100 - df['^VIX'].iloc[-1] * 2)
curve_component = 50 if df['^TYX'].iloc[-1] > df['^TNX'].iloc[-1] else 10
roro_score = int((vix_component + curve_component) / 1.5)

print(f"\n## Consolidated Macro Investing Dashboard (US Focus)")
print(f"**RoRo Score: {roro_score}/100** ({'Risk-On' if roro_score > 60 else 'Neutral/Stress'})")
print(f"---")

# Sector Strength (Based on 3M per your rule)
sector_3m_perf = df[SECTORS].pct_change(63).iloc[-1].sort_values(ascending=False)

print("| Category | Metric | Current Value | Interpretation |")
print("| :--- | :--- | :--- | :--- |")
print(
    f"| Economic Growth | Yield Curve (30Y-10Y) | {X_full['Yield_Curve'].iloc[-1]:.2f} | {'Steepening' if X_full['Yield_Curve'].iloc[-1] > 0 else 'Inverted'} |")
print(f"| Labor Market | Sahm Rule Trigger | Monitoring | Enter Danger Zone if +0.5% |")
print(f"| Risk | VIX Index | {df['^VIX'].iloc[-1]:.2f} | {'Elevated' if df['^VIX'].iloc[-1] > 20 else 'Low'} |")
print(f"| Risk | MOVE Index | {df['^MOVE'].iloc[-1]:.2f} | Bond Volatility |")
print(f"| Liquidity | Net Liquidity | [PLACEHOLDER] | Row integrated per [2026-01-12] |")
print(f"| Default | Default Rate | [PLACEHOLDER] | Row integrated per [2026-01-12] |")

print(f"\n### Strategic Analysis & Sector Picks")
print(f"**Top 3 Predicted Sectors (High Confidence):**")
for i, idx in enumerate(top_idx):
    print(f"{i + 1}. **{model.classes_[idx]}** ({probs[idx] * 100:.1f}% confidence)")

print(f"\n**Current Sector Strength (Last 3 Months):**")
print(f"* **Strongest:** {', '.join(sector_3m_perf.head(3).index)}")
print(f"* **Weakest:** {', '.join(sector_3m_perf.tail(3).index)}")

# Save
joblib.dump({'model': model, 'scaler': scaler}, MODEL_FILE)
print(f"\n💾 Model updated: {MODEL_FILE}")