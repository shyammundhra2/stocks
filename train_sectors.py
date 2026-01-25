import yfinance as yf
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings

# ---------------- CONFIG ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
START_DATE = "2000-01-01"
MODEL_FILE = "sector_model.joblib"

warnings.filterwarnings('ignore')

# ---------------- DATA DOWNLOAD ----------------
print("📥 Downloading sector & macro data...")
tickers = SECTORS + ['SPY', 'RSP', '^VIX', '^MOVE', '^TNX', '^TYX']
raw = yf.download(tickers, start=START_DATE, progress=False, auto_adjust=True)['Close']
raw = raw.ffill().bfill()

print(f"✅ Data loaded: {len(raw)} trading days")

# ---------------- FEATURE ENGINEERING ----------------
def compute_features(df):
    X = pd.DataFrame(index=df.index)
    horizon_q = 63  # 3 months

    # --- Macro features ---
    if '^VIX' in df.columns:
        X['VIX_level'] = df['^VIX'].rolling(21).mean()
    if '^TNX' in df.columns and '^TYX' in df.columns:
        X['Yield_Curve'] = df['^TYX'] - df['^TNX']
        X['TNX_vol'] = df['^TNX'].rolling(horizon_q).std()

    # --- Sector momentum ---
    for s in SECTORS:
        if s in df.columns:
            X[f'{s}_3M_Mom'] = df[s].pct_change(horizon_q)

    # Fill missing
    X = X.fillna(0)
    return X

X = compute_features(raw)

# ---------------- TARGET ----------------
# Next month return per sector
horizon_target = 21  # 1 month
y = pd.DataFrame(index=raw.index, columns=SECTORS)
for s in SECTORS:
    if s in raw.columns:
        y[s] = raw[s].pct_change(horizon_target).shift(-horizon_target)

# Assign class per row: sector with highest next-month return
y_class = y.idxmax(axis=1)
y_class = y_class.fillna(method='ffill')

# Align features and target
X, y_class = X.iloc[:-horizon_target], y_class.iloc[:-horizon_target]

# ---------------- TRAIN-TEST SPLIT ----------------
split_date = pd.Timestamp("2020-09-23")
X_train = X[X.index <= split_date]
X_test = X[X.index > split_date]
y_train = y_class[X_train.index]
y_test = y_class[X_test.index]

print(f"📊 Train samples: {len(X_train)}")
print(f"📊 Test samples : {len(X_test)}")
print(f"📅 Split date   : {split_date.date()}")

# ---------------- MODEL ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------------- EVALUATION ----------------
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print("\n============================================================")
print(f"📈 Test Accuracy: {test_acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=SECTORS, digits=2))

# ---------------- SAVE MODEL ----------------
joblib.dump({'model': model, 'scaler': scaler, 'features': X_train.columns.tolist()}, MODEL_FILE)
print(f"💾 Model saved: {MODEL_FILE}")
print("✅ Training complete")
