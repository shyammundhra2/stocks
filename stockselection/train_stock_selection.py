import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime
import os

# ---------------- CONFIG ----------------
START, END = "2001-01-01", datetime.now().strftime('%Y-%m-%d')
LOCAL_FILE = "sp500.csv"
MODEL_NAME = "macro_model_2027.json"
FEATURE_RANK_FILE = "feature_rank_2027.csv"

MACRO_TICKERS = ['SPY','RSP','^VIX','^MOVE','DX-Y.NYB',
                 '^IRX','^TNX','^TYX','HYG','LQD']

SECTOR_TICKERS = ['XLK','XLF','XLI','XLY','XLE','XLV',
                  'XLP','XLU','XLB','XLRE','XLC']

# ---------------- TRAIN ----------------
def train_model():
    symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()
    # Remove top-100 restriction; keeps enough rows for training
    train_pool = symbols

    print("Downloading macro + sector data...")
    ms = yf.download(MACRO_TICKERS + SECTOR_TICKERS,
                     start=START, end=END,
                     auto_adjust=True)['Close']

    m = pd.DataFrame(index=ms.index)
    m['Curve_Inversion'] = ms['^IRX'] - ms['^TNX']
    m['Bond_Stress'] = ms['^MOVE']
    m['Liquidity'] = ms['HYG'] / ms['LQD']
    m['Concentration'] = ms['SPY'] / ms['RSP']

    for s in SECTOR_TICKERS:
        m[f'{s}_3M'] = ms[s].pct_change(63)

    # shift to avoid look-ahead bias
    m = m.shift(1)

    all_rows = []

    print("Building stock-level dataset...")
    for t in train_pool:
        s_raw = yf.download(t, start=START, end=END,
                            auto_adjust=True, progress=False)
        if s_raw.empty or len(s_raw) < 300:
            continue

        s = pd.DataFrame(index=s_raw.index)
        s['Stock_Mom_3M'] = s_raw['Close'].pct_change(63)
        s['Target_2027'] = s_raw['Close'].pct_change(252).shift(-252)

        joined = s.join(m).dropna()
        if joined.empty:
            continue

        joined['Ticker'] = t
        all_rows.append(joined)

    if not all_rows:
        raise ValueError("No data available for training. Check your CSV or Yahoo downloads.")

    df_ml = pd.concat(all_rows)

    feats = ['Curve_Inversion','Bond_Stress','Liquidity',
             'Concentration','Stock_Mom_3M'] + [f'{s}_3M' for s in SECTOR_TICKERS]

    # Cross-sectional label per date
    y = df_ml.groupby(df_ml.index)['Target_2027'] \
             .transform(lambda x: x > x.median()).astype(int)
    X = df_ml[feats]

    # Ensure dataset is not empty
    valid_idx = X.index.intersection(y.dropna().index)
    X_train, y_train = X.loc[valid_idx], y.loc[valid_idx]

    print(f"Training on {len(X_train)} rows...")
    print("Label distribution:\n", y_train.value_counts())

    # ---------------- TRAIN MODEL ----------------
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=0.1,      # allows small cross-sections
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="approx",       # less conservative than hist
        base_score=0.5
    )

    model.fit(X_train, y_train)

    # ---------------- SAVE MODEL ----------------
    model.save_model(MODEL_NAME)
    print(f"Model saved → {MODEL_NAME}")

    # ---------------- FEATURE IMPORTANCE ----------------
    imp = pd.DataFrame({
        "Feature": feats,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    imp.to_csv(FEATURE_RANK_FILE, index=False)

    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (2027 Drivers)")
    print("="*60)
    for _, r in imp.iterrows():
        bar = "█" * int(r.Importance * 100)
        print(f"{r.Feature:<18} | {bar} {r.Importance:.2%}")
    print("="*60)

if __name__ == "__main__":
    train_model()
