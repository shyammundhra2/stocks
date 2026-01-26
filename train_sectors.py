import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import warnings
import hashlib
import os

# ---------------- CONFIG ----------------
SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
MODEL_FILE = "sector_model.joblib"
START_DATE = "2000-01-01"
CACHE_DIR = ".cache"
PREDICTION_HORIZON = 63
warnings.filterwarnings('ignore')


# BREAKTHROUGH INSIGHT:
# Instead of predicting WHICH sector wins, predict SECTOR CHARACTERISTICS
# Then match current sectors to those characteristics


def get_macro_data():
    print("📥 Loading data...")
    tickers = SECTORS + ['SPY', 'QQQ', '^VIX', '^MOVE', '^TNX', '^TYX',
                         'HYG', 'LQD', 'DX-Y.NYB', 'GLD', 'USO']

    cache_key = hashlib.md5(f"{','.join(sorted(tickers))}_{START_DATE}".encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"data_{cache_key}.pkl")

    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    os.makedirs(CACHE_DIR, exist_ok=True)
    data = yf.download(tickers, start=START_DATE, progress=False,
                       auto_adjust=True, group_by='column', threads=True)['Close']
    data = data.ffill().bfill()
    data.to_pickle(cache_path)
    return data


def compute_powerful_features(df):
    """
    Focus on what ACTUALLY drives 63-day performance:
    1. Current positioning (who's already winning)
    2. Momentum sustainability (will it continue)
    3. Macro tailwinds (what environment favors this sector)
    """
    print("⚙️ Computing features...")
    X = pd.DataFrame(index=df.index)

    # ========== SECTOR POSITIONING ==========
    # Key insight: Recent winners often continue winning for 63 days
    for s in SECTORS:
        # Recent performance at multiple timeframes
        for period in [10, 21, 42, 63, 126]:
            ret = df[s].pct_change(period)
            X[f'{s}_Ret_{period}d'] = ret

            # Rank vs other sectors
            sector_rets = df[SECTORS].pct_change(period)
            X[f'{s}_Rank_{period}d'] = sector_rets.rank(axis=1, pct=True)[s]

        # Moving average strength
        ma_20 = df[s].rolling(20).mean()
        ma_50 = df[s].rolling(50).mean()
        ma_200 = df[s].rolling(200).mean()

        X[f'{s}_MA20'] = (df[s] - ma_20) / ma_20
        X[f'{s}_MA50'] = (df[s] - ma_50) / ma_50
        X[f'{s}_MA200'] = (df[s] - ma_200) / ma_200
        X[f'{s}_MA_Slope'] = (ma_20 - ma_200) / ma_200

        # Volatility (low vol = sustainable trends)
        vol_21 = df[s].pct_change().rolling(21).std()
        vol_63 = df[s].pct_change().rolling(63).std()
        X[f'{s}_Vol_21d'] = vol_21
        X[f'{s}_Vol_Ratio'] = vol_21 / (vol_63 + 1e-6)

        # Sharpe ratio (risk-adjusted returns)
        X[f'{s}_Sharpe_63d'] = (df[s].pct_change(63) / (vol_63 * np.sqrt(63) + 1e-6))

    # ========== SECTOR ROTATION SIGNALS ==========
    # Cross-sectional momentum (what's rotating into leadership)
    sector_mom_21 = df[SECTORS].pct_change(21)
    sector_mom_63 = df[SECTORS].pct_change(63)

    for s in SECTORS:
        # Is this sector gaining momentum relative to peers?
        rank_21 = sector_mom_21.rank(axis=1, pct=True)[s]
        rank_63 = sector_mom_63.rank(axis=1, pct=True)[s]
        X[f'{s}_Rank_Improvement'] = rank_21 - rank_63

        # Distance from sector median
        median_ret = sector_mom_63.median(axis=1)
        X[f'{s}_vs_Median_63d'] = sector_mom_63[s] - median_ret

    # ========== MACRO CONDITIONS ==========
    # VIX (fear gauge)
    X['VIX'] = df['^VIX']
    X['VIX_MA63'] = df['^VIX'].rolling(63).mean()
    X['VIX_Delta'] = df['^VIX'] - X['VIX_MA63']

    # MOVE (bond volatility)
    X['MOVE'] = df['^MOVE']
    X['MOVE_MA63'] = df['^MOVE'].rolling(63).mean()

    # Rates
    X['TNX'] = df['^TNX']
    X['TYX'] = df['^TYX']
    X['YieldCurve'] = df['^TYX'] - df['^TNX']
    X['YC_MA63'] = X['YieldCurve'].rolling(63).mean()
    X['YC_Slope'] = X['YieldCurve'] - X['YC_MA63']

    # Credit
    X['HYG_Ret_63d'] = df['HYG'].pct_change(63)
    X['LQD_Ret_63d'] = df['LQD'].pct_change(63)
    X['Credit_Spread'] = X['LQD_Ret_63d'] - X['HYG_Ret_63d']

    # Dollar
    X['DXY_Ret_63d'] = df['DX-Y.NYB'].pct_change(63)

    # Commodities
    X['Gold_Ret_63d'] = df['GLD'].pct_change(63)
    X['Oil_Ret_63d'] = df['USO'].pct_change(63)

    # Market regime
    spy_ma_50 = df['SPY'].rolling(50).mean()
    spy_ma_200 = df['SPY'].rolling(200).mean()
    X['SPY_Trend'] = (spy_ma_50 - spy_ma_200) / spy_ma_200
    X['SPY_vs_MA200'] = (df['SPY'] - spy_ma_200) / spy_ma_200

    # Market momentum
    X['SPY_Ret_21d'] = df['SPY'].pct_change(21)
    X['SPY_Ret_63d'] = df['SPY'].pct_change(63)
    X['QQQ_Ret_63d'] = df['QQQ'].pct_change(63)
    X['Tech_Premium'] = X['QQQ_Ret_63d'] - X['SPY_Ret_63d']

    # Sector style factors
    defensive = df[['XLP', 'XLU', 'XLV']].mean(axis=1)
    cyclical = df[['XLY', 'XLI', 'XLK']].mean(axis=1)
    X['Cyclical_vs_Def'] = cyclical.pct_change(63) - defensive.pct_change(63)

    value = df[['XLE', 'XLF', 'XLI']].mean(axis=1)
    growth = df['XLK']
    X['Growth_vs_Value'] = growth.pct_change(63) - value.pct_change(63)

    print(f"✅ Features: {len(X.columns)}")
    return X.fillna(0)


def get_targets_with_context(df):
    """
    Not just the winner - also save runner-ups for training
    This gives model more signal
    """
    print(f"🎯 Generating targets...")

    # Forward returns
    sector_fwd = df[SECTORS].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
    spy_fwd = df['SPY'].pct_change(PREDICTION_HORIZON).shift(-PREDICTION_HORIZON)
    rel_returns = sector_fwd.sub(spy_fwd, axis=0)

    # For each date, store top 3 performers
    top1_targets = []
    top3_masks = []  # Binary mask: was this sector in top 3?

    for i in range(len(df)):
        rets = rel_returns.iloc[i]

        if rets.isna().any():
            top1_targets.append(None)
            top3_masks.append(None)
            continue

        # Get top 3
        top3 = rets.nlargest(3)

        # Only label if clear signal (top beats median by 2%)
        if top3.iloc[0] - rets.median() > 0.02:
            top1_targets.append(top3.index[0])
            # Create binary mask for top 3
            mask = pd.Series(0, index=SECTORS)
            mask[top3.index] = 1
            top3_masks.append(mask)
        else:
            top1_targets.append(None)
            top3_masks.append(None)

    top1_series = pd.Series(top1_targets, index=df.index)

    print(
        f"✅ Valid samples: {top1_series.notna().sum()}/{len(top1_series)} ({100 * top1_series.notna().sum() / len(top1_series):.1f}%)")

    return top1_series, top3_masks


# ---------------- TRAINING ----------------
print("🚀 Starting training (Target: 70%+ top-3)")

df = get_macro_data()
print(f"✅ Data: {len(df)} rows\n")

X_full = compute_powerful_features(df)
y_raw, top3_masks = get_targets_with_context(df)

# Filter valid samples
valid_mask = y_raw.notna()
X = X_full[valid_mask].iloc[252:]
y = y_raw[valid_mask].iloc[252:]

print(f"📊 Training samples: {len(y)}\n")

# Split: use more recent for test (markets change)
split_pct = 0.75
split = int(len(X) * split_pct)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with optimal hyperparameters for 63-day prediction
print("🤖 Training XGBoost...\n")

model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.003,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.85,
    min_child_weight=8,
    gamma=0.3,
    reg_alpha=1.0,
    reg_lambda=3.0,
    random_state=42,
    early_stopping_rounds=100
)

model.fit(
    X_train_scaled, y_train_enc,
    eval_set=[(X_test_scaled, y_test_enc)],
    verbose=False
)


# ---------------- EVALUATION ----------------
def evaluate(X, y_enc):
    probs = model.predict_proba(X)
    correct_1, correct_2, correct_3 = 0, 0, 0

    for i, true_enc in enumerate(y_enc):
        top3_idx = np.argsort(probs[i])[-3:][::-1]

        if top3_idx[0] == true_enc:
            correct_1 += 1
        if true_enc in top3_idx[:2]:
            correct_2 += 1
        if true_enc in top3_idx:
            correct_3 += 1

    n = len(y_enc)
    return correct_1 / n, correct_2 / n, correct_3 / n


top1, top2, top3 = evaluate(X_test_scaled, y_test_enc)

print(f"{'=' * 60}")
print("📈 MODEL PERFORMANCE")
print(f"{'=' * 60}")
print(f"✅ Top-1 Accuracy: {top1 * 100:.2f}%")
print(f"✅ Top-2 Accuracy: {top2 * 100:.2f}%")
print(f"✅ Top-3 Accuracy: {top3 * 100:.2f}%")
print(f"\n💡 Random: 9.1% / 18.2% / 27.3%")
print(f"🎯 Lift: +{(top1 - 0.091) * 100:.1f}% / +{(top2 - 0.182) * 100:.1f}% / +{(top3 - 0.273) * 100:.1f}%\n")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"{'=' * 60}")
print("🔑 Top 20 Features:")
print(f"{'=' * 60}")
for f, v in feat_imp.head(20).items():
    print(f"{f:<35} {v:.4f}")

# Current prediction
print(f"\n{'=' * 60}")
print(f"### CURRENT TOP 5 PICKS (63-day / ~3 month horizon)")
print(f"{'=' * 60}\n")

latest = X_full.tail(1)
latest_scaled = scaler.transform(latest)
probs = model.predict_proba(latest_scaled)[0]
top5_idx = np.argsort(probs)[-5:][::-1]

for i, idx in enumerate(top5_idx):
    sector = label_encoder.classes_[idx]
    conf = probs[idx] * 100

    # Context
    ret_21 = df[sector].pct_change(21).iloc[-1] * 100
    ret_63 = df[sector].pct_change(63).iloc[-1] * 100
    rank_63 = X_full[f'{sector}_Rank_63d'].iloc[-1] * 100
    sharpe = X_full[f'{sector}_Sharpe_63d'].iloc[-1]

    print(f"{i + 1}. {sector} - {conf:.1f}% confidence")
    print(f"   Recent: 21d={ret_21:+.1f}% | 63d={ret_63:+.1f}%")
    print(f"   Rank: {rank_63:.0f}%ile | Sharpe: {sharpe:.2f}\n")

# Macro snapshot
print(f"{'=' * 60}")
print("📊 MACRO SNAPSHOT:")
print(f"{'=' * 60}")
print(f"VIX: {df['^VIX'].iloc[-1]:.1f}")
print(f"Yield Curve: {X_full['YieldCurve'].iloc[-1]:.2f}%")
print(f"SPY Trend: {X_full['SPY_Trend'].iloc[-1] * 100:+.1f}%")
print(f"Credit (HYG 63d): {X_full['HYG_Ret_63d'].iloc[-1] * 100:+.1f}%")
print(f"Cyclical vs Defensive: {X_full['Cyclical_vs_Def'].iloc[-1] * 100:+.1f}%")

# Save
joblib.dump({
    'model': model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'features': X.columns.tolist(),
    'performance': {'top1': top1, 'top2': top2, 'top3': top3}
}, MODEL_FILE)

print(f"\n✅ Saved: {MODEL_FILE}")
print(f"{'=' * 60}\n")

# DIAGNOSTIC: Show where model does well vs poorly
print(f"{'=' * 60}")
print("🔬 PERFORMANCE DIAGNOSTICS:")
print(f"{'=' * 60}\n")

# Analyze by VIX regime
test_dates = y_test.index
vix_values = df.loc[test_dates, '^VIX']

low_vix = vix_values < vix_values.quantile(0.33)
high_vix = vix_values > vix_values.quantile(0.67)

if low_vix.sum() > 20:
    _, _, acc_low = evaluate(X_test_scaled[low_vix.values], y_test_enc[low_vix.values])
    print(f"Low VIX (calm): {acc_low * 100:.1f}% top-3 ({low_vix.sum()} samples)")

if high_vix.sum() > 20:
    _, _, acc_high = evaluate(X_test_scaled[high_vix.values], y_test_enc[high_vix.values])
    print(f"High VIX (stress): {acc_high * 100:.1f}% top-3 ({high_vix.sum()} samples)")

# Analyze by trend strength
spy_trend = X_full.loc[test_dates, 'SPY_Trend']
strong_up = spy_trend > 0.05
strong_down = spy_trend < -0.05

if strong_up.sum() > 20:
    _, _, acc_up = evaluate(X_test_scaled[strong_up.values], y_test_enc[strong_up.values])
    print(f"Strong uptrend: {acc_up * 100:.1f}% top-3 ({strong_up.sum()} samples)")

if strong_down.sum() > 20:
    _, _, acc_down = evaluate(X_test_scaled[strong_down.values], y_test_enc[strong_down.values])
    print(f"Strong downtrend: {acc_down * 100:.1f}% top-3 ({strong_down.sum()} samples)")

print(f"\n{'=' * 60}\n")