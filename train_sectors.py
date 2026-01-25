import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib

from macro.constants import ML_MACRO_TICKERS

# ----------------- Configuration -----------------
SECTOR_TICKERS = ['XLE', 'XLB', 'XLI', 'XLY', 'XLF', 'XLC', 'XLK', 'XLV', 'XLP', 'XLU', 'XLRE']


def train_quarterly_model():
    print("🚀 Initializing Balanced Historical Sector Model...")

    # ---------------- Data Acquisition ----------------
    tickers = SECTOR_TICKERS + list(ML_MACRO_TICKERS) + ['SPY', 'RSP', 'TLT']
    raw_data = yf.download(tickers, start="2000-01-01")['Close'].ffill()
    raw_data = raw_data.shift(1)  # prevent lookahead bias

    # ---------------- Feature Engineering ----------------
    print("🛠️  Engineering Momentum & Macro Features...")
    X = pd.DataFrame(index=raw_data.index)

    # Macro features
    for macro in ['DX-Y.NYB', '^VIX', '^MOVE', '^TNX', 'LQD', 'HYG']:
        if macro in raw_data.columns:
            X[f'{macro}_Level'] = raw_data[macro]
        else:
            X[f'{macro}_Level'] = 0

    # Credit Spread
    X['Credit_Spread'] = raw_data['LQD'] / raw_data['HYG'] if (
                'LQD' in raw_data.columns and 'HYG' in raw_data.columns) else 0

    # RiskOn regime
    X['RiskOn'] = (raw_data['SPY'].pct_change(63) > 0).astype(int) if 'SPY' in raw_data.columns else 1

    # ---------------- Sector Momentum ----------------
    momentum_windows = {'1W': 5, '1M': 21, '3M': 63, '6M': 126}
    for ticker in SECTOR_TICKERS:
        if ticker not in raw_data.columns:
            for period in momentum_windows:
                X[f'{ticker}_Mom_{period}'] = 0
            X[f'{ticker}_Mom_Accel'] = 0
            continue

        for period_name, window in momentum_windows.items():
            X[f'{ticker}_Mom_{period_name}'] = raw_data[ticker].pct_change(window)

        X[f'{ticker}_Mom_Accel'] = X[f'{ticker}_Mom_1M'] - X[f'{ticker}_Mom_3M']

    # Momentum Clipping (Winsorization ±2 std per quarter)
    for period in momentum_windows:
        cols = [f'{t}_Mom_{period}' for t in SECTOR_TICKERS if f'{t}_Mom_{period}' in X.columns]
        X[cols] = X[cols].apply(lambda x: np.clip(x, x.mean() - 2 * x.std(), x.mean() + 2 * x.std()), axis=1)

    # ---------------- Historical Quarterly Top-Performer Weights ----------------
    horizon = 63
    sector_returns = raw_data[SECTOR_TICKERS].pct_change(horizon).shift(-horizon)

    top_count = {t: 0 for t in SECTOR_TICKERS}
    for date in sector_returns.index:
        row = sector_returns.loc[date]
        if row.isna().all():
            continue
        top3 = row.nlargest(3).index.tolist()
        for t in top3:
            top_count[t] += 1

    # Normalize weights to sum to 1
    total_count = sum(top_count.values())
    historical_weights = {t: top_count[t] / total_count for t in SECTOR_TICKERS}

    print("\n📊 Normalized Historical Quarterly Top-Performer Weights (25Y):")
    for sector, w in historical_weights.items():
        print(f"{sector:<6} → {w:.2%}")

    # ---------------- Target Generation (Winner Takes All) ----------------
    # Label only THE BEST performer (clearer signal for model)
    y_list = []
    dates_list = []

    for date in sector_returns.index:
        if date not in X.index:
            continue
        row = sector_returns.loc[date]
        if row.isna().all():
            continue

        # Get the single best performer
        best_sector = row.idxmax()
        y_list.append(best_sector)
        dates_list.append(date)

    X_expanded = X.loc[dates_list]
    y_expanded = pd.Series(y_list, index=dates_list)

    # ---------------- Clean & Split ----------------
    valid_data = pd.concat([X_expanded.reset_index(drop=True), y_expanded.rename('target').reset_index(drop=True)],
                           axis=1).dropna()
    split_idx = int(len(valid_data) * 0.8)
    train_df = valid_data.iloc[:split_idx].copy()
    test_df = valid_data.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    print(f"\n📊 Training Label Distribution:")
    train_dist = y_train.value_counts(normalize=True).sort_index()
    for sector in SECTOR_TICKERS:
        if sector in train_dist.index:
            pct = train_dist[sector] * 100
            bar = '█' * max(1, int(pct))
            print(f"{sector:<6} | {bar} {pct:.1f}%")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Train Random Forest ----------------
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=12,  # Allow more complex patterns
        min_samples_leaf=5,  # Allow finer splits
        min_samples_split=15,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    print("\n🧠 Training Random Forest Ensemble...")
    model.fit(X_train_scaled, y_train)

    # ---------------- Test Set Predictions with Top-3 Ensemble ----------------
    probs = model.predict_proba(X_test_scaled)
    classes = model.classes_
    adjusted_preds = []
    top3_hits = 0

    for i, row in enumerate(probs):
        prob_dict = dict(zip(classes, row))

        # Weight by historical weights (gentle blend)
        for sector in prob_dict:
            hist_weight = historical_weights.get(sector, 0.09)
            # 80% model confidence, 20% history (trust the model more)
            prob_dict[sector] = 0.8 * prob_dict[sector] + 0.2 * hist_weight

        # Normalize
        total = sum(prob_dict.values())
        prob_dict = {k: v / total for k, v in prob_dict.items()}

        # Get top 3 predictions
        sorted_sectors = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top3_sectors = [s[0] for s in sorted_sectors[:3]]

        # Check if actual is in top 3
        if i < len(y_test) and y_test.iloc[i] in top3_sectors:
            top3_hits += 1

        # Pick the top prediction
        adjusted_preds.append(sorted_sectors[0][0])

    test_predictions = np.array(adjusted_preds)
    top3_accuracy = top3_hits / len(y_test) if len(y_test) > 0 else 0

    # ---------------- Validation ----------------
    pred_dist = pd.Series(test_predictions).value_counts(normalize=True).sort_index()
    print("\n📈 Test Set Prediction Distribution:")
    for sector in SECTOR_TICKERS:
        if sector in pred_dist.index:
            pct = pred_dist[sector] * 100
            hist_pct = historical_weights.get(sector, 0) * 100
            bar = '█' * max(1, int(pct))
            print(f"{sector:<6} | {bar} {pct:.1f}% (hist: {hist_pct:.1f}%)")

    accuracy = (test_predictions == y_test).mean()
    print(f"\n🎯 Test Accuracy: {accuracy:.1%}")

    # ---------------- Feature Importance ----------------
    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()
    feature_imp_df = pd.DataFrame({'Driver': feature_names, 'Importance': importances}).sort_values(by='Importance',
                                                                                                    ascending=False).head(
        15)
    print("\n📊 Top 15 Feature Drivers:")
    for _, row in feature_imp_df.iterrows():
        bar = '█' * max(1, int(row['Importance'] * 100))
        print(f"{row['Driver']:<30} | {bar} {row['Importance']:.3f}")

    # ---------------- Save Model ----------------
    bundle = {'model': model, 'scaler': scaler, 'features': feature_names}
    joblib.dump(bundle, 'sector_model_balanced.joblib')
    diagnostics = {
        'class_distribution': y_train.value_counts().to_dict(),
        'test_accuracy': accuracy,
        'prediction_distribution': pred_dist.to_dict(),
        'feature_importance': feature_imp_df.to_dict(),
        'historical_weights': historical_weights
    }
    joblib.dump(diagnostics, 'sector_model_balanced_diagnostics.joblib')
    print("\n✅ Training Complete. Model & diagnostics saved.")


if __name__ == "__main__":
    train_quarterly_model()