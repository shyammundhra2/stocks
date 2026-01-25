import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime

# ----------------- Configuration -----------------
SECTOR_TICKERS = ['XLE', 'XLB', 'XLI', 'XLY', 'XLF', 'XLC', 'XLK', 'XLV', 'XLP', 'XLU', 'XLRE']
MACRO_DRIVERS = ['^TNX', '^IRX', 'DX-Y.NYB', 'CL=F', 'GLD', '^VIX']  # Removed ^MOVE due to data availability


def train_enhanced_macro_model():
    print("🚀 Initializing Macro-Regime Sector Model...")
    print(f"⏰ Training as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Data Acquisition with Error Handling
    all_tickers = list(set(SECTOR_TICKERS + MACRO_DRIVERS + ['SPY']))
    print(f"📊 Downloading {len(all_tickers)} tickers from 2000-01-01...")

    try:
        raw_data = yf.download(all_tickers, start="2000-01-01", progress=False)['Close']
    except Exception as e:
        print(f"❌ Download error: {e}")
        return

    # Handle missing data
    missing_tickers = [t for t in all_tickers if t not in raw_data.columns]
    if missing_tickers:
        print(f"⚠️  Warning: Missing tickers: {missing_tickers}")
        all_tickers = [t for t in all_tickers if t in raw_data.columns]

    raw_data = raw_data.ffill().bfill()  # Forward fill then backfill
    print(f"✅ Data loaded: {len(raw_data)} trading days\n")

    # 2. Feature Engineering (Match predict.py naming conventions)
    print("🛠️  Building Macro & Factor Features...")
    X = pd.DataFrame(index=raw_data.index)

    # Macro features using predict.py conventions
    if '^TNX' in raw_data.columns and '^IRX' in raw_data.columns:
        X['Yield_Curve'] = raw_data['^TNX'] - raw_data['^IRX']
        X['TNX_vol'] = raw_data['^TNX'].rolling(63).std()

    if 'DX-Y.NYB' in raw_data.columns:
        X['DXY_mom'] = raw_data['DX-Y.NYB'].pct_change(63)
        X['DXY_Mom'] = raw_data['DX-Y.NYB'].pct_change(21)

    if 'CL=F' in raw_data.columns:
        X['Oil_Trend'] = raw_data['CL=F'].pct_change(63)

    if '^VIX' in raw_data.columns:
        X['VIX_Level'] = raw_data['^VIX']
        X['VIX_level'] = raw_data['^VIX'].rolling(21).mean()

    if 'SPY' in raw_data.columns:
        X['Equity_Regime'] = raw_data['SPY'].pct_change(126)
        X['SPY_Volatility'] = raw_data['SPY'].pct_change(21).rolling(63).std()

    # Relative momentum features (match predict.py naming)
    for ticker in SECTOR_TICKERS:
        if ticker in raw_data.columns and 'SPY' in raw_data.columns:
            relative_ratio = raw_data[ticker] / raw_data['SPY']
            X[f'{ticker}_Rel_Mom_3M'] = relative_ratio.pct_change(63)
            X[f'{ticker}_Rel_Mom_1M'] = relative_ratio.pct_change(21)
            # Add direct momentum features for consistency
            X[f'{ticker}_mom'] = raw_data[ticker].pct_change(63)
            X[f'{ticker}_vol'] = raw_data[ticker].rolling(63).std()

    # 3. Target Generation (FIXED: No Look-Ahead Bias)
    print("🎯 Generating Multi-Label Targets (No Look-Ahead Bias)...")
    horizon = 63  # ~3 months forward

    # Calculate future returns properly
    future_prices = raw_data[SECTOR_TICKERS].shift(-horizon)
    current_prices = raw_data[SECTOR_TICKERS]
    sector_returns = (future_prices - current_prices) / current_prices

    # Only keep dates where we have both features and future returns
    valid_dates = X.dropna().index.intersection(sector_returns.dropna().index)
    print(f"📅 Valid training dates: {len(valid_dates)}")

    # Build dataset - predict each sector as a separate class
    X_list, y_list, date_labels = [], [], []

    for date in valid_dates:
        returns_on_date = sector_returns.loc[date]

        # Find the best performing sector (not top 3, but THE top)
        best_sector = returns_on_date.idxmax()

        # Create one sample per date with best sector as target
        sector_feat = X.loc[date].to_dict()

        X_list.append(sector_feat)
        y_list.append(best_sector)  # String label (ticker)
        date_labels.append(date)

    X_final = pd.DataFrame(X_list).fillna(0)
    y_final = pd.Series(y_list)
    date_labels = pd.Series(date_labels)

    print(f"✅ Dataset built: {len(X_final)} samples")
    print(f"   Target sectors: {y_final.nunique()} unique")
    print(f"   Class distribution:")
    for sector, count in y_final.value_counts().items():
        print(f"      {sector}: {count} ({count / len(y_final) * 100:.1f}%)")

    # 4. Proper Time-Based Split (No Data Leakage)
    unique_dates = sorted(date_labels.unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_mask = date_labels < split_date
    test_mask = date_labels >= split_date

    X_train, X_test = X_final[train_mask], X_final[test_mask]
    y_train, y_test = y_final[train_mask], y_final[test_mask]

    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} samples ({train_mask.sum()} dates)")
    print(f"   Test:  {len(X_test)} samples ({test_mask.sum()} dates)")
    print(f"   Split Date: {split_date.strftime('%Y-%m-%d')}\n")

    # 5. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Training
    print("🧠 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("✅ Training complete\n")

    # 7. Evaluation
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)

    print("=" * 60)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\n{'METRIC':<30} {'TRAIN':<15} {'TEST (OOS)':<15}")
    print("-" * 60)
    print(
        f"{'Overall Accuracy':<30} {accuracy_score(y_train, train_preds):>14.1%} {accuracy_score(y_test, test_preds):>14.1%}")

    # Sector-wise accuracy (how often each sector was correctly predicted when it was the best)
    print("\n" + "=" * 60)
    print("📊 SECTOR PREDICTION ACCURACY (When Sector Was Best)")
    print("=" * 60)

    sector_results = []
    for ticker in SECTOR_TICKERS:
        mask = (y_test == ticker)
        if mask.sum() > 0:
            correct_predictions = (test_preds[mask] == ticker).sum()
            total = mask.sum()
            acc = correct_predictions / total
            sector_results.append((ticker, acc, total, correct_predictions))

    # Sort by accuracy
    sector_results.sort(key=lambda x: x[1], reverse=True)

    for ticker, acc, total, correct in sector_results:
        bar = '█' * int(acc * 30)
        print(f"{ticker:<6} ({correct:>2}/{total:<2} correct) | {bar:<30} {acc:.1%}")

    # Classification report
    print("\n" + "=" * 60)
    print("📋 DETAILED CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)

    # For multi-class, show per-class precision/recall
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, test_preds, labels=SECTOR_TICKERS)

    print(f"\n{'Sector':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, sector in enumerate(SECTOR_TICKERS):
        print(f"{sector:<8} {precision[i]:>11.3f} {recall[i]:>11.3f} {f1[i]:>11.3f} {support[i]:>9}")

    print(f"\n{'Macro Avg':<8} {precision.mean():>11.3f} {recall.mean():>11.3f} {f1.mean():>11.3f}")
    print(f"{'Accuracy':<8} {accuracy_score(y_test, test_preds):>11.3f}")

    # 8. Feature Importance
    print("=" * 60)
    print("🔍 TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 60)

    importances = model.feature_importances_
    feat_names = X_train.columns
    indices = np.argsort(importances)[::-1][:15]

    for i, idx in enumerate(indices):
        bar = '█' * int(importances[idx] * 100)
        print(f"{i + 1:>2}. {feat_names[idx]:<30} | {bar:<20} {importances[idx]:.4f}")

    # 9. Top-3 Accuracy Analysis
    print("\n" + "=" * 60)
    print("🎯 TOP-3 PREDICTION ACCURACY")
    print("=" * 60)

    # Get top 3 predicted probabilities for each test sample
    test_probs = model.predict_proba(X_test_scaled)
    top3_correct = 0

    for i, true_sector in enumerate(y_test):
        # Get indices of top 3 predicted sectors
        top3_indices = np.argsort(test_probs[i])[-3:]
        top3_sectors = [model.classes_[idx] for idx in top3_indices]

        if true_sector in top3_sectors:
            top3_correct += 1

    top3_accuracy = top3_correct / len(y_test)
    print(f"\nAccuracy (Best Sector in Top 3): {top3_accuracy:.1%}")
    print(f"   {top3_correct} out of {len(y_test)} test samples")

    # 10. Confusion Matrix (simplified view)
    print("\n" + "=" * 60)
    print("📊 PREDICTION DISTRIBUTION")
    print("=" * 60)

    pred_counts = pd.Series(test_preds).value_counts()
    actual_counts = pd.Series(y_test).value_counts()

    print(f"\n{'Sector':<8} {'Predicted':<12} {'Actual':<12}")
    print("-" * 40)
    for sector in SECTOR_TICKERS:
        pred = pred_counts.get(sector, 0)
        actual = actual_counts.get(sector, 0)
        print(f"{sector:<8} {pred:>11} {actual:>11}")

    # 10. Save Model
    print("\n" + "=" * 60)
    model_artifact = {
        'model': model,
        'scaler': scaler,
        'features': X_train.columns.tolist(),
        'sector_tickers': SECTOR_TICKERS,
        'train_date': split_date,
        'train_samples': len(X_train),
        'test_accuracy': accuracy_score(y_test, test_preds)
    }

    joblib.dump(model_artifact, 'sector_model.joblib')
    print("💾 Model saved to: sector_model.joblib")
    print("=" * 60)
    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    train_enhanced_macro_model()