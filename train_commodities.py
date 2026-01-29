import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Assuming these are imported correctly from your local structure
from macro.constants import ML_MACRO_TICKERS, COMMODITIES

commodity_tickers = list(COMMODITIES.keys())
macro_tickers = ML_MACRO_TICKERS


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def train_quarterly_commodity_model(
        start_date="2000-01-01",
        horizon=63,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        use_cross_validation=True
):
    """
    Train a quarterly commodity prediction model with enhanced features.

    Parameters:
    -----------
    start_date : str
        Start date for historical data
    horizon : int
        Prediction horizon in trading days (63 ≈ 1 quarter)
    n_estimators : int
        Number of trees in random forest
    max_depth : int
        Maximum depth of trees
    min_samples_leaf : int
        Minimum samples required at leaf node
    use_cross_validation : bool
        Whether to perform time-series cross-validation
    """

    print("🚀 Training Quarterly Commodity Model...")
    print(f"📅 Data Range: {start_date} to present")
    print(f"🎯 Prediction Horizon: {horizon} days (~{horizon // 21} months)")
    print(f"📊 Commodities: {len(commodity_tickers)}")
    print(f"📈 Macro Indicators: {len(macro_tickers)}\n")

    # ========== DATA FETCHING ==========
    print("📥 Fetching data from Yahoo Finance...")
    all_tickers = commodity_tickers + macro_tickers

    try:
        data = yf.download(all_tickers, start=start_date, progress=False)['Close'].ffill()
        print(f"✅ Downloaded {len(data)} days of data\n")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

    # ========== FEATURE ENGINEERING ==========
    print("⚙️  Engineering features...")
    X = pd.DataFrame(index=data.index)

    # --- Macro Features ---
    if 'DX-Y.NYB' in data.columns:
        X['DXY_mom_3m'] = data['DX-Y.NYB'].pct_change(63)
        X['DXY_mom_1m'] = data['DX-Y.NYB'].pct_change(21)
        X['DXY_trend'] = data['DX-Y.NYB'].rolling(63).mean() / data['DX-Y.NYB'].rolling(252).mean()

    if '^VIX' in data.columns:
        X['VIX_level'] = data['^VIX'].rolling(21).mean()
        X['VIX_change'] = data['^VIX'].pct_change(21)
        X['VIX_regime'] = (data['^VIX'] > data['^VIX'].rolling(252).mean()).astype(int)

    if '^TYX' in data.columns and '^TNX' in data.columns:
        X['Yield_Curve'] = data['^TYX'] - data['^TNX']
        X['Yield_Curve_change'] = X['Yield_Curve'].diff(21)

    if 'LQD' in data.columns and 'HYG' in data.columns:
        X['Credit_Spread'] = data['LQD'] / data['HYG']
        X['Credit_Spread_change'] = X['Credit_Spread'].pct_change(21)

    # --- Commodity-Specific Features ---
    for ticker in commodity_tickers:
        if ticker not in data.columns:
            continue

        # Momentum features (multiple timeframes)
        X[f'{ticker}_mom_3m'] = data[ticker].pct_change(63)
        X[f'{ticker}_mom_1m'] = data[ticker].pct_change(21)
        X[f'{ticker}_mom_1w'] = data[ticker].pct_change(5)

        # Volatility features
        X[f'{ticker}_vol_3m'] = data[ticker].rolling(63).std()
        X[f'{ticker}_vol_1m'] = data[ticker].rolling(21).std()

        # Trend features
        X[f'{ticker}_sma_ratio'] = data[ticker] / data[ticker].rolling(63).mean()
        X[f'{ticker}_ema_ratio'] = data[ticker] / data[ticker].ewm(span=21).mean()

        # Range features
        rolling_high = data[ticker].rolling(63).max()
        rolling_low = data[ticker].rolling(63).min()
        X[f'{ticker}_pct_rank'] = (data[ticker] - rolling_low) / (rolling_high - rolling_low)

        # RSI-like feature
        delta = data[ticker].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        X[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))

    # --- Cross-Commodity Features ---
    commodity_prices = data[commodity_tickers].copy()
    X['commodity_corr_avg'] = commodity_prices.rolling(63).corr().groupby(level=0).mean().mean(axis=1)

    # Relative strength across commodities
    for ticker in commodity_tickers:
        if ticker in data.columns:
            X[f'{ticker}_rel_strength'] = (
                    data[ticker].pct_change(21) -
                    commodity_prices.pct_change(21).mean(axis=1)
            )

    print(f"✅ Created {len(X.columns)} features\n")

    # ========== TARGET VARIABLE ==========
    print("🎯 Creating target variable (best-performing commodity)...")
    returns = data[commodity_tickers].pct_change(horizon).shift(-horizon)
    y = returns.idxmax(axis=1)

    # Store actual returns for later analysis
    actual_returns = returns.copy()

    # ========== DATA PREPARATION ==========
    print("🧹 Cleaning data...")
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_train = valid.drop(columns=['target'])
    y_train = valid['target']

    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Class distribution:")
    class_dist = y_train.value_counts()
    for ticker, count in class_dist.items():
        print(f"   {ticker}: {count} ({count / len(y_train) * 100:.1f}%)")
    print()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ========== MODEL TRAINING ==========
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y_train)
    print("✅ Model trained!\n")

    # ========== MODEL EVALUATION ==========
    print("📊 MODEL PERFORMANCE")
    print("=" * 70)

    # Get predictions and probabilities
    proba = model.predict_proba(X_scaled)
    classes = model.classes_

    # Top-1 Accuracy
    top1_preds = classes[np.argmax(proba, axis=1)]
    top1_acc = accuracy_score(y_train, top1_preds)

    # Top-3 Accuracy
    top3_correct = 0
    for i in range(len(y_train)):
        top3_idx = np.argsort(proba[i])[-3:]
        top3_classes = classes[top3_idx]
        if y_train.iloc[i] in top3_classes:
            top3_correct += 1
    top3_acc = top3_correct / len(y_train)

    # Top-5 Accuracy
    top5_correct = 0
    for i in range(len(y_train)):
        top5_idx = np.argsort(proba[i])[-5:]
        top5_classes = classes[top5_idx]
        if y_train.iloc[i] in top5_classes:
            top5_correct += 1
    top5_acc = top5_correct / len(y_train)

    print(f"✅ Top-1 Accuracy: {top1_acc:.2%}")
    print(f"✅ Top-3 Accuracy: {top3_acc:.2%}")
    print(f"✅ Top-5 Accuracy: {top5_acc:.2%}")

    # Baseline (random guess)
    baseline_acc = 1 / len(classes)
    print(f"📍 Baseline (random): {baseline_acc:.2%}")
    print(f"📈 Improvement: {(top1_acc - baseline_acc) / baseline_acc * 100:.1f}%")
    print("=" * 70 + "\n")

    # ========== CROSS-VALIDATION ==========
    if use_cross_validation:
        print("🔄 Performing Time-Series Cross-Validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
            X_cv_train, X_cv_test = X_scaled[train_idx], X_scaled[test_idx]
            y_cv_train, y_cv_test = y_train.iloc[train_idx], y_train.iloc[test_idx]

            cv_model = RandomForestClassifier(
                n_estimators=200,  # Reduced for speed
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            cv_model.fit(X_cv_train, y_cv_train)
            cv_score = cv_model.score(X_cv_test, y_cv_test)
            cv_scores.append(cv_score)
            print(f"   Fold {fold}: {cv_score:.2%}")

        print(f"✅ CV Mean: {np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})\n")

    # ========== FEATURE IMPORTANCE ==========
    print("📊 TOP 20 FEATURE IMPORTANCES")
    print("-" * 70)

    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()

    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    for idx, row in feature_imp_df.head(20).iterrows():
        bar = '█' * int(row['Importance'] * 200)  # Scale for visibility
        print(f"{row['Feature']:<30} | {bar} {row['Importance']:.4f}")
    print("-" * 70 + "\n")

    # ========== BACKTESTING ANALYSIS ==========
    print("💰 BACKTESTING STRATEGY RETURNS")
    print("-" * 70)

    # Align predictions with actual returns
    aligned_returns = actual_returns.loc[y_train.index]

    # Strategy 1: Always pick top-1 prediction
    strategy_returns = []
    for i, idx in enumerate(y_train.index):
        predicted_commodity = top1_preds[i]
        if predicted_commodity in aligned_returns.columns:
            ret = aligned_returns.loc[idx, predicted_commodity]
            if pd.notna(ret):
                strategy_returns.append(ret)

    if strategy_returns:
        avg_return = np.mean(strategy_returns)
        sharpe = calculate_sharpe_ratio(pd.Series(strategy_returns))
        win_rate = sum(r > 0 for r in strategy_returns) / len(strategy_returns)

        print(f"Top-1 Strategy:")
        print(f"   Average Return: {avg_return:.2%}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")

    # Strategy 2: Equally weight top-3 predictions
    strategy_returns_top3 = []
    for i, idx in enumerate(y_train.index):
        top3_idx = np.argsort(proba[i])[-3:]
        top3_classes = classes[top3_idx]

        period_returns = []
        for commodity in top3_classes:
            if commodity in aligned_returns.columns:
                ret = aligned_returns.loc[idx, commodity]
                if pd.notna(ret):
                    period_returns.append(ret)

        if period_returns:
            strategy_returns_top3.append(np.mean(period_returns))

    if strategy_returns_top3:
        avg_return_top3 = np.mean(strategy_returns_top3)
        sharpe_top3 = calculate_sharpe_ratio(pd.Series(strategy_returns_top3))
        win_rate_top3 = sum(r > 0 for r in strategy_returns_top3) / len(strategy_returns_top3)

        print(f"\nTop-3 Equal-Weight Strategy:")
        print(f"   Average Return: {avg_return_top3:.2%}")
        print(f"   Win Rate: {win_rate_top3:.2%}")
        print(f"   Sharpe Ratio: {sharpe_top3:.2f}")

    # Benchmark: Buy-and-hold equal-weight all commodities
    benchmark_returns = aligned_returns.mean(axis=1).dropna()
    if len(benchmark_returns) > 0:
        benchmark_avg = benchmark_returns.mean()
        benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)
        benchmark_win_rate = sum(benchmark_returns > 0) / len(benchmark_returns)

        print(f"\nBenchmark (Equal-Weight All):")
        print(f"   Average Return: {benchmark_avg:.2%}")
        print(f"   Win Rate: {benchmark_win_rate:.2%}")
        print(f"   Sharpe Ratio: {benchmark_sharpe:.2f}")

    print("-" * 70 + "\n")

    model_data = {
        'model': model,
        'scaler': scaler,
        'features': X_train.columns.tolist(),
        'classes': classes.tolist(),
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X_train),
        'horizon': horizon,
        'performance': {
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
        },
        'feature_importance': feature_imp_df.to_dict('records')
    }
    # Also save as default name for easy loading
    joblib.dump(model_data, 'commodity_model.joblib')
    print(f"✅ Model also saved as: commodity_model.joblib\n")

    return model_data


if __name__ == "__main__":
    model_data = train_quarterly_commodity_model(
        start_date="2000-01-01",
        horizon=63,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        use_cross_validation=True
    )