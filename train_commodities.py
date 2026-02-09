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
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def verify_no_lookahead(feature_date, return_start_date, return_end_date, horizon):
    """
    Verify that features don't use future information.

    Feature at date t should only use data up to and including date t.
    Returns should be calculated from t+1 to t+horizon.
    """
    if pd.isna(feature_date) or pd.isna(return_start_date) or pd.isna(return_end_date):
        return False

    # Feature date must be before return calculation period
    if feature_date >= return_start_date:
        return False

    # Return period should start immediately after feature date
    # (allowing for 1-2 day gap for practical purposes)
    days_gap = (return_start_date - feature_date).days
    if days_gap > 5:  # Allow small gap for weekends
        print(f"⚠️ Large gap between feature date ({feature_date}) and return start ({return_start_date})")

    return True


def create_features_safe(data, commodity_tickers, macro_tickers, current_date):
    """
    Create features using ONLY data up to and including current_date.
    This prevents lookahead bias.
    """
    # Use only data up to current_date
    historical_data = data.loc[:current_date].copy()

    # Need minimum data for feature calculation
    if len(historical_data) < 252:  # Need at least 1 year
        return None

    X = pd.Series(dtype=float)

    # --- Macro Features ---
    if 'DX-Y.NYB' in historical_data.columns:
        X['DXY_mom_3m'] = historical_data['DX-Y.NYB'].pct_change(63).iloc[-1]
        X['DXY_mom_1m'] = historical_data['DX-Y.NYB'].pct_change(21).iloc[-1]
        X['DXY_trend'] = (historical_data['DX-Y.NYB'].rolling(63).mean().iloc[-1] /
                         historical_data['DX-Y.NYB'].rolling(252).mean().iloc[-1])

    if '^VIX' in historical_data.columns:
        X['VIX_level'] = historical_data['^VIX'].rolling(21).mean().iloc[-1]
        X['VIX_change'] = historical_data['^VIX'].pct_change(21).iloc[-1]
        X['VIX_regime'] = int(historical_data['^VIX'].iloc[-1] >
                             historical_data['^VIX'].rolling(252).mean().iloc[-1])

    if '^TYX' in historical_data.columns and '^TNX' in historical_data.columns:
        X['Yield_Curve'] = historical_data['^TYX'].iloc[-1] - historical_data['^TNX'].iloc[-1]
        yield_curve_series = historical_data['^TYX'] - historical_data['^TNX']
        X['Yield_Curve_change'] = yield_curve_series.diff(21).iloc[-1]

    if 'LQD' in historical_data.columns and 'HYG' in historical_data.columns:
        X['Credit_Spread'] = historical_data['LQD'].iloc[-1] / historical_data['HYG'].iloc[-1]
        credit_spread_series = historical_data['LQD'] / historical_data['HYG']
        X['Credit_Spread_change'] = credit_spread_series.pct_change(21).iloc[-1]

    # --- Commodity-Specific Features ---
    for ticker in commodity_tickers:
        if ticker not in historical_data.columns:
            continue

        ticker_data = historical_data[ticker].dropna()
        if len(ticker_data) < 63:  # Need enough data
            continue

        # Momentum features
        X[f'{ticker}_mom_3m'] = ticker_data.pct_change(63).iloc[-1]
        X[f'{ticker}_mom_1m'] = ticker_data.pct_change(21).iloc[-1]
        X[f'{ticker}_mom_1w'] = ticker_data.pct_change(5).iloc[-1]

        # Volatility features
        X[f'{ticker}_vol_3m'] = ticker_data.rolling(63).std().iloc[-1]
        X[f'{ticker}_vol_1m'] = ticker_data.rolling(21).std().iloc[-1]

        # Trend features
        X[f'{ticker}_sma_ratio'] = ticker_data.iloc[-1] / ticker_data.rolling(63).mean().iloc[-1]
        X[f'{ticker}_ema_ratio'] = ticker_data.iloc[-1] / ticker_data.ewm(span=21).mean().iloc[-1]

        # Range features
        rolling_high = ticker_data.rolling(63).max().iloc[-1]
        rolling_low = ticker_data.rolling(63).min().iloc[-1]
        if rolling_high > rolling_low:
            X[f'{ticker}_pct_rank'] = (ticker_data.iloc[-1] - rolling_low) / (rolling_high - rolling_low)

        # RSI-like feature
        delta = ticker_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if loss > 0:
            rs = gain / loss
            X[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))

    # --- Cross-Commodity Features ---
    commodity_prices = historical_data[commodity_tickers].copy()

    # Average correlation
    recent_prices = commodity_prices.tail(63)
    if len(recent_prices) >= 63:
        corr_matrix = recent_prices.corr()
        # Get average correlation (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        X['commodity_corr_avg'] = corr_matrix.values[mask].mean()

    # Relative strength
    commodity_returns = commodity_prices.pct_change(21)
    if len(commodity_returns) > 0:
        avg_return = commodity_returns.iloc[-1].mean()
        for ticker in commodity_tickers:
            if ticker in commodity_prices.columns:
                X[f'{ticker}_rel_strength'] = commodity_returns[ticker].iloc[-1] - avg_return

    return X


def calculate_forward_returns(data, commodity_tickers, start_date, horizon):
    """
    Calculate forward returns starting AFTER start_date.
    Returns are calculated from start_date+1 to start_date+horizon+1.
    """
    try:
        # Find the index position of start_date
        if start_date not in data.index:
            return None, None, None

        start_idx = data.index.get_loc(start_date)

        # Need horizon+1 periods ahead for return calculation
        if start_idx + horizon + 1 >= len(data):
            return None, None, None

        # Start price is the NEXT day after start_date (no lookahead)
        return_start_idx = start_idx + 1
        return_end_idx = start_idx + horizon + 1

        start_prices = data[commodity_tickers].iloc[return_start_idx]
        end_prices = data[commodity_tickers].iloc[return_end_idx]

        returns = (end_prices - start_prices) / start_prices

        # Get actual dates
        return_start_date = data.index[return_start_idx]
        return_end_date = data.index[return_end_idx]

        return returns, return_start_date, return_end_date

    except Exception as e:
        print(f"Error calculating returns for {start_date}: {e}")
        return None, None, None


def train_quarterly_commodity_model(
        start_date="2000-01-01",
        horizon=63,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        min_train_size=504,  # 2 years minimum
        step_size=21,  # Monthly retraining
        auto_filter_commodities=True,  # Filter out bad data
        min_coverage=0.90,  # Minimum 90% data coverage
        max_null_streak=200  # Maximum 200-day gap
):
    """
    Train a quarterly commodity prediction model with FIXED lookahead bias.

    Uses walk-forward validation to ensure true out-of-sample testing.

    Parameters:
    -----------
    auto_filter_commodities : bool
        If True, automatically remove commodities with poor data quality
    min_coverage : float
        Minimum percentage of non-null data (0.0-1.0)
    max_null_streak : int
        Maximum consecutive days of missing data
    """

    print("🚀 Training Quarterly Commodity Model (FIXED VERSION)")
    print("=" * 70)
    print(f"📅 Data Range: {start_date} to present")
    print(f"🎯 Prediction Horizon: {horizon} days (~{horizon // 21} months)")
    print(f"📊 Commodities: {len(commodity_tickers)}")
    print(f"📈 Macro Indicators: {len(macro_tickers)}")
    print(f"🔒 Anti-Lookahead Safeguards: ENABLED")
    if auto_filter_commodities:
        print(f"🔍 Auto-Filter: ENABLED (coverage≥{min_coverage*100:.0f}%, gap≤{max_null_streak}d)")
    print("=" * 70 + "\n")

    # ========== DATA FETCHING ==========
    print("📥 Fetching data from Yahoo Finance...")
    all_tickers = commodity_tickers + macro_tickers

    try:
        raw_data = yf.download(all_tickers, start=start_date, progress=False)['Close']
        # Limited forward fill - only 5 days max
        data = raw_data.fillna(method='ffill', limit=5)
        print(f"✅ Downloaded {len(data)} days of data")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}\n")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None

    # ========== COMMODITY FILTERING ==========
    if auto_filter_commodities:
        print("🔍 Filtering commodities by data quality...")
        print(f"   Criteria: coverage≥{min_coverage*100:.0f}%, max gap≤{max_null_streak} days")
        print("-" * 70)

        good_commodities = []
        removed_commodities = []

        for ticker in commodity_tickers:
            if ticker not in data.columns:
                removed_commodities.append((ticker, "NOT IN DATA"))
                continue

            ticker_data = data[ticker]
            total_points = len(ticker_data)
            non_null_points = ticker_data.notna().sum()
            coverage = non_null_points / total_points

            # Check for long null streaks
            is_null = ticker_data.isna()
            max_streak = 0
            current_streak = 0

            for val in is_null:
                if val:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            # Evaluate
            if coverage >= min_coverage and max_streak <= max_null_streak:
                good_commodities.append(ticker)
                print(f"   ✅ {ticker:<10} Coverage: {coverage*100:>5.1f}% | Max gap: {max_streak:>4}d")
            else:
                reason = []
                if coverage < min_coverage:
                    reason.append(f"coverage {coverage*100:.1f}%")
                if max_streak > max_null_streak:
                    reason.append(f"{max_streak}d gap")
                removed_commodities.append((ticker, ", ".join(reason)))
                print(f"   ❌ {ticker:<10} Coverage: {coverage*100:>5.1f}% | Max gap: {max_streak:>4}d | {', '.join(reason)}")

        print("-" * 70)
        print(f"   ✅ Kept: {len(good_commodities)} commodities")
        print(f"   ❌ Removed: {len(removed_commodities)} commodities")

        if len(removed_commodities) > 0:
            print(f"\n   Removed: {', '.join([t for t, _ in removed_commodities])}")

        if len(good_commodities) < 3:
            print(f"\n❌ ERROR: Only {len(good_commodities)} commodities passed quality checks!")
            print("   Try lowering min_coverage or max_null_streak")
            return None

        # Update commodity list
        commodity_tickers_filtered = good_commodities
        print(f"\n   Using {len(commodity_tickers_filtered)} high-quality commodities\n")
    else:
        commodity_tickers_filtered = commodity_tickers
        print("⚠️ Commodity filtering disabled - using all commodities\n")

    # ========== WALK-FORWARD VALIDATION ==========
    print("🚶 Starting Walk-Forward Validation...")
    print(f"   Minimum training size: {min_train_size} days (~{min_train_size/252:.1f} years)")
    print(f"   Step size: {step_size} days (~{step_size/21:.1f} months)")
    print(f"   Horizon: {horizon} days (~{horizon/21:.1f} months)\n")

    results = []
    feature_list = None

    # Start from min_train_size and walk forward
    for i in range(min_train_size, len(data) - horizon - 1, step_size):
        current_date = data.index[i]

        # Create features using ONLY data up to current_date
        features = create_features_safe(data, commodity_tickers_filtered, macro_tickers, current_date)

        if features is None:
            continue

        # Calculate forward returns starting AFTER current_date
        returns, return_start_date, return_end_date = calculate_forward_returns(
            data, commodity_tickers_filtered, current_date, horizon
        )

        if returns is None or returns.isna().all():
            continue

        # Verify no lookahead bias
        if not verify_no_lookahead(current_date, return_start_date, return_end_date, horizon):
            print(f"⚠️ Lookahead bias detected at {current_date}, skipping...")
            continue

        # Find best performing commodity
        best_commodity = returns.idxmax()
        best_return = returns[best_commodity]

        if pd.isna(best_commodity) or pd.isna(best_return):
            continue

        # Store result
        result = {
            'feature_date': current_date,
            'return_start_date': return_start_date,
            'return_end_date': return_end_date,
            'features': features,
            'target': best_commodity,
            'returns': returns
        }
        results.append(result)

        # Track feature names (should be consistent)
        if feature_list is None:
            feature_list = features.index.tolist()

    print(f"✅ Created {len(results)} training samples\n")

    if len(results) < 50:
        print("❌ Insufficient samples for training")
        return None

    # ========== PREPARE TRAINING DATA ==========
    print("🧹 Preparing training data...")

    # Convert to DataFrame
    X_list = []
    y_list = []
    return_list = []

    for result in results:
        # Ensure all features are present
        feature_vector = []
        for feat in feature_list:
            feature_vector.append(result['features'].get(feat, np.nan))
        X_list.append(feature_vector)
        y_list.append(result['target'])
        return_list.append(result['returns'])

    X = pd.DataFrame(X_list, columns=feature_list)
    y = pd.Series(y_list)
    returns_df = pd.DataFrame(return_list)

    # Remove rows with NaN features
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    returns_df = returns_df[valid_mask]

    print(f"✅ Training samples after cleaning: {len(X)}")
    print(f"✅ Features: {len(X.columns)}")

    # Diagnostic: Check if low sample count is due to parameters
    if len(X) < 100:
        print(f"\n⚠️ LOW SAMPLE COUNT DETECTED ({len(X)} samples)")
        print(f"   This is likely due to:")
        print(f"   1. Large horizon ({horizon} days) reduces available samples")
        print(f"   2. Large step_size ({step_size} days) creates fewer training points")
        print(f"   3. Strict lookahead prevention removes invalid samples")
        print(f"\n   💡 TO INCREASE SAMPLES:")
        print(f"   - Use earlier start_date (currently {start_date})")
        print(f"   - Reduce step_size (currently {step_size}, try 5-10)")
        print(f"   - Reduce min_train_size (currently {min_train_size})")
        print(f"   - Accept that stricter validation = fewer samples (this is correct!)\n")

    print(f"\n✅ Class distribution:")
    class_dist = y.value_counts()
    for ticker, count in class_dist.head(10).items():
        print(f"   {ticker}: {count} ({count / len(y) * 100:.1f}%)")
    if len(class_dist) > 10:
        print(f"   ... and {len(class_dist) - 10} more commodities")
    print()

    # ========== WALK-FORWARD CROSS-VALIDATION ==========
    print("🔄 Walk-Forward Cross-Validation (True Out-of-Sample)")
    print("-" * 70)

    # Use 80% for training, 20% for testing in each fold
    n_splits = 5
    fold_size = len(X) // (n_splits + 1)

    print(f"   Total samples: {len(X)}")
    print(f"   Fold size: {fold_size}")
    print(f"   Minimum fold size required: 10\n")

    if fold_size < 10:
        print(f"⚠️ Insufficient data for cross-validation (fold size: {fold_size})")
        print(f"   Need at least {10 * (n_splits + 1)} samples for {n_splits}-fold CV")
        print(f"   Current samples: {len(X)}")
        print(f"   Skipping CV and using all data for training...\n")
        cv_scores = []
        cv_predictions = []
        cv_actuals = []
        cv_returns = []
    else:
        cv_scores = []
        cv_predictions = []
        cv_actuals = []
        cv_returns = []

        for fold in range(n_splits):
            # Training: use all data up to this point
            train_end = (fold + 1) * fold_size
            test_start = train_end
            test_end = min(test_start + fold_size, len(X))

            if test_end - test_start < 10:  # Need minimum test samples
                continue

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            returns_test = returns_df.iloc[test_start:test_end]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestClassifier(
                n_estimators=200,  # Reduced for speed
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            predictions = model.predict(X_test_scaled)
            score = accuracy_score(y_test, predictions)
            cv_scores.append(score)

            cv_predictions.extend(predictions)
            cv_actuals.extend(y_test)
            cv_returns.append(returns_test)

            print(f"   Fold {fold + 1}: {score:.2%} accuracy (train size: {len(X_train)}, test size: {len(X_test)})")

        if len(cv_scores) > 0:
            print(f"\n✅ CV Mean Accuracy: {np.mean(cv_scores):.2%} (±{np.std(cv_scores):.2%})")

            # Baseline
            baseline_acc = 1 / len(y.unique())
            print(f"📍 Baseline (random): {baseline_acc:.2%}")
            print(f"📈 Improvement: {((np.mean(cv_scores) - baseline_acc) / baseline_acc * 100):.1f}%")
        else:
            print("\n⚠️ No CV folds completed")

    print("-" * 70 + "\n")

    # ========== TRAIN FINAL MODEL ==========
    print("🤖 Training final model on all data...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    print("✅ Final model trained!\n")

    # ========== BACKTESTING ANALYSIS ==========
    print("💰 BACKTESTING STRATEGY RETURNS (Out-of-Sample)")
    print("-" * 70)

    # Combine all CV test periods
    if len(cv_returns) == 0:
        print("⚠️ No CV returns available for backtesting")
        print("   This likely means insufficient data for cross-validation")
        print("-" * 70 + "\n")
    else:
        all_cv_returns = pd.concat(cv_returns)

        # Strategy: Follow model predictions
        strategy_returns = []
        for pred, (idx, actual_returns) in zip(cv_predictions, all_cv_returns.iterrows()):
            if pred in actual_returns.index and pd.notna(actual_returns[pred]):
                strategy_returns.append(actual_returns[pred])

        if len(strategy_returns) > 0:
            strategy_returns = pd.Series(strategy_returns)
            avg_return = strategy_returns.mean()
            sharpe = calculate_sharpe_ratio(strategy_returns)
            win_rate = (strategy_returns > 0).sum() / len(strategy_returns)

            print(f"Model Strategy:")
            print(f"   Average Return: {avg_return:.2%} per {horizon}-day period")
            print(f"   Win Rate: {win_rate:.2%}")
            print(f"   Sharpe Ratio: {sharpe:.2f}")
            print(f"   Volatility: {strategy_returns.std():.2%}")

        # Benchmark: Equal-weight all commodities
        benchmark_returns = all_cv_returns.mean(axis=1).dropna()
        if len(benchmark_returns) > 0:
            benchmark_avg = benchmark_returns.mean()
            benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns)
            benchmark_win_rate = (benchmark_returns > 0).sum() / len(benchmark_returns)

            print(f"\nBenchmark (Equal-Weight All):")
            print(f"   Average Return: {benchmark_avg:.2%} per {horizon}-day period")
            print(f"   Win Rate: {benchmark_win_rate:.2%}")
            print(f"   Sharpe Ratio: {benchmark_sharpe:.2f}")
            print(f"   Volatility: {benchmark_returns.std():.2%}")

        print("-" * 70 + "\n")

    # ========== FEATURE IMPORTANCE ==========
    print("📊 TOP 20 FEATURE IMPORTANCES")
    print("-" * 70)

    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    for idx, row in feature_imp_df.head(20).iterrows():
        bar = '█' * int(row['Importance'] * 200)
        print(f"{row['Feature']:<30} | {bar} {row['Importance']:.4f}")
    print("-" * 70 + "\n")

    # ========== SAVE MODEL ==========

    # Calculate baseline accuracy
    baseline_acc = 1 / len(y.unique())

    model_data = {
        'model': model,
        'scaler': scaler,
        'features': X.columns.tolist(),
        'classes': model.classes_.tolist(),
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X),
        'horizon': horizon,
        'performance': {
            'cv_mean_accuracy': np.mean(cv_scores) if len(cv_scores) > 0 else None,
            'cv_std_accuracy': np.std(cv_scores) if len(cv_scores) > 0 else None,
            'baseline_accuracy': baseline_acc,
        },
        'feature_importance': feature_imp_df.to_dict('records')
    }

    joblib.dump(model_data, 'commodity_model_fixed.joblib')
    print(f"✅ Model saved as: commodity_model_fixed.joblib\n")

    return model_data


if __name__ == "__main__":
    model_data = train_quarterly_commodity_model(
        start_date="2000-01-01",
        horizon=63,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        min_train_size=252,  # 1 year (reduced from 2 years)
        step_size=5,  # Weekly (reduced from monthly)
        auto_filter_commodities=True,  # Enable filtering
        min_coverage=0.90,  # 90% minimum coverage
        max_null_streak=200  # 200-day maximum gap
    )