import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from macro.constants import ML_MACRO_TICKERS

# ----------------- 1. Configuration -----------------
# Sector tickers (ETFs)
SECTOR_TICKERS = ['VDE', 'XLB', 'VIS', 'XLY', 'XLF', 'XLC', 'VGT', 'XLV', 'XLP', 'XLU', 'XLRE']


def train_quarterly_model():
    print("🚀 Initializing Institutional-Grade Quarterly Model Training...")

    # 2. Data Acquisition (25 Years of History)
    tickers = SECTOR_TICKERS + list(ML_MACRO_TICKERS) + ['SPY', 'RSP', 'TLT']  # Added SPY, RSP for breadth
    raw_data = yf.download(tickers, start="2000-01-01")['Close'].ffill()

    # 3. Feature Engineering (Matches Live Dashboard Names)
    print("🛠️  Engineering Macro Features...")
    X = pd.DataFrame(index=raw_data.index)

    # === MACRO FEATURES ===
    # Quarterly Momentum (63 trading days)
    X['DXY_mom'] = raw_data['DX-Y.NYB'].pct_change(63)
    X['DXY_Mom'] = raw_data['DX-Y.NYB'].pct_change(21)  # Monthly momentum too

    # Risk & Volatility Metrics
    X['VIX_level'] = raw_data['^VIX'].rolling(21).mean()  # Monthly smoothed
    X['VIX_Level'] = raw_data['^VIX']  # Current level (low VIX = risk-on = tech)
    X['MOVE_level'] = raw_data['^MOVE'].ffill().fillna(raw_data['^MOVE'].mean())
    X['MOVE_Level'] = raw_data['^MOVE']
    X['TNX_vol'] = raw_data['^TNX'].rolling(63).std()  # Quarterly yield volatility

    # Spread Metrics (Credit & Yield Curve)
    X['Yield_Curve'] = raw_data['^TYX'] - raw_data['^TNX']  # 30Y - 10Y (steepening = growth)
    X['Credit_Spread'] = raw_data['LQD'] / raw_data['HYG']  # IG vs HY ratio
    X['Credit_Spread_Proxy'] = X['Credit_Spread']

    # === COMPREHENSIVE SECTOR MOMENTUM FEATURES ===
    print("🚀 Adding Comprehensive Sector Momentum Features...")

    # For each sector, calculate multiple momentum timeframes
    momentum_windows = {
        '1W': 5,  # Weekly (captures very short-term moves)
        '1M': 21,  # Monthly (short-term trend)
        '3M': 63,  # Quarterly (medium-term trend)
        '6M': 126,  # Semi-annual (longer trend)
    }

    volatility_windows = {
        '1M': 21,  # Short-term volatility
        '3M': 63,  # Medium-term volatility
    }

    for ticker in SECTOR_TICKERS:
        if ticker not in raw_data.columns:
            continue

        # Momentum across multiple timeframes
        for period_name, window in momentum_windows.items():
            X[f'{ticker}_Mom_{period_name}'] = raw_data[ticker].pct_change(window)

        # Volatility (risk-adjusted returns)
        for period_name, window in volatility_windows.items():
            X[f'{ticker}_Vol_{period_name}'] = raw_data[ticker].rolling(window).std()

        # Rate of change in momentum (acceleration)
        X[f'{ticker}_Mom_Accel'] = X[f'{ticker}_Mom_1M'] - X[f'{ticker}_Mom_3M']

        # Moving average relationships (trend strength)
        sma_50 = raw_data[ticker].rolling(50).mean()
        sma_200 = raw_data[ticker].rolling(200).mean()
        X[f'{ticker}_SMA50_dist'] = (raw_data[ticker] - sma_50) / sma_50
        X[f'{ticker}_SMA200_dist'] = (raw_data[ticker] - sma_200) / sma_200
        X[f'{ticker}_Golden_Cross'] = (sma_50 > sma_200).astype(int)  # Bullish signal

        # Risk-adjusted momentum (Sharpe-like ratio)
        returns_3m = raw_data[ticker].pct_change(63)
        vol_3m = raw_data[ticker].rolling(63).std()
        X[f'{ticker}_Risk_Adj_Mom'] = returns_3m / (vol_3m + 1e-6)

    # === CROSS-SECTIONAL MOMENTUM (Relative Rankings) ===
    print("📊 Computing Cross-Sectional Momentum Rankings...")

    # For each date, rank sectors by momentum (1 = best, 11 = worst)
    for period_name, window in momentum_windows.items():
        sector_momentum = pd.DataFrame({
            ticker: raw_data[ticker].pct_change(window)
            for ticker in SECTOR_TICKERS if ticker in raw_data.columns
        })

        # Rank from 0 (worst) to 1 (best)
        momentum_ranks = sector_momentum.rank(axis=1, pct=True)

        for ticker in SECTOR_TICKERS:
            if ticker in momentum_ranks.columns:
                X[f'{ticker}_Rank_{period_name}'] = momentum_ranks[ticker]

    # === MOMENTUM SPREAD FEATURES (Identify divergences) ===
    print("🔄 Adding Momentum Spread Features...")

    # Calculate momentum difference between pairs of sectors
    key_pairs = [
        ('VGT', 'XLF', 'Tech_vs_Finance'),  # Growth vs cyclical
        ('VGT', 'XLV', 'Growth_vs_Defensive'),  # Growth vs defensive
        ('VGT', 'XLU', 'Tech_vs_Utilities'),  # Risk-on vs risk-off
        ('XLF', 'XLU', 'Cyclical_vs_Defensive'),  # Economic sensitivity
        ('XLY', 'XLP', 'Discretionary_vs_Staples'),  # Consumer strength
    ]

    for ticker1, ticker2, name in key_pairs:
        if ticker1 in raw_data.columns and ticker2 in raw_data.columns:
            # Relative strength ratio
            X[name] = raw_data[ticker1] / raw_data[ticker2]
            X[f'{name}_Mom'] = X[name].pct_change(63)

            # Momentum differential
            X[f'{name}_Spread'] = X[f'{ticker1}_Mom_3M'] - X[f'{ticker2}_Mom_3M']

    # === RELATIVE STRENGTH VS MARKET ===
    if 'SPY' in raw_data.columns:
        spy = raw_data['SPY']
        for ticker in SECTOR_TICKERS:
            if ticker not in raw_data.columns:
                continue

            # Relative strength (sector/market ratio)
            X[f'{ticker}_vs_SPY'] = raw_data[ticker] / spy

            # Momentum of relative strength
            for period_name, window in momentum_windows.items():
                X[f'{ticker}_vs_SPY_Mom_{period_name}'] = X[f'{ticker}_vs_SPY'].pct_change(window)

    # === MARKET BREADTH (Confirms healthy rallies vs narrow leadership) ===
    if 'RSP' in raw_data.columns and 'SPY' in raw_data.columns:
        X['Breadth_Ratio'] = raw_data['RSP'] / raw_data['SPY']
        X['Breadth_MA_Diff'] = X['Breadth_Ratio'] - X['Breadth_Ratio'].rolling(50).mean()
        X['Breadth_Mom'] = X['Breadth_Ratio'].pct_change(63)

    # === RATE REGIME INDICATORS ===
    X['TNX_Level'] = raw_data['^TNX']  # Absolute 10Y yield
    X['TNX_Momentum'] = raw_data['^TNX'].pct_change(63)  # Rising/falling
    X['TNX_MA_Diff'] = raw_data['^TNX'] - raw_data['^TNX'].rolling(200).mean()  # Trend

    # === LIQUIDITY PROXY ===
    if 'TLT' in raw_data.columns:
        X['Bond_Momentum'] = raw_data['TLT'].pct_change(63)  # Rising bonds = falling rates
        X['Bond_Vol'] = raw_data['TLT'].rolling(63).std()

    # === MOMENTUM PERSISTENCE (Are winners staying winners?) ===
    print("⏱️  Computing Momentum Persistence Indicators...")

    for ticker in SECTOR_TICKERS:
        if ticker not in raw_data.columns:
            continue

        # Count consecutive positive/negative periods
        mom_1m = X[f'{ticker}_Mom_1M']

        # Momentum streak (positive periods in last 3 months)
        mom_streak = mom_1m.rolling(3).apply(lambda x: (x > 0).sum(), raw=True)
        X[f'{ticker}_Mom_Streak'] = mom_streak

        # Momentum consistency (standard deviation of returns)
        X[f'{ticker}_Mom_Consistency'] = raw_data[ticker].pct_change(21).rolling(63).std()

    print(f"✅ Created {len(X.columns)} features (including comprehensive sector momentum)")

    # 4. Target Generation - FIXED VERSION
    # Option A: Relative performance (outperformed equal-weight average)
    horizon = 63
    sector_returns = raw_data[SECTOR_TICKERS].pct_change(horizon).shift(-horizon)

    # Calculate equal-weight sector average
    avg_return = sector_returns.mean(axis=1)

    # Label sectors that beat the average
    relative_perf = sector_returns.sub(avg_return, axis=0)

    # Create target: top 3 performers get labeled (more balanced)
    def get_top_performer(row):
        """Get the best performer, but only if meaningfully better"""
        if row.isna().all():
            return np.nan
        # Rank sectors by relative performance
        ranked = row.sort_values(ascending=False)
        # Only label if top performer beats median by >2%
        if ranked.iloc[0] - ranked.median() > 0.02:
            return ranked.index[0]
        else:
            return None  # Neutral/choppy regime

    y_raw = relative_perf.apply(get_top_performer, axis=1)

    # Alternative Option B (Recommended): Use percentile-based labeling
    print("\n🎯 Using Percentile-Based Target Construction...")

    # For each date, label top 3 sectors as "outperformers"
    # This ensures balanced classes over time
    y_balanced = pd.Series(index=sector_returns.index, dtype=object)

    for date in sector_returns.index:
        row = sector_returns.loc[date]
        if row.isna().all():
            continue
        # Get top 3 sectors
        top_3 = row.nlargest(3).index.tolist()
        # Randomly pick one to avoid bias (or use weighted random based on magnitude)
        weights = row[top_3].values
        weights = np.exp(weights * 10)  # Exponential weighting favors best
        weights /= weights.sum()
        y_balanced.loc[date] = np.random.choice(top_3, p=weights)

    y = y_balanced

    # 5. Data Cleaning & Class Balance Check
    valid_data = pd.concat([X, y.rename('target')], axis=1).dropna()

    # CRITICAL: Check class distribution BEFORE training
    print("\n📊 Class Distribution in Training Data:")
    print("-" * 60)
    class_counts = valid_data['target'].value_counts()
    total = len(valid_data)
    for sector, count in class_counts.items():
        pct = count / total * 100
        bar = '█' * int(pct)
        print(f"{sector:<8} | {bar} {count:>4} samples ({pct:>5.1f}%)")
    print("-" * 60)

    # Check for severe imbalance
    max_pct = class_counts.max() / total
    if max_pct > 0.3:
        print(f"\n⚠️  WARNING: '{class_counts.idxmax()}' dominates with {max_pct:.1%} of samples!")
        print("   Consider using SMOTE or adjusting target generation logic.\n")

    # Split: Train on 80% of history, Test on most recent 20%
    split_idx = int(len(valid_data) * 0.8)
    train_df = valid_data.iloc[:split_idx]
    test_df = valid_data.iloc[split_idx:]

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    # 6. Scaling (Fit ONLY on Training Data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model: Random Forest (Regime Switching Logic)
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,  # Increased from 5 to capture momentum interactions
        min_samples_leaf=15,  # Slightly lower to capture nuanced patterns
        min_samples_split=40,  # Prevent overfitting
        max_features='sqrt',  # Prevents any single feature from dominating
        class_weight='balanced',  # Critical: compensates for any remaining imbalance
        random_state=42
    )

    print("\n🧠 Training Random Forest Ensemble...")
    model.fit(X_train_scaled, y_train)

    # 8. Validation: Check prediction distribution
    test_predictions = model.predict(X_test_scaled)
    pred_dist = pd.Series(test_predictions).value_counts()

    print("\n📈 Test Set Prediction Distribution:")
    print("-" * 60)
    for sector, count in pred_dist.items():
        pct = count / len(test_predictions) * 100
        bar = '█' * int(pct)
        print(f"{sector:<8} | {bar} {count:>4} predictions ({pct:>5.1f}%)")
    print("-" * 60)

    # Calculate accuracy
    accuracy = (test_predictions == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.1%}")

    # 9. Feature Importance Audit (The Strategist's View)
    importances = model.feature_importances_
    feature_names = X_train.columns.tolist()
    feature_imp_df = pd.DataFrame({
        'Driver': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n📊 Macro Driver Hierarchy (Why the AI is picking sectors):")
    print("-" * 55)
    for _, row in feature_imp_df.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f"{row['Driver']:<15} | {bar} {row['Importance']:.2%}")
    print("-" * 55)

    # 10. Save Bundle for Dashboard Integration
    bundle = {
        'model': model,
        'scaler': scaler,
        'features': feature_names
    }
    joblib.dump(bundle, 'sector_model.joblib')
    print("\n✅ Training Complete. 'sector_model.joblib' is ready for live use.")

    # Save diagnostics
    diagnostics = {
        'class_distribution': class_counts.to_dict(),
        'test_accuracy': accuracy,
        'prediction_distribution': pred_dist.to_dict(),
        'feature_importance': feature_imp_df.to_dict()
    }
    joblib.dump(diagnostics, 'sector_model_diagnostics.joblib')
    print("📋 Diagnostics saved to 'sector_model_diagnostics.joblib'")


if __name__ == "__main__":
    train_quarterly_model()