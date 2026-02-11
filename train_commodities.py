import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==================== SECTOR MAPPING ====================
SECTOR_MAP = {
    'GC=F': 'Precious_Metals',  # Gold
    'SI=F': 'Precious_Metals',  # Silver
    'PL=F': 'Precious_Metals',  # Platinum
    'PA=F': 'Precious_Metals',  # Palladium

    'HG=F': 'Industrial_Metals',  # Copper
    'ALI=F': 'Industrial_Metals', # Aluminum (if available)

    'CL=F': 'Energy',  # Crude Oil
    'NG=F': 'Energy',  # Natural Gas
    'RB=F': 'Energy',  # Gasoline (if available)

    'ZC=F': 'Grains',  # Corn
    'ZS=F': 'Grains',  # Soybeans
    'ZW=F': 'Grains',  # Wheat

    'KC=F': 'Softs',   # Coffee
    'SB=F': 'Softs',   # Sugar
    'CT=F': 'Softs',   # Cotton
    'CC=F': 'Softs',   # Cocoa (if available)
}

# Commodity tickers (adjust based on your constants)
COMMODITY_TICKERS = [
    'GC=F', 'SI=F', 'PL=F', 'HG=F',  # Metals
    'CL=F', 'NG=F',  # Energy
    'ZC=F', 'ZS=F', 'ZW=F',  # Grains
    'KC=F', 'SB=F', 'CT=F',  # Softs
]

# Macro tickers
MACRO_TICKERS = [
    'DX-Y.NYB',  # Dollar Index
    '^VIX',      # Volatility
    '^TNX',      # 10-Year Treasury
    '^IRX',      # 13-Week Treasury (for yield curve)
]


def get_sector(ticker):
    """Map commodity ticker to sector"""
    return SECTOR_MAP.get(ticker, 'Other')


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def verify_no_lookahead(feature_date, return_start_date, return_end_date, horizon):
    """Verify no lookahead bias"""
    if pd.isna(feature_date) or pd.isna(return_start_date) or pd.isna(return_end_date):
        return False
    if feature_date >= return_start_date:
        return False
    return True


def create_enhanced_features(data, commodity_tickers, macro_tickers, current_date):
    """
    Create features with cross-sectional analysis and relative performance
    """
    historical_data = data.loc[:current_date].copy()
    if len(historical_data) < 252:
        return None

    X = pd.Series(dtype=float)

    # ==================== MACRO FEATURES ====================
    if 'DX-Y.NYB' in historical_data.columns:
        dxy = historical_data['DX-Y.NYB'].dropna()
        if len(dxy) >= 63:
            X['DXY_mom_3m'] = dxy.pct_change(63).iloc[-1]
            X['DXY_mom_1m'] = dxy.pct_change(21).iloc[-1]
            X['DXY_trend'] = int(dxy.iloc[-1] > dxy.rolling(63).mean().iloc[-1])

    if '^VIX' in historical_data.columns:
        vix = historical_data['^VIX'].dropna()
        if len(vix) >= 252:
            X['VIX_level'] = vix.rolling(21).mean().iloc[-1]
            X['VIX_regime'] = int(vix.iloc[-1] > vix.rolling(252).mean().iloc[-1])

    if '^TNX' in historical_data.columns and '^IRX' in historical_data.columns:
        tnx = historical_data['^TNX'].dropna()
        irx = historical_data['^IRX'].dropna()
        if len(tnx) >= 21 and len(irx) >= 21:
            # Yield curve slope
            X['Yield_Curve'] = tnx.iloc[-1] - irx.iloc[-1]

    # ==================== COMMODITY ABSOLUTE FEATURES ====================
    commodity_returns_3m = {}
    commodity_returns_1m = {}
    commodity_vols_3m = {}

    for ticker in commodity_tickers:
        if ticker not in historical_data.columns:
            continue

        s = historical_data[ticker].dropna()
        if len(s) < 63:
            continue

        # Calculate returns
        ret_3m = s.pct_change(63).iloc[-1]
        ret_1m = s.pct_change(21).iloc[-1]
        vol_3m = s.pct_change(1).tail(63).std()

        # Store for cross-sectional analysis
        commodity_returns_3m[ticker] = ret_3m
        commodity_returns_1m[ticker] = ret_1m
        commodity_vols_3m[ticker] = vol_3m

        # Absolute features
        X[f'{ticker}_mom_3m'] = ret_3m
        X[f'{ticker}_mom_1m'] = ret_1m
        X[f'{ticker}_vol_3m'] = vol_3m
        X[f'{ticker}_vol_1m'] = s.rolling(21).std().iloc[-1]
        X[f'{ticker}_sma_ratio'] = s.iloc[-1] / s.rolling(63).mean().iloc[-1]

    # ==================== CROSS-SECTIONAL FEATURES ====================
    # Compare each commodity to peer group average

    if len(commodity_returns_3m) > 0:
        mean_ret_3m = np.mean(list(commodity_returns_3m.values()))
        mean_ret_1m = np.mean(list(commodity_returns_1m.values()))
        mean_vol_3m = np.mean(list(commodity_vols_3m.values()))

        for ticker in commodity_returns_3m.keys():
            # Relative momentum (vs all commodities)
            X[f'{ticker}_mom_3m_vs_all'] = commodity_returns_3m[ticker] - mean_ret_3m
            X[f'{ticker}_mom_1m_vs_all'] = commodity_returns_1m[ticker] - mean_ret_1m

            # Relative volatility
            X[f'{ticker}_vol_3m_vs_all'] = commodity_vols_3m[ticker] - mean_vol_3m

            # Risk-adjusted momentum
            if commodity_vols_3m[ticker] > 0:
                X[f'{ticker}_risk_adj_mom'] = commodity_returns_3m[ticker] / commodity_vols_3m[ticker]

    # ==================== SECTOR-LEVEL FEATURES ====================
    # Compare each commodity to its sector peers

    sector_returns_3m = {}
    for ticker, ret in commodity_returns_3m.items():
        sector = get_sector(ticker)
        if sector not in sector_returns_3m:
            sector_returns_3m[sector] = []
        sector_returns_3m[sector].append(ret)

    # Calculate sector averages
    sector_avg_3m = {sector: np.mean(rets) for sector, rets in sector_returns_3m.items()}

    for ticker in commodity_returns_3m.keys():
        sector = get_sector(ticker)
        if sector in sector_avg_3m:
            # Momentum vs sector peers
            X[f'{ticker}_mom_3m_vs_sector'] = commodity_returns_3m[ticker] - sector_avg_3m[sector]

    # Overall sector momentum
    for sector, avg_ret in sector_avg_3m.items():
        X[f'Sector_{sector}_mom_3m'] = avg_ret

    # ==================== CORRELATION FEATURES ====================
    # When correlations are low, stock picking matters more

    if len(commodity_tickers) >= 3:
        recent_prices = historical_data[commodity_tickers].tail(63).dropna(axis=1)
        if len(recent_prices.columns) >= 3:
            corr_matrix = recent_prices.corr()
            # Average correlation (excluding diagonal)
            triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
            X['commodity_corr_avg'] = corr_matrix.values[triu_indices].mean()

            # Correlation dispersion (when high, some commodities are diverging)
            X['commodity_corr_std'] = corr_matrix.values[triu_indices].std()

    # ==================== SEASONALITY ====================
    month = current_date.month

    for ticker in commodity_tickers:
        if ticker not in historical_data.columns:
            continue

        s = historical_data[ticker].dropna()
        if len(s) < 252:
            continue

        # Calculate historical average return for this month
        monthly_rets = s.pct_change(21)
        same_month_rets = monthly_rets[monthly_rets.index.month == month]

        if len(same_month_rets) >= 3:
            X[f'{ticker}_seasonal_avg'] = same_month_rets.mean()
            X[f'{ticker}_seasonal_std'] = same_month_rets.std()

    return X


def calculate_forward_returns(data, commodity_tickers, start_date, horizon):
    """Calculate forward returns for all commodities"""
    if start_date not in data.index:
        return None, None, None, None

    idx = data.index.get_loc(start_date)
    if idx + horizon + 1 >= len(data):
        return None, None, None, None

    start_idx = idx + 1
    end_idx = idx + horizon + 1

    start_prices = data[commodity_tickers].iloc[start_idx]
    end_prices = data[commodity_tickers].iloc[end_idx]

    returns = (end_prices - start_prices) / start_prices

    # Map to sectors
    sector_returns = {}
    for ticker, ret in returns.items():
        sector = get_sector(ticker)
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(ret)

    # Average return per sector
    sector_avg_returns = {sector: np.mean(rets) for sector, rets in sector_returns.items()}

    return returns, sector_avg_returns, data.index[start_idx], data.index[end_idx]


def train_sector_commodity_model(
    start_date="2000-01-01",
    horizon=63,
    n_estimators=1000,
    max_depth=7,
    min_samples_leaf=10,
    min_train_size=252,
    step_size=5,
    predict_sectors=True  # NEW: Toggle between sector and commodity prediction
):

    print("🚀 Training Enhanced Commodity Model")
    print(f"   Prediction Target: {'SECTORS' if predict_sectors else 'INDIVIDUAL COMMODITIES'}")
    print(f"   Sectors: {set(SECTOR_MAP.values())}")

    # Download data
    all_tickers = COMMODITY_TICKERS + MACRO_TICKERS
    raw_data = yf.download(all_tickers, start=start_date, progress=False)['Close']
    data = raw_data.ffill(limit=5)

    results = []
    feature_list = None

    print("\n📊 Building dataset...")
    for i in range(min_train_size, len(data) - horizon - 1, step_size):
        current_date = data.index[i]

        features = create_enhanced_features(
            data, COMMODITY_TICKERS, MACRO_TICKERS, current_date
        )
        if features is None:
            continue

        returns, sector_returns, rs, re = calculate_forward_returns(
            data, COMMODITY_TICKERS, current_date, horizon
        )
        if returns is None:
            continue

        if not verify_no_lookahead(current_date, rs, re, horizon):
            continue

        # Determine target
        if predict_sectors:
            # Best performing sector
            target = max(sector_returns.items(), key=lambda x: x[1])[0]
        else:
            # Best performing commodity
            target = returns.idxmax()

        results.append({
            'features': features,
            'target': target,
            'commodity_returns': returns,
            'sector_returns': sector_returns if predict_sectors else None
        })

        if feature_list is None:
            feature_list = features.index.tolist()

    # Create feature matrix
    X = pd.DataFrame(
        [[r['features'].get(f, np.nan) for f in feature_list] for r in results],
        columns=feature_list
    )
    y = pd.Series([r['target'] for r in results])

    if predict_sectors:
        returns_df = pd.DataFrame([r['sector_returns'] for r in results])
    else:
        returns_df = pd.DataFrame([r['commodity_returns'] for r in results])

    # Remove rows with NaN
    mask = ~X.isna().any(axis=1)
    X, y, returns_df = X[mask], y[mask], returns_df[mask]

    print(f"\n✅ Dataset created:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {len(y.unique())}")
    print(f"\n   Class distribution:")
    for target, count in y.value_counts().sort_index().items():
        print(f"      {target}: {count} ({count/len(y)*100:.1f}%)")

    # ==================== WALK-FORWARD CV ====================
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_acc = []
    cv_top3 = []
    cv_sharpe = []

    print(f"\n🔄 Cross-validation ({n_splits} folds)...")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        ret_test = returns_df.iloc[test_idx]

        # No scaling needed for Random Forest (tree-based)
        model = RandomForestClassifier(
            n_estimators=200,  # Use consistent params
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Top-1 predictions
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cv_acc.append(acc)

        # Top-3 accuracy
        proba = model.predict_proba(X_test)
        classes = model.classes_
        top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
        top3_labels = [[classes[i] for i in row] for row in top3_idx]

        top3_hits = [y_test.iloc[i] in top3_labels[i] for i in range(len(y_test))]
        top3_acc = np.mean(top3_hits)
        cv_top3.append(top3_acc)

        # Sharpe ratio of predictions
        pred_returns = []
        for i in range(len(preds)):
            pred_returns.append(ret_test[preds[i]].iloc[i])

        sharpe = calculate_sharpe_ratio(pd.Series(pred_returns))
        cv_sharpe.append(sharpe)

        print(f"   Fold {fold_idx+1}: Top-1={acc:.2%}, Top-3={top3_acc:.2%}, Sharpe={sharpe:.2f}")

    # Random baseline
    n_classes = len(y.unique())
    random_top1 = 1 / n_classes
    random_top3 = min(3, n_classes) / n_classes

    print(f"\n📈 CV RESULTS:")
    print(f"   Top-1 Accuracy: {np.mean(cv_acc):.2%} (random: {random_top1:.2%})")
    print(f"   Top-3 Accuracy: {np.mean(cv_top3):.2%} (random: {random_top3:.2%})")
    print(f"   Avg Sharpe:     {np.mean(cv_sharpe):.2f}")

    # ==================== TRAIN FINAL MODEL ====================
    print("\n🎯 Training final model on all data...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Feature importance
    print("\n📊 TOP 20 FEATURE IMPORTANCES:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("-" * 70)
    for i, row in feature_importance.head(20).iterrows():
        bar = '█' * int(row['importance'] * 200)
        print(f"   {row['feature']:<30} | {bar} {row['importance']:.4f}")
    print("-" * 70)

    # Save model
    model_name = 'commodity_sector_model.joblib' if predict_sectors else 'commodity_model.joblib'

    joblib.dump({
        'model': model,
        'features': X.columns.tolist(),
        'classes': model.classes_.tolist(),
        'predict_sectors': predict_sectors,
        'sector_map': SECTOR_MAP,
        'cv_top1': np.mean(cv_acc),
        'cv_top3': np.mean(cv_top3),
        'cv_sharpe': np.mean(cv_sharpe),
        'feature_importance': feature_importance.to_dict('records'),
        'training_date': datetime.now().isoformat()
    }, model_name)

    print(f"\n✅ Model saved: {model_name}")

    return model, feature_importance


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRAINING SECTOR-BASED MODEL (5 classes)")
    print("="*70)
    model_sector, fi_sector = train_sector_commodity_model(
        predict_sectors=True,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10
    )

    print("\n\n" + "="*70)
    print("TRAINING COMMODITY-BASED MODEL (13 classes)")
    print("="*70)
    model_commodity, fi_commodity = train_sector_commodity_model(
        predict_sectors=False,
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10
    )