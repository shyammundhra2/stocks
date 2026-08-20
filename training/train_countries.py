import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from macro.paths import model_path

import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from macro.constants import COUNTRIES, ML_MACRO_TICKERS, CURRENCIES

# ------------------ CONFIG ------------------
country_tickers = list(COUNTRIES.keys())
currency_tickers = list(CURRENCIES.keys())
macro_tickers = list(set(ML_MACRO_TICKERS + ['^FVX']))


def create_enhanced_features(data, ticker, country_name):
    """Create enhanced feature set for a country"""
    rets = data[ticker].pct_change()
    features = {}

    # === MOMENTUM (multiple timeframes) ===
    features[f'{ticker}_mom_1m'] = rets.rolling(21).sum()
    features[f'{ticker}_mom_3m'] = rets.rolling(63).sum()
    features[f'{ticker}_mom_6m'] = rets.rolling(126).sum()
    features[f'{ticker}_mom_12m'] = rets.rolling(252).sum()

    # === RISK-ADJUSTED MOMENTUM ===
    mean_ret = rets.rolling(63).mean()
    std_ret = rets.rolling(63).std()
    features[f'{ticker}_sharpe_3m'] = (mean_ret / std_ret) * np.sqrt(252)

    # === TREND STRENGTH ===
    features[f'{ticker}_sma_ratio'] = (
            data[ticker].rolling(20).mean() / data[ticker].rolling(63).mean() - 1
    )
    features[f'{ticker}_vs_ma200'] = (
            data[ticker] / data[ticker].rolling(200).mean() - 1
    )

    # === VOLATILITY (capped) ===
    vol_annual = rets.rolling(63).std() * np.sqrt(252)
    vol_long = vol_annual.rolling(252).mean()
    vol_z_raw = vol_annual / vol_long
    features[f'{ticker}_vol_z'] = np.clip(vol_z_raw, 0.5, 2.5)

    # === RELATIVE STRENGTH ===
    spy_rets = data['SPY'].pct_change()
    features[f'{ticker}_rel_spy_3m'] = (
            rets.rolling(63).sum() - spy_rets.rolling(63).sum()
    )

    if ticker not in ['SPY']:
        if ticker in ['EWJ', 'EWG', 'EWU', 'EFA']:
            benchmark = 'EFA'
        else:
            benchmark = 'EEM'

        if benchmark in data.columns:
            bench_rets = data[benchmark].pct_change()
            features[f'{ticker}_rel_region_3m'] = (
                    rets.rolling(63).sum() - bench_rets.rolling(63).sum()
            )

    # === MOMENTUM ACCELERATION ===
    mom_3m = rets.rolling(63).sum()
    features[f'{ticker}_mom_accel'] = mom_3m - mom_3m.shift(21)

    # === DRAWDOWN ===
    cummax = data[ticker].expanding().max()
    drawdown = (data[ticker] / cummax - 1)
    features[f'{ticker}_drawdown'] = drawdown

    # === FX ===
    fx_pair = next((k for k, v in CURRENCIES.items() if v == country_name), None)
    if fx_pair and fx_pair in data.columns:
        fx_rets = data[fx_pair].pct_change()
        usd_rets = data['DX-Y.NYB'].pct_change()

        features[f'{ticker}_FX_rel_3m'] = (
                fx_rets.rolling(63).sum() - usd_rets.rolling(63).sum()
        )
        features[f'{ticker}_FX_trend'] = (
                data[fx_pair].rolling(20).mean() / data[fx_pair].rolling(63).mean() - 1
        )

    return pd.DataFrame(features, index=data.index)


def create_macro_features(data):
    """Create enhanced macro features"""
    X = pd.DataFrame(index=data.index)

    # === USD STRENGTH ===
    usd_rets = data['DX-Y.NYB'].pct_change()
    X['DXY_mom_1m'] = usd_rets.rolling(21).sum()
    X['DXY_mom_3m'] = usd_rets.rolling(63).sum()
    X['DXY_trend'] = (
            data['DX-Y.NYB'].rolling(20).mean() /
            data['DX-Y.NYB'].rolling(63).mean() - 1
    )

    # === VOLATILITY REGIME ===
    vix = data['^VIX']
    X['VIX_level'] = vix.rolling(21).mean()
    X['VIX_percentile'] = vix.rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
    )
    X['VIX_spike'] = (vix > vix.rolling(63).quantile(0.75)).astype(int)

    # === YIELD CURVE ===
    if '^IRX' in data.columns:
        X['Yield_Curve'] = data['^TNX'] - data['^IRX']
    elif '^FVX' in data.columns:
        X['Yield_Curve'] = data['^TNX'] - data['^FVX']
    else:
        raise ValueError("No short-rate treasury available")

    X['Yield_Curve_mom'] = X['Yield_Curve'].diff(21)
    X['TNX_level'] = data['^TNX'].rolling(21).mean()
    X['TNX_mom'] = data['^TNX'].diff(63)

    # === CREDIT SPREADS ===
    lqd_rets = data['LQD'].pct_change()
    X['LQD_mom_3m'] = lqd_rets.rolling(63).sum()

    hyg_rets = data['HYG'].pct_change()
    X['HYG_mom_3m'] = hyg_rets.rolling(63).sum()

    X['Credit_Spread'] = X['LQD_mom_3m'] - X['HYG_mom_3m']

    # === RISK APPETITE ===
    vix_normalized = -vix / vix.rolling(252).mean()
    hyg_normalized = data['HYG'].pct_change().rolling(63).sum()
    X['Risk_Appetite'] = (vix_normalized + hyg_normalized) / 2

    # === COMMODITIES ===
    if 'GLD' in data.columns:
        X['Gold_mom_3m'] = data['GLD'].pct_change().rolling(63).sum()

    if 'USO' in data.columns:
        X['Oil_mom_3m'] = data['USO'].pct_change().rolling(63).sum()

    return X


def train_gb_only_model():
    print("🚀 Training GB-Only Country Model (Optimized for 60%+ Top-3)")
    print("=" * 60)

    # ------------------ DATA DOWNLOAD ------------------
    all_tickers = list(set(country_tickers + macro_tickers + currency_tickers + ['GLD', 'USO']))

    print("\n📥 Downloading data...")
    data = yf.download(
        all_tickers,
        start="2000-01-01",
        auto_adjust=True,
        progress=False
    )['Close'].ffill()

    print(f"✅ Downloaded {len(data)} trading days")

    # ------------------ BUILD FEATURES ------------------
    print("\n🔧 Building enhanced features...")

    X = create_macro_features(data)

    for ticker in country_tickers:
        country_name = COUNTRIES[ticker]
        country_features = create_enhanced_features(data, ticker, country_name)
        X = pd.concat([X, country_features], axis=1)

    print(f"✅ Created {len(X.columns)} features")

    # ------------------ TARGET ------------------
    horizon = 63
    fwd_returns = data[country_tickers].pct_change(horizon).shift(-horizon)
    y = fwd_returns.idxmax(axis=1)

    # ------------------ CLEAN ------------------
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_all = valid.drop(columns='target')
    y_all = valid['target']

    print(f"✅ Training samples: {len(y_all)}")
    print(f"✅ Date range: {y_all.index[0].date()} to {y_all.index[-1].date()}")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ------------------ OPTIMIZED GB MODEL ------------------
    print("\n🤖 Training optimized Gradient Boosting model...")

    # Tuned hyperparameters for better performance
    model = GradientBoostingClassifier(
        n_estimators=400,  # More trees
        max_depth=5,  # Slightly deeper
        learning_rate=0.03,  # Lower learning rate
        subsample=0.8,  # Keep subsampling
        max_features='sqrt',  # Reduce overfitting
        min_samples_leaf=8,  # Lower threshold
        random_state=42
    )

    # ------------------ WALK-FORWARD CV ------------------
    print("\n📊 Running walk-forward validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    top1_scores, top3_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled), 1):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_tr, y_te = y_all.iloc[train_idx], y_all.iloc[test_idx]

        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        classes = model.classes_

        # Top-1
        preds = classes[np.argmax(proba, axis=1)]
        top1_scores.append(accuracy_score(y_te, preds))

        # Top-3
        correct = sum(
            y_te.iloc[i] in classes[np.argsort(proba[i])[-3:]]
            for i in range(len(y_te))
        )
        top3_acc = correct / len(y_te)
        top3_scores.append(top3_acc)

        print(f"  Fold {fold}: Top-1 = {top1_scores[-1]:.2%}, Top-3 = {top3_acc:.2%}")

    print("\n" + "=" * 60)
    print("📊 WALK-FORWARD PERFORMANCE")
    print("=" * 60)
    print(f"Top-1 Accuracy: {np.mean(top1_scores):.2%} (±{np.std(top1_scores):.2%})")
    print(f"Top-3 Accuracy: {np.mean(top3_scores):.2%} (±{np.std(top3_scores):.2%})")
    print("=" * 60)

    # ------------------ FINAL FIT ------------------
    print("\n🎯 Training final model on all data...")
    model.fit(X_scaled, y_all)

    joblib.dump(
        {
            'model': model,
            'scaler': scaler,
            'features': X_all.columns.tolist()
        },
        model_path('country_model.joblib')
    )

    print("✅ Model saved to 'country_model.joblib'")

    # ------------------ FEATURE IMPORTANCE ------------------
    importance = (
        pd.DataFrame({
            'Feature': X_all.columns,
            'Importance': model.feature_importances_
        })
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("🔑 TOP 25 FEATURES")
    print("=" * 60)
    for i, row in importance.head(25).iterrows():
        print(f"{i + 1:02d}. {row['Feature']:35} {row['Importance']:.4f}")
    print("=" * 60)

    if np.mean(top3_scores) >= 0.60:
        print("\n🎉 TARGET ACHIEVED! Top-3 accuracy >= 60%")
    else:
        print(f"\n📊 Top-3 accuracy: {np.mean(top3_scores):.2%}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    train_gb_only_model()