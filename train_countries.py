import yfinance as yf
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from macro.constants import COUNTRIES, ML_MACRO_TICKERS, CURRENCIES


# ------------------ CONFIG ------------------
country_tickers = list(COUNTRIES.keys())
currency_tickers = list(CURRENCIES.keys())

# Ensure fallback short-rate is available
macro_tickers = list(set(ML_MACRO_TICKERS + ['^FVX']))  # 2Y fallback


def train_country_model_with_fx():
    print("🚀 Training Country Model with FX (PRODUCTION SAFE)")

    # ------------------ DATA DOWNLOAD ------------------
    all_tickers = country_tickers + macro_tickers + currency_tickers

    data = yf.download(
        all_tickers,
        start="2000-01-01",
        auto_adjust=True,
        progress=False
    )['Close'].ffill()

    # ------------------ FEATURE FRAME ------------------
    X = pd.DataFrame(index=data.index)

    # ------------------ MACRO FEATURES ------------------
    # USD momentum
    X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63)

    # Volatility regime
    X['VIX_level'] = data['^VIX'].rolling(21).mean()

    # ---- Yield Curve (robust) ----
    if '^IRX' in data.columns:
        # 10Y – 3M (best)
        X['Yield_Curve'] = data['^TNX'] - data['^IRX']
    elif '^FVX' in data.columns:
        # 10Y – 2Y (fallback)
        X['Yield_Curve'] = data['^TNX'] - data['^FVX']
    else:
        raise ValueError("No short-rate treasury (^IRX or ^FVX) available")

    # ---- Credit Spread (stationary) ----
    X['Credit_Spread'] = (
        data['LQD'].pct_change(63) -
        data['HYG'].pct_change(63)
    )

    # ------------------ COUNTRY + FX FEATURES ------------------
    for ticker in country_tickers:
        rets = data[ticker].pct_change()

        # Momentum
        X[f'{ticker}_mom_3m'] = rets.rolling(63).sum()
        X[f'{ticker}_mom_1m'] = rets.rolling(21).sum()

        # Volatility (normalized regime)
        vol = rets.rolling(63).std() * np.sqrt(252)
        X[f'{ticker}_vol_z'] = vol / vol.rolling(252).mean()

        # FX relative to USD
        country_name = COUNTRIES[ticker]
        fx_pair = next((k for k, v in CURRENCIES.items() if v == country_name), None)

        if fx_pair and fx_pair in data.columns:
            X[f'{ticker}_FX_rel'] = (
                data[fx_pair].pct_change(63) -
                data['DX-Y.NYB'].pct_change(63)
            )

    # ------------------ TARGET ------------------
    horizon = 63  # ~1 quarter
    fwd_returns = data[country_tickers].pct_change(horizon).shift(-horizon)
    y = fwd_returns.idxmax(axis=1)

    # ------------------ CLEAN ------------------
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_all = valid.drop(columns='target')
    y_all = valid['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ------------------ MODEL ------------------
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # ------------------ WALK-FORWARD CV ------------------
    tscv = TimeSeriesSplit(n_splits=5)
    top1_scores, top3_scores = [], []

    for train_idx, test_idx in tscv.split(X_scaled):
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
        top3_scores.append(correct / len(y_te))

    print("\n📊 WALK-FORWARD PERFORMANCE")
    print("===================================")
    print(f"✅ Top-1 Accuracy: {np.mean(top1_scores):.2%}")
    print(f"✅ Top-3 Accuracy: {np.mean(top3_scores):.2%}")
    print("===================================\n")

    # ------------------ FINAL FIT ------------------
    model.fit(X_scaled, y_all)

    joblib.dump(
        {
            'model': model,
            'scaler': scaler,
            'features': X_all.columns.tolist()
        },
        'country_model.joblib'
    )

    # ------------------ FEATURE IMPORTANCE ------------------
    importance = (
        pd.DataFrame({
            'Feature': X_all.columns,
            'Importance': model.feature_importances_
        })
        .sort_values('Importance', ascending=False)
        .reset_index(drop=True)
    )

    print("🔑 TOP FEATURE IMPORTANCE")
    print("===================================")
    for i, row in importance.head(20).iterrows():
        print(f"{i+1:02d}. {row['Feature']:30} {row['Importance']:.4f}")
    print("===================================")

    print("✅ Country model trained and saved successfully.")


if __name__ == "__main__":
    train_country_model_with_fx()
