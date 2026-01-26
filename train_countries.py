import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from macro.constants import COUNTRIES, ML_MACRO_TICKERS, CURRENCIES

country_tickers = list(COUNTRIES.keys())
macro_tickers = ML_MACRO_TICKERS
currency_tickers = list(CURRENCIES.keys())


def train_country_model_with_fx():
    print("🚀 Training Country Model with Currency Trends...")

    # 1. Download historical data
    data = yf.download(
        country_tickers + macro_tickers + currency_tickers,
        start="2000-01-01"
    )['Close'].ffill()

    # 2. Feature Engineering
    X = pd.DataFrame(index=data.index)

    # Macro features
    X['DXY_mom'] = data['DX-Y.NYB'].pct_change(63)
    X['VIX_level'] = data['^VIX'].rolling(21).mean()
    X['Yield_Curve'] = data['^TYX'] - data['^TNX']
    X['Credit_Spread'] = data['LQD'] / data['HYG']

    # Country-specific features
    for ticker in country_tickers:
        X[f'{ticker}_mom'] = data[ticker].pct_change(63)
        X[f'{ticker}_vol'] = data[ticker].rolling(63).std()
        X[f'{ticker}_mom_1m'] = data[ticker].pct_change(21)
        X[f'{ticker}_vol_1m'] = data[ticker].rolling(21).std()

        # Add currency trends
        country_name = COUNTRIES[ticker]
        fx_pair = None
        for k, v in CURRENCIES.items():
            if v == country_name:
                fx_pair = k
                break

        if fx_pair and fx_pair in data.columns:
            X[f'{ticker}_FX_mom'] = data[fx_pair].pct_change(63)
            X[f'{ticker}_FX_mom_1m'] = data[fx_pair].pct_change(21)

    # 3. Target: best-performing country ETF over next quarter
    horizon = 63
    returns = data[country_tickers].pct_change(horizon).shift(-horizon)
    y = returns.idxmax(axis=1)

    # 4. Clean and scale
    valid = pd.concat([X, y.rename('target')], axis=1).dropna()
    X_train = valid.drop(columns=['target'])
    y_train = valid['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 5. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y_train)

    # ------------------ TOP-K ACCURACY ------------------

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

    print("\n📊 MODEL PERFORMANCE")
    print("===================================")
    print(f"✅ Top-1 Accuracy: {top1_acc:.2%}")
    print(f"✅ Top-3 Accuracy: {top3_acc:.2%}")
    print("===================================\n")

    # 6. Save model
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': X_train.columns.tolist()
    }, 'country_model.joblib')

    print("✅ Country model with FX trends trained!")
    # ------------------ FEATURE IMPORTANCE ------------------
    importances = model.feature_importances_
    feature_ranking = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n============================================================")
    print("🔑 Top Feature Rankings:")
    print("============================================================")
    for i, row in feature_ranking.iterrows():
        print(f"{i + 1:02d}. {row['Feature']:20} {row['Importance']:.4f}")
    print("============================================================\n")


if __name__ == "__main__":
    train_country_model_with_fx()
