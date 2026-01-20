import pandas as pd
import numpy as np
from macro.helpers import compute_RSI, compute_ATR
import joblib
import os

# ----------------------
# Load Bundle
# ----------------------
MODEL_PATH = 'trend_model.joblib'
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    m_fast = bundle.get('model_fast', None)
    m_slow = bundle.get('model_slow', None)
    scaler = bundle.get('scaler', None)
    features = bundle.get('features', None)
else:
    m_fast = m_slow = scaler = features = None

# ----------------------
# Dual ML Confidence
# ----------------------
def get_dual_ml_confidence(df: pd.DataFrame, debug=False) -> dict:
    """Returns Fast and Slow confidence scores + Regime label."""
    try:
        if len(df) < 200:
            return {'fast': 50.0, 'slow': 50.0, 'regime': 'Insufficient Data'}

        close = df['Close'].squeeze()
        vol = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(0, index=df.index)

        # ----------------------
        # Feature Engineering
        # ----------------------
        rsi = compute_RSI(close, 14).iloc[-1]
        atr = compute_ATR(df, 14).iloc[-1]
        s50 = close.rolling(50).mean().iloc[-1]
        s200 = close.rolling(200).mean().iloc[-1]

        # Basic trend features
        feat_dict = {
            'RSI14': rsi,
            'SMA50_dist': (close.iloc[-1]-s50)/s50,
            'SMA200_dist': (close.iloc[-1]-s200)/s200,
        }

        # Fill missing features with 0 for safety
        if features:
            X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
            X_live = pd.DataFrame([X_live_dict])

            if scaler:
                X_scaled = scaler.transform(X_live)

                # ----------------------
                # Predict
                # ----------------------
                p_fast = m_fast.predict_proba(X_scaled)[0][list(m_fast.classes_).index(1)] * 100
                p_slow = m_slow.predict_proba(X_scaled)[0][list(m_slow.classes_).index(1)] * 100

                # Volatility Adjustment (caps at 10% of price)
                vol_adj = 1 - min(atr / close.iloc[-1], 0.1)
                f_conf = round(p_fast * vol_adj, 1)
                s_conf = round(p_slow * vol_adj, 1)

                # ----------------------
                # Regime Determination
                # ----------------------
                if f_conf > 60 and s_conf > 60:
                    regime = "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
                elif f_conf < 40 and s_conf < 40:
                    regime = "Capitulation/Bear"
                elif f_conf > 60 and s_conf < 45:
                    regime = "Dead Cat Bounce"
                else:
                    regime = "Neutral/Chop"

                if debug:
                    print("Features:", X_live_dict)
                    print("Scaled features:", X_scaled)
                    print(f"Fast prob: {p_fast:.1f}, Slow prob: {p_slow:.1f}, Vol adj: {vol_adj:.2f}")
                    print("Regime:", regime)

                return {'fast': f_conf, 'slow': s_conf, 'regime': regime}

        # Fallback if models not loaded
        return {'fast': 50.0, 'slow': 50.0, 'regime': 'No Model'}

    except Exception as e:
        if debug:
            print(f"Error in get_dual_ml_confidence: {e}")
        return {'fast': 50.0, 'slow': 50.0, 'regime': 'Error'}

# ----------------------
# Blended ML Confidence
# ----------------------
def get_ml_confidence(df, debug=False):
    res = get_dual_ml_confidence(df, debug)

    f_conf = res.get('fast', 50.0)
    s_conf = res.get('slow', 50.0)

    # Weighted blend: 70% Slow, 30% Fast
    blended_score = (s_conf * 0.7) + (f_conf * 0.3)
    return round(blended_score, 1)
