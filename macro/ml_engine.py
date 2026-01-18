import pandas as pd
import numpy as np
from macro.helpers import compute_RSI, compute_ATR
import joblib
import os

# Load Bundle
MODEL_PATH = 'trend_model.joblib'
if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    m_fast = bundle['model_fast']
    m_slow = bundle['model_slow']
    scaler = bundle['scaler']
    features = bundle['features']
else:
    m_fast = m_slow = scaler = features = None

def get_dual_ml_confidence(df: pd.DataFrame, debug=False) -> dict:
    """Returns Fast and Slow confidence scores + Regime label."""
    try:
        if len(df) < 200: return {'fast': 50.0, 'slow': 50.0, 'regime': 'Insufficient Data'}
        
        close = df['Close'].squeeze()
        vol = df['Volume'].squeeze() if 'Volume' in df.columns else pd.Series(0, index=df.index)

        # Feature Engineering
        rsi = compute_RSI(close, 14).iloc[-1]
        atr = compute_ATR(df, 14).iloc[-1]
        s50 = close.rolling(50).mean().iloc[-1]
        s200 = close.rolling(200).mean().iloc[-1]
        
        feat_vals = [
            rsi, 
            atr, 
            (close.iloc[-1]-s50)/s50, 
            (close.iloc[-1]-s200)/s200,
            close.pct_change(5).iloc[-1],
            close.pct_change(10).iloc[-1],
            (vol.iloc[-1]-vol.rolling(20).mean().iloc[-1])/(vol.rolling(20).std().iloc[-1]+1e-6)
        ]

        if m_fast and m_slow and scaler:
            X_live = pd.DataFrame([feat_vals], columns=features)
            X_scaled = scaler.transform(X_live)
            
            # Predict
            p_fast = m_fast.predict_proba(X_scaled)[0][list(m_fast.classes_).index(1)] * 100
            p_slow = m_slow.predict_proba(X_scaled)[0][list(m_slow.classes_).index(1)] * 100
            
            # Volatility Adj
            vol_adj = 1 - min(atr / close.iloc[-1], 0.1)
            f_conf = round(p_fast * vol_adj, 1)
            s_conf = round(p_slow * vol_adj, 1)

            # Regime Determination
            if f_conf > 60 and s_conf > 60:
                regime = "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
            elif f_conf < 40 and s_conf < 40:
                regime = "Capitulation/Bear"
            elif f_conf > 60 and s_conf < 45:
                regime = "Dead Cat Bounce"
            else:
                regime = "Neutral/Chop"

            return {'fast': f_conf, 'slow': s_conf, 'regime': regime}
            
    except Exception as e:
        if debug: print(f"Error: {e}")
        return {'fast': 50.0, 'slow': 50.0, 'regime': 'Error'}

# Backward compatibility alias
def get_ml_confidence(df, debug=False):
    res = get_dual_ml_confidence(df, debug)
    # Return the slow confidence as the default "confidence" score
    return res.get('slow', 50.0)
