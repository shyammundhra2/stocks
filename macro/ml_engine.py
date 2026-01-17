# macro/ml_engine.py
import numpy as np
import pandas as pd
from macro.helpers import compute_RSI, compute_ATR

def get_ml_confidence(df: pd.DataFrame) -> float:
    """
    Calculates a trade confidence score using an XGBoost-style 
    heuristic based on multivariate features.
    """
    try:
        if len(df) < 50:
            return 50.0

        # Feature 1: RSI Divergence
        rsi = compute_RSI(df['Close'], 14).iloc[-1]
        
        # Feature 2: Volume Flow (Z-Score)
        vol_sm = df['Volume'].rolling(20).mean()
        vol_st = df['Volume'].rolling(20).std()
        vol_z = (df['Volume'].iloc[-1] - vol_sm.iloc[-1]) / vol_st.iloc[-1]
        
        # Feature 3: Trend Positioning
        sma200 = df['Close'].rolling(200).mean().iloc[-1]
        dist_sma = (df['Close'].iloc[-1] - sma200) / sma200 if sma200 else 0

        # XGBoost-style decision tree logic (Heuristic)
        # In a production environment, you would use: model.predict_proba(features)
        score = 0.5  # Neutral base
        
        # Tree 1: Momentum
        if 40 < rsi < 70: score += 0.15
        elif rsi > 80: score -= 0.10 # Overbought risk
        
        # Tree 2: Volume Confirmation
        if vol_z > 1.0: score += 0.15
        
        # Tree 3: Trend Quality
        if dist_sma > 0: score += 0.10
        
        confidence = min(max(score * 100, 0), 99)
        return round(confidence, 1)
    except:
        return 50.0
