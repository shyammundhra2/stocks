import pandas as pd
import numpy as np
from macro.helpers import compute_RSI, compute_ATR
import joblib
import os
from functools import lru_cache

# ----------------------
# Load Bundle (Cached)
# ----------------------
MODEL_PATH = 'trend_model.joblib'


@lru_cache(maxsize=1)
def _load_model_bundle():
    """Load model bundle once and cache it"""
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return (
            bundle.get('model_fast', None),
            bundle.get('model_slow', None),
            bundle.get('scaler', None),
            bundle.get('features', None)
        )
    return None, None, None, None


def get_models():
    """Get cached models"""
    return _load_model_bundle()


# ----------------------
# Use predict_trends for batch processing
# ----------------------
def get_ml_predictions_batch(tickers, friendly_names, use_cache=True):
    """
    Batch predict using predict_trends from predict.py.
    This is more efficient than calling get_ml_confidence repeatedly.

    Args:
        tickers: List of ticker symbols
        friendly_names: Dict mapping tickers to names
        use_cache: Whether to use shared data cache

    Returns:
        Dict mapping ticker to blended confidence score
    """
    try:
        from predict import predict_trends

        result = predict_trends(
            MODEL_PATH,
            tickers,
            friendly_names,
            as_of_date=None,
            use_cache=use_cache
        )

        # Extract just the blended scores
        predictions = result.get('predictions', {})
        return {
            ticker: data.get('blended', 50.0)
            for ticker, data in predictions.items()
        }
    except Exception as e:
        print(f"Error in batch predictions: {e}")
        return {friendly_names.get(t, t): 50.0 for t in tickers}


# ----------------------
# Feature Engineering (Optimized)
# ----------------------
def _compute_features_fast(df: pd.DataFrame):
    """
    Optimized feature computation.
    Pre-computes all rolling windows and indicators in one pass.
    """
    close = df['Close'].squeeze()

    # Pre-compute all rolling means at once (more efficient)
    rolling_50 = close.rolling(50, min_periods=1).mean()
    rolling_200 = close.rolling(200, min_periods=1).mean()

    # Get last values
    last_close = close.iloc[-1]
    s50 = rolling_50.iloc[-1]
    s200 = rolling_200.iloc[-1]

    # Compute indicators
    rsi = compute_RSI(close, 14).iloc[-1]
    atr = compute_ATR(df, 14).iloc[-1]

    # Build feature dict
    feat_dict = {
        'RSI14': rsi,
        'SMA50_dist': (last_close - s50) / s50,
        'SMA200_dist': (last_close - s200) / s200,
    }

    return feat_dict, last_close, atr


# ----------------------
# Regime Logic (Vectorized)
# ----------------------
def _determine_regime(f_conf: float, s_conf: float) -> str:
    """
    Vectorized regime determination logic.
    Uses numpy comparisons for speed.
    """
    # Convert to numpy for vectorized operations
    conditions = np.array([
        (f_conf > 60) & (s_conf > 60),  # Bull regimes
        (f_conf < 40) & (s_conf < 40),  # Bear regime
        (f_conf > 60) & (s_conf < 45),  # Dead cat bounce
    ])

    if conditions[0]:  # Bull
        return "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
    elif conditions[1]:  # Bear
        return "Capitulation/Bear"
    elif conditions[2]:  # Dead cat
        return "Dead Cat Bounce"
    else:
        return "Neutral/Chop"


# ----------------------
# Dual ML Confidence (Optimized)
# ----------------------
def get_dual_ml_confidence(df: pd.DataFrame, debug=False) -> dict:
    """
    Returns Fast and Slow confidence scores + Regime label.
    Optimized version with cached models and vectorized operations.
    """
    try:
        if len(df) < 200:
            return {'fast': 50.0, 'slow': 50.0, 'regime': 'Insufficient Data'}

        # Get cached models
        m_fast, m_slow, scaler, features = get_models()

        if not all([m_fast, m_slow, scaler, features]):
            return {'fast': 50.0, 'slow': 50.0, 'regime': 'No Model'}

        # ----------------------
        # Feature Engineering (Optimized)
        # ----------------------
        feat_dict, last_close, atr = _compute_features_fast(df)

        # Fill missing features with 0 for safety
        X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
        X_live = pd.DataFrame([X_live_dict])

        # Scale features
        X_scaled = scaler.transform(X_live)

        # ----------------------
        # Predict (Vectorized)
        # ----------------------
        # Get class indices once
        fast_idx = list(m_fast.classes_).index(1)
        slow_idx = list(m_slow.classes_).index(1)

        # Predict both models
        p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
        p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

        # Volatility Adjustment (caps at 10% of price)
        vol_adj = 1 - min(atr / last_close, 0.1)
        f_conf = round(p_fast * vol_adj, 1)
        s_conf = round(p_slow * vol_adj, 1)

        # ----------------------
        # Regime Determination (Vectorized)
        # ----------------------
        regime = _determine_regime(f_conf, s_conf)

        if debug:
            print("Features:", X_live_dict)
            print("Scaled features:", X_scaled)
            print(f"Fast prob: {p_fast:.1f}, Slow prob: {p_slow:.1f}, Vol adj: {vol_adj:.2f}")
            print("Regime:", regime)

        return {'fast': f_conf, 'slow': s_conf, 'regime': regime}

    except Exception as e:
        if debug:
            print(f"Error in get_dual_ml_confidence: {e}")
        return {'fast': 50.0, 'slow': 50.0, 'regime': 'Error'}


# ----------------------
# Blended ML Confidence (Optimized)
# ----------------------
def get_ml_confidence(df, debug=False):
    """
    Returns blended confidence score: 70% Slow, 30% Fast.
    Optimized to avoid redundant calculations.
    """
    res = get_dual_ml_confidence(df, debug)

    f_conf = res.get('fast', 50.0)
    s_conf = res.get('slow', 50.0)

    # Weighted blend: 70% Slow, 30% Fast
    # Use float multiplication directly (faster than separate ops)
    blended_score = s_conf * 0.7 + f_conf * 0.3
    return round(blended_score, 1)


# ----------------------
# Batch Processing Support
# ----------------------
def get_ml_confidence_batch(dfs: list, debug=False):
    """
    Process multiple dataframes efficiently.
    Useful when analyzing many assets at once.

    Args:
        dfs: List of DataFrames to process
        debug: Enable debug output

    Returns:
        List of confidence scores
    """
    # Get models once for all predictions
    m_fast, m_slow, scaler, features = get_models()

    if not all([m_fast, m_slow, scaler, features]):
        return [50.0] * len(dfs)

    results = []
    for df in dfs:
        try:
            if len(df) < 200:
                results.append(50.0)
                continue

            # Compute features
            feat_dict, last_close, atr = _compute_features_fast(df)

            # Prepare input
            X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
            X_live = pd.DataFrame([X_live_dict])
            X_scaled = scaler.transform(X_live)

            # Predict
            fast_idx = list(m_fast.classes_).index(1)
            slow_idx = list(m_slow.classes_).index(1)

            p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
            p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

            # Adjust and blend
            vol_adj = 1 - min(atr / last_close, 0.1)
            f_conf = p_fast * vol_adj
            s_conf = p_slow * vol_adj

            blended = round(s_conf * 0.7 + f_conf * 0.3, 1)
            results.append(blended)

        except Exception as e:
            if debug:
                print(f"Batch error: {e}")
            results.append(50.0)

    return results


# ----------------------
# Confidence with Caching
# ----------------------
_CONF_CACHE = {}
_CACHE_TTL = 30  # seconds


def get_ml_confidence_cached(df, symbol=None, debug=False):
    """
    Cached version of get_ml_confidence.
    Use when repeatedly analyzing the same asset.

    Args:
        df: DataFrame to analyze
        symbol: Optional symbol identifier for cache key
        debug: Enable debug output
    """
    import time

    if symbol is None:
        # No caching if no symbol provided
        return get_ml_confidence(df, debug)

    now = time.time()
    cache_key = (symbol, df.index[-1])  # Use symbol + last timestamp

    # Check cache
    if cache_key in _CONF_CACHE:
        value, timestamp = _CONF_CACHE[cache_key]
        if now - timestamp < _CACHE_TTL:
            return value

    # Compute and cache
    result = get_ml_confidence(df, debug)
    _CONF_CACHE[cache_key] = (result, now)

    # Cleanup old entries
    expired = [k for k, (_, ts) in _CONF_CACHE.items() if now - ts >= _CACHE_TTL]
    for k in expired:
        del _CONF_CACHE[k]

    return result