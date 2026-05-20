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
# Linear Regression Features
# ----------------------
def _compute_linreg_features(series, period=14):
    slopes = np.full(len(series), np.nan)
    r2s = np.full(len(series), np.nan)
    x = np.arange(period)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    vals = series.values
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1:i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_yy = ((y - y_mean) ** 2).sum()
        slope = ss_xy / ss_xx
        slopes[i] = slope / y_mean
        r2s[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)

    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


# ----------------------
# Feature Engineering
# ----------------------
def _compute_features_fast(df: pd.DataFrame):
    close = df['Close'].squeeze()

    rolling_50 = close.rolling(50, min_periods=1).mean()
    rolling_200 = close.rolling(200, min_periods=1).mean()

    last_close = close.iloc[-1]
    s50 = rolling_50.iloc[-1]
    s200 = rolling_200.iloc[-1]

    rsi = compute_RSI(close, 14).iloc[-1]
    atr = compute_ATR(df, 14).iloc[-1]

    lr_slope, lr_r2 = _compute_linreg_features(close, 14)
    log_returns = np.log(close / close.shift(1))
    real_vol = log_returns.rolling(21).std().iloc[-1] * np.sqrt(252)
    high_low = close.rolling(2).max() - close.rolling(2).min()
    atr21_pct = (high_low.rolling(21).mean() / close).iloc[-1]

    feat_dict = {
        'RSI14':       rsi,
        'SMA50_dist':  (last_close - s50) / s50,
        'SMA200_dist': (last_close - s200) / s200,
        'LR14_slope':  lr_slope.iloc[-1],
        'LR14_r2':     lr_r2.iloc[-1],
        'RealVol21':   real_vol,
        'ATR21_pct':   atr21_pct,
    }

    return feat_dict, last_close, atr


# ----------------------
# Regime Logic
# ----------------------
def _determine_regime(f_conf: float, s_conf: float) -> str:
    if f_conf > 60 and s_conf > 60:
        return "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
    elif f_conf < 40 and s_conf < 40:
        return "Capitulation/Bear"
    elif f_conf > 60 and s_conf < 45:
        return "Dead Cat Bounce"
    else:
        return "Neutral/Chop"


# ----------------------
# Dual ML Confidence
# ----------------------
def get_dual_ml_confidence(df: pd.DataFrame, debug=False) -> dict:
    """
    Returns Fast and Slow confidence scores + Regime label.
    """
    try:
        if len(df) < 200:
            return {'fast': 50.0, 'slow': 50.0, 'regime': 'Insufficient Data'}

        m_fast, m_slow, scaler, features = get_models()

        if not all([m_fast, m_slow, scaler, features]):
            return {'fast': 50.0, 'slow': 50.0, 'regime': 'No Model'}

        feat_dict, last_close, atr = _compute_features_fast(df)

        X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
        X_live = pd.DataFrame([X_live_dict])
        X_scaled = scaler.transform(X_live)

        fast_idx = list(m_fast.classes_).index(1)
        slow_idx = list(m_slow.classes_).index(1)

        p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
        p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

        vol_adj = 1 - min(atr / last_close, 0.1)
        f_conf = round(p_fast * vol_adj, 1)
        s_conf = round(p_slow * vol_adj, 1)

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
# Blended ML Confidence
# ----------------------
def get_ml_confidence(df, debug=False):
    """
    Returns blended confidence score: 70% Slow, 30% Fast.
    """
    res = get_dual_ml_confidence(df, debug)

    f_conf = res.get('fast', 50.0)
    s_conf = res.get('slow', 50.0)

    blended_score = s_conf * 0.7 + f_conf * 0.3
    return round(blended_score, 1)


# ----------------------
# Batch Processing Support
# ----------------------
def get_ml_confidence_batch(dfs: list, debug=False):
    """
    Process multiple dataframes efficiently.
    """
    m_fast, m_slow, scaler, features = get_models()

    if not all([m_fast, m_slow, scaler, features]):
        return [50.0] * len(dfs)

    results = []
    for df in dfs:
        try:
            if len(df) < 200:
                results.append(50.0)
                continue

            feat_dict, last_close, atr = _compute_features_fast(df)

            X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
            X_live = pd.DataFrame([X_live_dict])
            X_scaled = scaler.transform(X_live)

            fast_idx = list(m_fast.classes_).index(1)
            slow_idx = list(m_slow.classes_).index(1)

            p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
            p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

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
    """
    import time

    if symbol is None:
        return get_ml_confidence(df, debug)

    now = time.time()
    cache_key = (symbol, df.index[-1])

    if cache_key in _CONF_CACHE:
        value, timestamp = _CONF_CACHE[cache_key]
        if now - timestamp < _CACHE_TTL:
            return value

    result = get_ml_confidence(df, debug)
    _CONF_CACHE[cache_key] = (result, now)

    expired = [k for k, (_, ts) in _CONF_CACHE.items() if now - ts >= _CACHE_TTL]
    for k in expired:
        del _CONF_CACHE[k]

    return result
