import pandas as pd
import numpy as np
try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:  # numpy < 1.20
    sliding_window_view = None
from macro.helpers import compute_RSI, compute_ATR
import joblib
import os
import warnings
from functools import lru_cache

# Per-prediction inference runs single-row on shallow tree ensembles. The
# models were trained with n_jobs=-1, which makes sklearn fan each predict
# out through joblib and emit the `sklearn.utils.parallel.delayed` UserWarning
# once per call. _load_model_bundle forces n_jobs=1 below (identical outputs,
# no dispatch); this filter additionally keeps the console clean across
# sklearn versions, since formatting+writing hundreds of warnings to stderr
# is itself a measurable chunk of dashboard load time.
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

MODEL_PATH = 'trend_model.joblib'


@lru_cache(maxsize=1)
def _load_model_bundle():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        model_fast = bundle.get('model_fast', None)
        model_slow = bundle.get('model_slow', None)

        # Force single-threaded inference (2026-06-18): models were saved
        # with n_jobs=-1 from training. For predict_proba on a single row,
        # the joblib parallel dispatch dwarfs the actual per-tree compute
        # (shallow trees) and triggers the repeated
        # `sklearn.utils.parallel.delayed` UserWarning on every call -
        # incurred ~33x per dashboard load via get_dual_ml_confidence_for_kelly.
        # n_jobs changes execution strategy only, not results: predictions
        # are identical, just computed sequentially without dispatch.
        # (Mirrors the same fix already in predict._get_model_bundle.)
        for est in (model_fast, model_slow):
            if est is not None and hasattr(est, 'n_jobs'):
                est.n_jobs = 1

        return (
            model_fast,
            model_slow,
            bundle.get('scaler', None),
            bundle.get('features', None)
        )
    return None, None, None, None


def get_models():
    return _load_model_bundle()


def get_ml_predictions_batch(tickers, friendly_names, use_cache=True):
    try:
        from predict import predict_trends
        result = predict_trends(MODEL_PATH, tickers, friendly_names, as_of_date=None, use_cache=use_cache)
        predictions = result.get('predictions', {})
        return {ticker: data.get('blended', 50.0) for ticker, data in predictions.items()}
    except Exception as e:
        print(f"Error in batch predictions: {e}")
        return {friendly_names.get(t, t): 50.0 for t in tickers}


def _compute_linreg_features(series, period=14):
    # Rolling degree-1 regression slope (normalized by window mean) and r2.
    # The heavy per-window reductions (window mean and the two centered
    # sums) are vectorized with a sliding window; they reproduce the old
    # per-window scalar reductions bit-for-bit. The final cheap combine is
    # kept as scalar float64 arithmetic because an array elementwise combine
    # can diverge from the scalar result by ~1 ULP via SIMD codepaths, and
    # the r2 series must stay exactly identical. Verified 0 mismatches vs the
    # original loop across random walks, NaN-injection, and edge sizes
    # (2026-06-17). NaN windows are left NaN exactly as the old `continue` did.
    vals = series.values
    n = len(vals)
    slopes = np.full(n, np.nan)
    r2s = np.full(n, np.nan)
    if n >= period:
        x = np.arange(period)
        x_mean = x.mean()
        xc = x - x_mean
        ss_xx = (xc ** 2).sum()
        if sliding_window_view is not None:
            windows = sliding_window_view(vals, period)
        else:  # numpy < 1.20 fallback
            windows = np.stack([vals[k:k + period] for k in range(n - period + 1)])
        y_mean = windows.mean(axis=1)
        centered = windows - y_mean[:, None]
        ss_xy = (xc[None, :] * centered).sum(axis=1)
        ss_yy = (centered ** 2).sum(axis=1)
        m = windows.shape[0]
        out_slope = np.empty(m)
        out_r2 = np.empty(m)
        for k in range(m):
            ym = y_mean[k]
            if ym != ym:  # NaN window -> original skipped, leaving NaN
                out_slope[k] = np.nan
                out_r2[k] = np.nan
                continue
            sxy = ss_xy[k]
            syy = ss_yy[k]
            slope = sxy / ss_xx
            out_slope[k] = slope / ym
            out_r2[k] = (sxy ** 2) / (ss_xx * syy + 1e-9)
        slopes[period - 1:] = out_slope
        r2s[period - 1:] = out_r2
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def _compute_features_fast(df: pd.DataFrame):
    """
    Compute model features with 3-day EMA smoothing on RSI14 and LR14_slope
    to reduce single-session noise at the source.
    ATR also smoothed for vol_adj stability.
    """
    close = df['Close'].squeeze()
    rolling_50 = close.rolling(50, min_periods=1).mean()
    rolling_200 = close.rolling(200, min_periods=1).mean()
    last_close = close.iloc[-1]
    s50 = rolling_50.iloc[-1]
    s200 = rolling_200.iloc[-1]

    rsi_series = compute_RSI(close, 14)
    rsi = rsi_series.ewm(span=3, adjust=False).mean().iloc[-1]

    # Compute the ATR series once and hand it back to the caller; callers need
    # the full series for 3-day EMA smoothing, so returning it here avoids a
    # second identical compute_ATR(df, 14) call downstream. (The previous
    # scalar `atr` return value was discarded by every caller.)
    atr_series = compute_ATR(df, 14)

    lr_slope_series, lr_r2_series = _compute_linreg_features(close, 14)
    lr_slope = lr_slope_series.ewm(span=3, adjust=False).mean().iloc[-1]
    lr_r2 = lr_r2_series.iloc[-1]

    log_returns = np.log(close / close.shift(1))
    real_vol = log_returns.rolling(21).std().iloc[-1] * np.sqrt(252)
    high_low = close.rolling(2).max() - close.rolling(2).min()
    atr21_pct = (high_low.rolling(21).mean() / close).iloc[-1]

    feat_dict = {
        'RSI14':       rsi,
        'SMA50_dist':  (last_close - s50) / s50,
        'SMA200_dist': (last_close - s200) / s200,
        'LR14_slope':  lr_slope,
        'LR14_r2':     lr_r2,
        'RealVol21':   real_vol,
        'ATR21_pct':   atr21_pct,
    }
    return feat_dict, last_close, atr_series


def _determine_regime(f_conf: float, s_conf: float) -> str:
    """Classify slow/fast model agreement pattern."""
    divergence = abs(s_conf - f_conf)
    if f_conf > 60 and s_conf > 60:
        return "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
    elif f_conf < 40 and s_conf < 40:
        return "Capitulation/Bear"
    elif f_conf > 60 and s_conf < 45:
        return "Dead Cat Bounce"
    elif s_conf > f_conf and s_conf > 50 and divergence > 15:
        return "Structural Bull — Fading Momentum"
    elif f_conf > s_conf and f_conf > 50 and s_conf < 50:
        return "Recovering — Monitor"
    elif s_conf > 50 and f_conf > 50 and divergence < 10:
        return "Structural Bull — Agreement"
    else:
        return "Neutral/Chop"


def _run_dual_model(df: pd.DataFrame, debug=False) -> dict:
    """Internal: compute fast and slow raw confidences with smoothed vol_adj."""
    if len(df) < 200:
        return {'fast': 50.0, 'slow': 50.0, 'last_close': 0.0}

    m_fast, m_slow, scaler, features = get_models()
    if not all([m_fast, m_slow, scaler, features]):
        return {'fast': 50.0, 'slow': 50.0, 'last_close': 0.0}

    feat_dict, last_close, atr_series = _compute_features_fast(df)
    X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
    X_live = pd.DataFrame([X_live_dict])
    X_scaled = scaler.transform(X_live)

    fast_idx = list(m_fast.classes_).index(1)
    slow_idx = list(m_slow.classes_).index(1)

    p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
    p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

    # Smooth ATR over 3 days before vol adjustment (series reused from
    # _compute_features_fast; identical to recomputing compute_ATR(df, 14)).
    atr_smoothed = float(atr_series.ewm(span=3, adjust=False).mean().iloc[-1])
    vol_adj = 1 - min(atr_smoothed / last_close, 0.1)

    f_conf = round(p_fast * vol_adj, 1)
    s_conf = round(p_slow * vol_adj, 1)

    if debug:
        print(f"Fast: {p_fast:.1f} Slow: {p_slow:.1f} Vol adj: {vol_adj:.2f} ATR smoothed: {atr_smoothed:.4f}")

    return {'fast': f_conf, 'slow': s_conf, 'last_close': last_close}


def get_dual_ml_confidence(df: pd.DataFrame, debug=False) -> dict:
    """Returns Fast, Slow confidence scores + Regime label. For display."""
    try:
        res = _run_dual_model(df, debug)
        f_conf = res.get('fast', 50.0)
        s_conf = res.get('slow', 50.0)
        regime = _determine_regime(f_conf, s_conf)
        return {'fast': f_conf, 'slow': s_conf, 'regime': regime}
    except Exception as e:
        if debug:
            print(f"Error: {e}")
        return {'fast': 50.0, 'slow': 50.0, 'regime': 'Error'}


def get_ml_confidence(df, debug=False):
    """
    Returns blended confidence score: 70% Slow, 30% Fast.
    For display and legacy compatibility.
    Use get_dual_ml_confidence_for_kelly() for Kelly sizing.
    """
    res = get_dual_ml_confidence(df, debug)
    f_conf = res.get('fast', 50.0)
    s_conf = res.get('slow', 50.0)
    return round(s_conf * 0.7 + f_conf * 0.3, 1)


def get_dual_ml_confidence_for_kelly(df: pd.DataFrame, debug=False) -> dict:
    """
    Returns all components needed for Kelly sizing and enriched signal generation.

    Architecture:
        slow  → Kelly win probability (matches holding period)
        fast  → Pattern classification and divergence discount
        divergence_discount → Applied to kelly_fraction when models disagree

    Returns:
        slow:               Slow model confidence (0-100)
        fast:               Fast model confidence (0-100)
        blended:            70/30 blend for display
        divergence:         abs(slow - fast)
        divergence_discount: Fraction to reduce kelly_fraction (0.0 to 0.40)
        regime:             Pattern classification string
        kelly_p:            Win probability for Kelly formula (slow / 100)
    """
    try:
        res = _run_dual_model(df, debug)
        f_conf = res.get('fast', 50.0)
        s_conf = res.get('slow', 50.0)

        divergence = abs(s_conf - f_conf)
        regime = _determine_regime(f_conf, s_conf)
        blended = round(s_conf * 0.7 + f_conf * 0.3, 1)

        # Divergence discount on Kelly fraction
        # Below 15 points divergence: no discount
        # Above 15 points: linear scaling up to 40% max discount
        if divergence <= 15.0:
            discount = 0.0
        else:
            excess = divergence - 15.0
            discount = min(excess / 85.0, 0.40)

        return {
            'slow': s_conf,
            'fast': f_conf,
            'blended': blended,
            'divergence': round(divergence, 1),
            'divergence_discount': round(discount, 4),
            'regime': regime,
            'kelly_p': s_conf / 100.0,
        }
    except Exception as e:
        if debug:
            print(f"Error in get_dual_ml_confidence_for_kelly: {e}")
        return {
            'slow': 50.0, 'fast': 50.0, 'blended': 50.0,
            'divergence': 0.0, 'divergence_discount': 0.0,
            'regime': 'Error', 'kelly_p': 0.5,
        }


# ----------------------
# Batched dual confidence (2026-06-18)
#
# get_trends calls get_dual_ml_confidence_for_kelly once per asset (~33x).
# Each call's predict_proba is ~all fixed overhead (a single row through a
# large forest is ~1ms of tree work but ~0.19s of sklearn/array machinery),
# so 33 assets x 2 models = 66 calls dominated get_trends. This batches the
# whole universe into ONE predict_proba per model.
#
# Output-identical to calling get_dual_ml_confidence_for_kelly(df) per asset:
#   - StandardScaler.transform is per-row affine (row-independent)
#   - RandomForest.predict_proba on an (N,F) matrix returns exactly the
#     per-row probabilities of N single-row calls
#   - feature build, vol_adj, rounding, and regime logic are unchanged.
# Case mapping preserved: len<200 / models-missing -> normal {50,50} path
# (regime via _determine_regime, i.e. "Neutral/Chop"); per-df feature
# exception -> the same 'Error' dict the single function returns.
# ----------------------
def _dual_result_from_conf(s_conf, f_conf):
    """Reproduce get_dual_ml_confidence_for_kelly's success-path return dict."""
    divergence = abs(s_conf - f_conf)
    regime = _determine_regime(f_conf, s_conf)
    blended = round(s_conf * 0.7 + f_conf * 0.3, 1)
    if divergence <= 15.0:
        discount = 0.0
    else:
        excess = divergence - 15.0
        discount = min(excess / 85.0, 0.40)
    return {
        'slow': s_conf, 'fast': f_conf, 'blended': blended,
        'divergence': round(divergence, 1),
        'divergence_discount': round(discount, 4),
        'regime': regime, 'kelly_p': s_conf / 100.0,
    }


def _dual_result_error():
    """Reproduce get_dual_ml_confidence_for_kelly's except return dict."""
    return {
        'slow': 50.0, 'fast': 50.0, 'blended': 50.0,
        'divergence': 0.0, 'divergence_discount': 0.0,
        'regime': 'Error', 'kelly_p': 0.5,
    }


def get_dual_ml_confidence_for_kelly_batch(items, debug=False):
    """
    Batched equivalent of get_dual_ml_confidence_for_kelly over many frames.

    items: iterable of (key, df). Returns {key: dict}, each dict identical to
    get_dual_ml_confidence_for_kelly(df) for that frame, but computed with a
    single predict_proba per model across the whole batch.
    """
    items = list(items)
    out = {}

    m_fast, m_slow, scaler, features = get_models()
    models_ok = all([m_fast, m_slow, scaler, features])

    pending_keys, rows, meta = [], [], {}
    for key, df in items:
        if (not models_ok) or len(df) < 200:
            # matches _run_dual_model's early {50,50} -> normal dual path
            out[key] = _dual_result_from_conf(50.0, 50.0)
            continue
        try:
            feat_dict, last_close, atr_series = _compute_features_fast(df)
            row = [feat_dict.get(f, 0.0) for f in features]
            atr_smoothed = float(atr_series.ewm(span=3, adjust=False).mean().iloc[-1])
            pending_keys.append(key)
            rows.append(row)
            meta[key] = (last_close, atr_smoothed)
        except Exception as e:
            if debug:
                print(f"Error in dual batch feature build: {e}")
            out[key] = _dual_result_error()

    if pending_keys:
        try:
            X = pd.DataFrame(rows, columns=features)
            X_scaled = scaler.transform(X)
            fast_idx = list(m_fast.classes_).index(1)
            slow_idx = list(m_slow.classes_).index(1)
            proba_fast = m_fast.predict_proba(X_scaled)[:, fast_idx] * 100
            proba_slow = m_slow.predict_proba(X_scaled)[:, slow_idx] * 100
            for i, key in enumerate(pending_keys):
                last_close, atr_smoothed = meta[key]
                vol_adj = 1 - min(atr_smoothed / last_close, 0.1)
                f_conf = round(float(proba_fast[i]) * vol_adj, 1)
                s_conf = round(float(proba_slow[i]) * vol_adj, 1)
                out[key] = _dual_result_from_conf(s_conf, f_conf)
        except Exception as e:
            if debug:
                print(f"Error in dual batch predict: {e}")
            for key in pending_keys:
                out[key] = _dual_result_error()

    return out


# ----------------------
# Persistence Filter
# ----------------------
_ML_HISTORY: dict = {}
_ML_HISTORY_MAX = 3


def record_ml_confidence(symbol: str, conf: float) -> None:
    if symbol not in _ML_HISTORY:
        _ML_HISTORY[symbol] = []
    _ML_HISTORY[symbol].append(conf)
    if len(_ML_HISTORY[symbol]) > _ML_HISTORY_MAX:
        _ML_HISTORY[symbol].pop(0)


def get_ml_confidence_persistent(symbol: str, df, debug=False) -> float:
    conf = get_ml_confidence(df, debug)
    record_ml_confidence(symbol, conf)
    return conf


def is_ml_persistently_below(symbol: str, threshold: float, n: int = 2) -> bool:
    history = _ML_HISTORY.get(symbol, [])
    if len(history) < n:
        return False
    return all(h < threshold for h in history[-n:])


def is_ml_persistently_above(symbol: str, threshold: float, n: int = 2) -> bool:
    history = _ML_HISTORY.get(symbol, [])
    if len(history) < n:
        return False
    return all(h > threshold for h in history[-n:])


def get_ml_history(symbol: str) -> list:
    return _ML_HISTORY.get(symbol, [])


# ----------------------
# Batch Processing
# ----------------------
def get_ml_confidence_batch(dfs: list, debug=False):
    m_fast, m_slow, scaler, features = get_models()
    if not all([m_fast, m_slow, scaler, features]):
        return [50.0] * len(dfs)

    results = []
    for df in dfs:
        try:
            if len(df) < 200:
                results.append(50.0)
                continue
            feat_dict, last_close, atr_series = _compute_features_fast(df)
            X_live_dict = {f: feat_dict.get(f, 0.0) for f in features}
            X_live = pd.DataFrame([X_live_dict])
            X_scaled = scaler.transform(X_live)

            fast_idx = list(m_fast.classes_).index(1)
            slow_idx = list(m_slow.classes_).index(1)

            p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
            p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

            atr_smoothed = float(atr_series.ewm(span=3, adjust=False).mean().iloc[-1])
            vol_adj = 1 - min(atr_smoothed / last_close, 0.1)

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
_CACHE_TTL = 30


def get_ml_confidence_cached(df, symbol=None, debug=False):
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