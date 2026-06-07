import yfinance as yf
import numpy as np
import pandas as pd
import threading
import time
import math
from functools import wraps, lru_cache
from scipy.optimize import minimize

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets, predict_commodities


# =========================
# TTL Cache
# =========================
def ttl_cache(ttl_seconds=30):
    def decorator(func):
        cache = {}
        lock = threading.Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl_seconds:
                        return value
                expired = [k for k, (_, ts) in cache.items() if now - ts >= ttl_seconds]
                for k in expired:
                    del cache[k]
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, now)
            return result

        return wrapper
    return decorator


# =========================
# SHARED DATA MANAGER
# =========================
@ttl_cache(30)
def _get_shared_market_data():
    all_tickers = list(set(
        list(SECTORS.keys()) +
        list(COUNTRIES.keys()) +
        list(COMMODITIES.keys()) +
        list(CURRENCIES.keys()) +
        list(TREND_ASSETS.keys()) +
        list(SECTOR_NAMES.keys()) +
        ['SPY', 'RSP', '^VIX', '^MOVE', 'HYG', 'IEF', '^TNX', '^IRX',
         'JPY=X', 'HG=F', 'GC=F', 'QQQ']
    ))
    raw = yf.download(all_tickers, period="1y", progress=False,
                      auto_adjust=False, threads=True, group_by='column')
    return raw


@ttl_cache(30)
def _get_extended_data():
    macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
    risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
    all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))
    raw = yf.download(all_tickers, period="300d", progress=False,
                      auto_adjust=False, threads=True)
    return raw, risk_tickers


def _get_close(shared_data, ticker):
    if shared_data.empty or 'Close' not in shared_data:
        return pd.Series(dtype=float)
    close = shared_data['Close']
    if isinstance(close, pd.Series):
        return close.dropna()
    if not isinstance(close.columns, pd.MultiIndex):
        if ticker in close.columns:
            return close[ticker].dropna()
        return pd.Series(dtype=float)
    try:
        if ticker in close.columns.get_level_values(0):
            return close[ticker].dropna()
        return close.xs(ticker, level=1, axis=1).squeeze().dropna()
    except Exception:
        return pd.Series(dtype=float)


# =========================
# Internal Math Helpers
# =========================
@lru_cache(maxsize=128)
def _safe_r2_cached(y_tuple, coeffs_tuple):
    y = np.array(y_tuple)
    coeffs = np.array(coeffs_tuple)
    if len(y) < 2:
        return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum(np.square(y - y_hat))
    y_mean = np.mean(y)
    ss_tot = np.sum(np.square(y - y_mean))
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _safe_r2(y, coeffs):
    return _safe_r2_cached(tuple(y), tuple(coeffs))


def _trend_stats(series, window, scale):
    c = series.dropna().tail(window)
    if len(c) < 5:
        return 0.0, 0.0
    y = np.log(c.values)
    coeffs = np.polyfit(np.arange(len(y)), y, 1)
    slope = float(coeffs[0]) * scale * 100
    r2 = _safe_r2(y, coeffs)
    return round(slope, 2), round(r2, 2)


def _compute_gradient(series, window=5, slice_len=10, scale=1.0):
    if len(series) < window + slice_len:
        return 0.0
    slope_now, r2_now = _trend_stats(series.tail(window), window, scale)
    slope_prev, r2_prev = _trend_stats(series.tail(window + 5).iloc[:-5], window, scale)
    dx_raw = slope_now - slope_prev
    dy_raw = r2_now - r2_prev
    slope_scale = max(abs(slope_now), abs(slope_prev), 1e-6) * 2
    r2_scale = 2.0
    dx = dx_raw / slope_scale
    dy = dy_raw / r2_scale
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return round(angle_deg, 1)


def _compute_slope_change(series, window=20):
    if len(series) < window + 5:
        return 0.0
    slope_now, r2_now = _trend_stats(series.tail(window), window, window)
    slope_prev, r2_prev = _trend_stats(series.tail(window + 5).iloc[:-5], window, window)
    dx_raw = slope_now - slope_prev
    dy_raw = r2_now - r2_prev
    slope_scale = max(abs(slope_now), abs(slope_prev), 1e-6) * 2
    r2_scale = 2.0
    dx = dx_raw / slope_scale
    dy = dy_raw / r2_scale
    return round(math.sqrt(dx ** 2 + dy ** 2), 4)


def _compute_delta_slope(series, window=20):
    if len(series) < window + 5:
        return 0.0
    slope_now, _ = _trend_stats(series.tail(window), window, window)
    slope_prev, _ = _trend_stats(series.tail(window + 5).iloc[:-5], window, window)
    return round(slope_now - slope_prev, 4)


# =========================
# Hurst Exponent
#
# Measures whether a price series is trending, random, or mean-reverting.
#
#   H > 0.55  → trending     — momentum persists, trend system has edge
#   H 0.45–0.55 → random walk — no structural edge, reduce Kelly
#   H < 0.45  → mean-reverting — trend system is fighting the series
#
# Method: Rescaled Range (R/S) analysis over log returns.
# More robust than variance-ratio method for financial time series.
# Uses lags from 10 to max_lag to fit the scaling relationship.
#
# Returns 0.5 (random walk default) on insufficient data.
# =========================
def _hurst_exponent(series, max_lag=40):
    """
    Compute Hurst exponent via rescaled range (R/S) analysis.

    Args:
        series:   Price series (pd.Series or np.array). Uses log returns internally.
        max_lag:  Maximum lag for R/S computation. Min 10 lags required.

    Returns:
        float: Hurst exponent in [0, 1]. 0.5 = random walk default on error.
    """
    try:
        c = np.array(series.dropna()) if hasattr(series, 'dropna') else np.array(series)
        if len(c) < max_lag + 10:
            return 0.5

        log_returns = np.diff(np.log(c))
        if len(log_returns) < max_lag:
            return 0.5

        lags = range(10, max_lag)
        rs_values = []

        for lag in lags:
            # Slice log returns into non-overlapping windows of length lag
            n_windows = len(log_returns) // lag
            if n_windows < 2:
                continue

            rs_window = []
            for i in range(n_windows):
                window = log_returns[i * lag:(i + 1) * lag]
                mean_adj = window - window.mean()
                cumdev = np.cumsum(mean_adj)
                r = cumdev.max() - cumdev.min()   # range of cumulative deviation
                s = window.std(ddof=1)            # standard deviation
                if s > 0:
                    rs_window.append(r / s)

            if rs_window:
                rs_values.append(np.mean(rs_window))

        if len(rs_values) < 5:
            return 0.5

        # Fit log(R/S) ~ H * log(lag) via OLS
        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        h = float(coeffs[0])

        # Clip to valid range — numerical edge cases can exceed [0,1]
        return round(float(np.clip(h, 0.0, 1.0)), 3)

    except Exception:
        return 0.5


# =========================
# Stop Hit Probability
#
# Given current price, stop, and target — estimates the probability
# that price reaches the stop before the target under GBM with drift.
#
# Uses first-passage time for Brownian motion with two absorbing barriers.
# Drift derived from slope (annualised log return). Vol from ATR.
#
# Interpretation:
#   p_stop < 0.25  → strong trend, stop unlikely — hold full size
#   p_stop 0.25–0.40 → moderate risk — standard sizing
#   p_stop > 0.40  → nearly a coin flip — reduce Kelly fraction
#   p_stop > 0.55  → unfavorable — do not enter or exit existing
#
# This penalizes positions where the geometry is wrong regardless of
# ML confidence or slope — catches cases where stop is too tight
# relative to current volatility and expected drift.
# =========================
def _stop_hit_probability(price, stop, target, slope, atr, projection_days=63):
    """
    Probability that price hits stop before target under GBM with drift.

    Args:
        price:           Current price.
        stop:            Stop loss price (below current price).
        target:          Target price (above current price).
        slope:           Slope from _trend_stats (units: scale*100).
                         Converted internally: daily_drift = slope / 1000.
        atr:             Average True Range (14-day). Used as vol proxy.
        projection_days: Horizon for vol annualisation. Default 63 (quarter).

    Returns:
        float: Probability [0, 1] that stop is hit before target.
               Returns 0.5 (neutral) on invalid inputs.
    """
    try:
        if price <= 0 or stop >= price or target <= price:
            return 0.5
        if atr <= 0:
            return 0.5

        # Daily drift from slope
        # slope = log_slope * window * 100, so daily log return = slope / 1000
        daily_drift = slope / 1000.0

        # Daily vol from ATR — ATR is in price units, convert to returns
        # Annualise then de-annualise: daily_vol = (atr/price) / sqrt(252)
        daily_vol = (atr / price) / np.sqrt(252)

        if daily_vol <= 0:
            return 0.5

        # Log distances to barriers
        # a = log distance to stop (negative — below current price)
        # b = log distance to target (positive — above current price)
        a = np.log(stop / price)    # < 0
        b = np.log(target / price)  # > 0

        # GBM drift adjustment: mu = daily_drift - 0.5 * daily_vol^2
        mu = daily_drift - 0.5 * daily_vol ** 2

        # First-passage probability for BM with drift between two barriers
        # P(hit lower barrier a before upper barrier b)
        # Formula: (exp(2*mu*b/sigma^2) - 1) / (exp(2*mu*b/sigma^2) - exp(2*mu*a/sigma^2))
        sigma2 = daily_vol ** 2

        if abs(mu) < 1e-10:
            # Zero drift: probability proportional to distance
            p_stop = abs(a) / (abs(a) + abs(b))
        else:
            exp_b = np.exp(2 * mu * b / sigma2)
            exp_a = np.exp(2 * mu * a / sigma2)
            denom = exp_b - exp_a
            if abs(denom) < 1e-10:
                p_stop = abs(a) / (abs(a) + abs(b))
            else:
                p_stop = (exp_b - 1.0) / denom

        return round(float(np.clip(p_stop, 0.0, 1.0)), 3)

    except Exception:
        return 0.5


# =========================
# Dynamic Vol Cap
#
# Inverts the Kelly weighting logic:
#   Kelly sizing:  slow=40%, fast=10%  (patience, matches holding period)
#   Vol ceiling:   fast=50%, macro=30%, slow=20%  (speed, detects dislocations)
#
# Fast model is the crash detector — degrades quickly when market character
# changes. When fast drops well below slow (divergence), vol cap tightens
# immediately, before the covariance matrix catches up to realized vol.
#
# Min 8%  — floor prevents over-tightening in temporary fast-model dips
# Max 15% — ceiling set by portfolio risk mandate
#            Kelly fraction (0.25) already fractionalizes all weights before
#            the optimizer sees them — a tighter ceiling double-penalizes.
# =========================
def _effective_vol_cap(regime, base_min=0.08, base_max=0.15):
    """
    Compute dynamic portfolio vol ceiling driven by fast ML model.

    Args:
        regime:     Output of get_risk_regime() — contains ml_fast, ml_slow,
                    composite, and details list.
        base_min:   Minimum vol cap (floor). Default 8%.
        base_max:   Maximum vol cap (ceiling). Default 15%.
                    Kelly fraction (0.25) fractionalizes all weights before
                    the optimizer sees them — ceiling at 15% avoids
                    double-penalizing already-conservative Kelly sizing.

    Returns:
        float: Effective annualised vol ceiling for the optimizer (e.g. 0.11).
    """
    ml_fast = regime.get("ml_fast", 50.0) / 100.0        # 0.0 – 1.0
    ml_slow = regime.get("ml_slow", 50.0) / 100.0        # 0.0 – 1.0
    regime_scalar = get_regime_scalar(regime)             # 0.0 – 1.0

    # Weighted combination — fast dominates for vol cap (crash detection)
    # Opposite weighting to Kelly sizing (where slow dominates)
    combined = (
        ml_fast       * 0.50   # primary crash signal — most reactive
        + regime_scalar * 0.30   # macro environment — intermediate
        + ml_slow       * 0.20   # structural anchor — slowest to move
    )

    # Scale linearly between floor and ceiling
    # combined=0.0 → base_min (fully risk-off, tightest)
    # combined=1.0 → base_max (fully risk-on, loosest)
    vol_cap = base_min + combined * (base_max - base_min)

    return round(max(base_min, min(vol_cap, base_max)), 4)


# =========================
# I. Risk Regime
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        raw, risk_tickers = _get_extended_data()
        data = raw['Close'].ffill()

        # -----------------------------------------------
        # ML Risk Model — 20-day stride history (5 points)
        # Produces fast confidence (short-horizon signal)
        # -----------------------------------------------
        history_points = []
        recent_dates = data.index[::-20][:5][::-1]

        for ts in recent_dates:
            ml_res = predict_assets(
                model_path="risk_model.joblib",
                tickers=risk_tickers,
                friendly_names={},
                model_type="risk",
                as_of_date=ts
            )
            probs = ml_res.get('probabilities', {})
            conf_val = round(probs.get('Class 1', 0) * 100, 1)
            history_points.append(conf_val)

        ml_fast_conf = history_points[-1]

        # -----------------------------------------------
        # SPY Slow ML Prediction
        # Mirrors trend model dual architecture:
        # slow model = 60-90 day horizon, structural signal
        # Uses SPY dataframe through get_dual_ml_confidence_for_kelly
        # to produce a slow confidence reading on the market index.
        # -----------------------------------------------
        try:
            from macro.ml_engine import get_dual_ml_confidence_for_kelly

            # Build SPY dataframe from extended data
            spy_df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].xs(
                'SPY', level=1, axis=1
            ).dropna() if isinstance(raw.columns, pd.MultiIndex) else raw.dropna()

            spy_dual = get_dual_ml_confidence_for_kelly(spy_df)
            ml_slow_conf = spy_dual['slow']
            ml_fast_trend = spy_dual['fast']
            spy_regime = spy_dual['regime']
            spy_divergence = spy_dual['divergence']

        except Exception as e:
            print(f"SPY slow ML error: {e}")
            ml_slow_conf = ml_fast_conf   # fallback to fast if slow unavailable
            ml_fast_trend = ml_fast_conf
            spy_regime = "Unknown"
            spy_divergence = 0.0

        # -----------------------------------------------
        # Technical Conditions (6 independent checks)
        # -----------------------------------------------
        last_vals = data.iloc[-1]
        ma50 = data.rolling(50).mean().iloc[-1]

        credit_ratio = data['HYG'] / data['IEF']
        credit_pass = bool(
            last_vals['HYG'] / last_vals['IEF']
            > credit_ratio.rolling(50).mean().iloc[-1]
        )

        curve_spread = last_vals['^TNX'] - last_vals['^IRX']
        curve_pass = bool(curve_spread > 0)

        jpy_ret = data['JPY=X'].pct_change()
        jpy_vol = jpy_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
        carry_pass = bool(
            last_vals['JPY=X'] > ma50['JPY=X'] and jpy_vol < 0.15
        )

        spy_ma200 = data['SPY'].rolling(200).mean().iloc[-1]
        spy_trend = bool(last_vals['SPY'] > spy_ma200)

        vix_low = bool(
            last_vals['^VIX'] < 20 and last_vals['^MOVE'] < 110
        )

        rsp_spy_ratio = data['RSP'] / data['SPY']
        breadth_pass = bool(
            rsp_spy_ratio.iloc[-1]
            > rsp_spy_ratio.rolling(50).mean().iloc[-1]
        )

        details = [
            {"label": "Trend (SPY > 200MA)",      "pass": spy_trend},
            {"label": "Fear (VIX/MOVE Low)",       "pass": vix_low},
            {"label": "Breadth (RSP/SPY > 50MA)",  "pass": breadth_pass},
            {"label": "Credit (HYG/IEF Ratio)",    "pass": credit_pass},
            {"label": "Curve (10Y-3M Spread)",     "pass": curve_pass},
            {"label": "Carry (JPY Weak/Stable)",   "pass": carry_pass},
        ]

        # -----------------------------------------------
        # Composite RISK-ON / RISK-OFF Signal
        #
        # Three independent signals weighted:
        #   50% technical  — 6 observable macro conditions
        #   30% ML slow    — SPY structural trend (60-90 day)
        #   20% ML fast    — risk model short-horizon signal
        #
        # Rationale:
        #   Technical conditions are the most reliable —
        #   hard observable facts, no model inference.
        #   Slow ML matches holding period, stable signal.
        #   Fast ML (risk model) adds short-term sensitivity
        #   but is noisy — given lowest weight.
        #
        # Threshold 0.55:
        #   Requires combined signal above neutral.
        #   Prevents single-session ML collapse from
        #   flipping regime when technicals are intact.
        # -----------------------------------------------
        passes = sum(1 for d in details if d["pass"])
        technical_score = passes / 6.0          # 0.0 to 1.0

        ml_slow_score = ml_slow_conf / 100.0    # 0.0 to 1.0
        ml_fast_score = ml_fast_conf / 100.0    # 0.0 to 1.0

        composite_score = (
            technical_score * 0.50
            + ml_slow_score * 0.40
            + ml_fast_score * 0.10
        )

        is_risk_on = composite_score > 0.55

        return {
            "status":         "RISK-ON" if is_risk_on else "RISK-OFF",
            "confidence":     round(composite_score * 100, 1),
            "ml_slow":        round(ml_slow_conf, 1),
            "ml_fast":        round(ml_fast_conf, 1),
            "composite":      round(composite_score * 100, 1),
            "spy_regime":     spy_regime,
            "spy_divergence": round(spy_divergence, 1),
            "history":        history_points,
            "details":        details,
        }

    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {
            "status":         "ERROR",
            "confidence":     0,
            "ml_slow":        0,
            "ml_fast":        0,
            "composite":      0,
            "spy_regime":     "Error",
            "spy_divergence": 0,
            "history":        [],
            "details":        [],
        }


def get_regime_scalar(regime):
    passes = sum(1 for d in regime["details"] if d["pass"])
    scalars = {6: 1.0, 5: 0.85, 4: 0.70, 3: 0.50, 2: 0.30, 1: 0.15, 0: 0.0}
    return scalars.get(passes, 0.0)


# =========================
# II. ML Asset Predictions
# =========================
@ttl_cache(30)
def get_ml_sector_prediction():
    try:
        tickers = list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS
        res = predict_assets("sector_model.joblib", tickers, SECTOR_NAMES, "sector")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": SECTOR_NAMES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Sector Error: {e}")
        return {"top": {"name": "N/A", "ticker": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_country_prediction():
    try:
        tickers = list(COUNTRIES.keys()) + ML_MACRO_TICKERS + ['SPY']
        res = predict_assets("country_model.joblib", tickers, COUNTRIES, "country")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Country Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_commodity_prediction():
    try:
        res = predict_commodities(
            sector_model_path="commodity_sector_model.joblib",
            commodity_model_path="commodity_model.joblib",
            friendly_names=COMMODITIES,
            use_cache=True,
            top_n_sectors=5,
            top_n_per_sector=5
        )
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Commodity Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


# =========================
# III. Traditional Technical Signals
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        shared_data = _get_shared_market_data()
        vix = _get_close(shared_data, '^VIX')
        rsp = _get_close(shared_data, 'RSP')
        spy = _get_close(shared_data, 'SPY')

        vix_tail = vix.tail(50)
        v_last = float(vix.iloc[-1])
        vix_mean = vix_tail.mean()
        vix_std = vix_tail.std()
        z = float((v_last - vix_mean) / vix_std)

        ratio = rsp / spy
        current_ratio = float(ratio.iloc[-1])
        ma50_ratio = float(ratio.tail(50).mean())
        breadth_failing = current_ratio < ma50_ratio

        spy_ma50 = float(spy.rolling(50).mean().iloc[-1])
        spy_ma200 = float(spy.rolling(200).mean().iloc[-1])
        spy_last = float(spy.iloc[-1])
        spy_in_uptrend = spy_last > spy_ma50 and spy_last > spy_ma200

        spy_slope, spy_r2 = _trend_stats(spy, 10, 10)
        spy_trending = spy_slope > 0 and spy_r2 > 0.6

        if z > 4.0:
            signal = "AGGRESSIVE_BUY" if spy_trending else "SCALE_IN"
        elif z > 2.0:
            signal = "SCALE_IN" if spy_in_uptrend else "HOLD"
        elif z < -1.5:
            signal = "AGGRESSIVE_TRIM"
        elif z < -1.0 and breadth_failing:
            signal = "CAUTIOUS_TRIM"
        else:
            signal = "HOLD"

        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": signal}
    except Exception as e:
        print(f"VIX Signal Error: {e}")
        return {"vix": 0, "z": 0, "signal": "ERROR"}


def get_mean_reversion():
    try:
        shared_data = _get_shared_market_data()
        c = _get_close(shared_data, 'QQQ')
        if c.empty:
            raise ValueError("QQQ data empty")

        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        if pd.isna(rsi2):
            raise ValueError("RSI(2) returned NaN — insufficient data")

        price = float(c.iloc[-1])
        sma200 = float(c.rolling(200, min_periods=200).mean().iloc[-1])
        dma10 = float(c.rolling(10, min_periods=10).mean().iloc[-1])

        if pd.isna(sma200):
            raise ValueError("SMA200 returned NaN — insufficient data (need 200 bars)")
        if pd.isna(dma10):
            raise ValueError("DMA5 returned NaN — insufficient data (need 5 bars)")

        if price < sma200:
            signal = "RISK OFF"
        elif rsi2 <= 10:
            signal = "BUY"
        elif price < dma10 or rsi2 >= 70:
            signal = "EXIT"
        else:
            signal = "HOLD"

        return {"price": round(price, 2), "rsi2": round(rsi2, 1), "signal": signal}

    except Exception as e:
        print(f"Mean Reversion Error: {e}")
        return {"price": 0, "rsi2": 0, "signal": "ERROR"}


# =========================
# IV. Rotation
# =========================
@ttl_cache(30)
def get_sector_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        sector_data = data[list(SECTORS.keys())]
        rel = sector_data.div(data["SPY"], axis=0).pct_change(63).iloc[-1]
        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            gradient = _compute_gradient(data[t].tail(20))
            slope_change = _compute_slope_change(data[t])
            out.append({
                "name": name, "gain": f"{rel[t]:+.2%}",
                "is_positive": slope > 0, "r2": r2, "slope": slope,
                "gradient": gradient, "slope_change": slope_change
            })
        return {"all_ranked": sorted(out, key=lambda x: x["slope"], reverse=True)}
    except Exception as e:
        print(f"Sector Rotation Error: {e}")
        return {"all_ranked": []}


@ttl_cache(30)
def get_country_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COUNTRIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Country Rotation Error: {e}")
        return []


@ttl_cache(30)
def get_commodity_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        results = []
        for s, n in COMMODITIES.items():
            slope, r2 = _trend_stats(data[s], 20, 20)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Commodity Rotation Error: {e}")
        return []


@ttl_cache(30)
def get_currency_rotation():
    try:
        shared_data = _get_shared_market_data()
        data = shared_data['Close']
        invert_set = {"EURUSD=X", "GBPUSD=X", "AUDUSD=X"}
        results = []
        for s, n in CURRENCIES.items():
            c = data[s].dropna().tail(60)
            if s not in invert_set:
                c = (1 / c).replace([np.inf, -np.inf], np.nan).dropna()
            if len(c) < 5:
                continue
            slope, r2 = _trend_stats(c, 60, 60)
            gradient = _compute_gradient(c.tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(c)
            results.append({"sym": s, "name": n, "slope": round(slope, 4), "r2": r2,
                            "gradient": gradient, "slope_change": slope_change})
        return results
    except Exception as e:
        print(f"Currency Rotation Error: {e}")
        return []


# =========================
# V. Portfolio Optimizer
# =========================
_portfolio_summary = {}


def get_portfolio_summary():
    return _portfolio_summary


def _compute_covariance_matrix(shared_data, tickers, window=60):
    prices = shared_data['Close'][tickers].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    cov_matrix = log_returns.tail(window).cov() * 252
    corr_matrix = log_returns.tail(window).corr()
    return cov_matrix, corr_matrix


def _compute_portfolio_var(weights, cov_matrix):
    return float(weights @ cov_matrix.values @ weights)


def _compute_risk_contribution(weights, cov_matrix):
    port_var = _compute_portfolio_var(weights, cov_matrix)
    marginal = cov_matrix.values @ weights
    return weights * marginal / (port_var + 1e-9)


def _conviction_score(item):
    return max(item["slope"] * item["r2"] * item["ml_conf"] / 100, 0.0)


def _kelly_covariance_optimizer(
        trends,
        shared_data,
        portfolio_value=500000,
        max_single=0.15,
        max_risk_contribution=0.30,
        min_portfolio_vol=0.08,
        max_portfolio_vol=0.15,
        regime_scalar=1.0,
        conviction_threshold=0.5,
        regime=None,
):
    """
    Kelly-covariance portfolio optimizer with dynamic vol cap.

    Vol ceiling is driven by the fast ML model (crash detection) rather than
    a static limit. This inverts the Kelly sizing logic: where Kelly uses slow
    model (patient, matches holding period), the vol cap uses fast model
    (reactive, detects dislocations before covariance matrix catches up).

    Args:
        min_portfolio_vol:  Floor for the dynamic vol cap. Default 8%.
        max_portfolio_vol:  Ceiling for the dynamic vol cap. Default 15%.
                            Kelly fraction (0.25) already fractionalizes all
                            weights — ceiling at 15% avoids double-penalizing
                            already-conservative Kelly sizing.
        regime:             Output of get_risk_regime(). If provided, vol cap
                            is computed dynamically. If None, uses max_portfolio_vol.
    """
    empty_summary = {
        "total_allocated": 0.0, "portfolio_vol": 0.0, "n_positions": 0,
        "max_risk_contributor": "N/A", "optimization_success": False,
        "regime_scalar": round(regime_scalar, 2),
        "vol_cap": round(max_portfolio_vol, 4),
    }

    # Compute dynamic vol cap from fast model if regime is available
    # Falls back to max_portfolio_vol if regime not provided
    if regime is not None:
        effective_vol_cap = _effective_vol_cap(
            regime,
            base_min=min_portfolio_vol,
            base_max=max_portfolio_vol,
        )
    else:
        effective_vol_cap = max_portfolio_vol

    active_signals = [
        t for t in trends
        if any(s in t["status"] for s in ["BUY", "STRONG BUY", "BREAKOUT", "HOLD"])
    ]

    if not active_signals:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap, 4)}

    tickers = [t["sym"] for t in active_signals]
    available = [t for t in tickers if t in shared_data['Close'].columns]

    if not available:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap, 4)}

    # Single asset bypass path
    if len(available) < 2:
        sym = available[0]
        item = next(t for t in trends if t["sym"] == sym)
        w = min(max(item["dollar_amount"] / portfolio_value, 0.02) * regime_scalar, max_single)
        dollar_amount = round(w * portfolio_value, 2)
        updated_trends = []
        for t in trends:
            if t["sym"] == sym:
                t = {**t, "pos_size": f"${dollar_amount:,.0f}", "dollar_amount": dollar_amount,
                     "weight_pct": round(w * 100, 1), "risk_contribution": 100.0,
                     "top_correlations": {}}
            else:
                t = {**t, "weight_pct": 0.0, "risk_contribution": 0.0, "top_correlations": {}}
            updated_trends.append(t)
        return updated_trends, {
            "total_allocated": round(w * 100, 1), "portfolio_vol": 0.0,
            "n_positions": 1, "max_risk_contributor": sym,
            "optimization_success": True,
            "regime_scalar": round(regime_scalar, 2),
            "vol_cap": round(effective_vol_cap, 4),
        }

    cov_matrix, corr_matrix = _compute_covariance_matrix(shared_data, available)

    kelly_weights = np.array([
        next(t["dollar_amount"] for t in active_signals if t["sym"] == sym)
        / portfolio_value * regime_scalar
        for sym in available
    ])

    # Instruments with zero Kelly weight get zero upper bound —
    # prevents optimizer allocating to instruments below Kelly quality threshold
    def upper_bound(sym, kelly_w):
        if kelly_w > 0:
            return min(kelly_w, max_single)
        return 0.0

    bounds = [(0.0, upper_bound(sym, kelly_weights[i])) for i, sym in enumerate(available)]

    # x0 starts at zero for instruments with zero Kelly weight
    x0 = np.array([w if w > 0 else 0.0 for w in kelly_weights])

    conviction_raw = np.array([
        _conviction_score(next(t for t in active_signals if t["sym"] == sym))
        for sym in available
    ])

    conviction_sum = conviction_raw.sum()
    if conviction_sum == 0:
        return trends, {**empty_summary, "vol_cap": round(effective_vol_cap, 4)}

    conviction = conviction_raw / conviction_sum
    n = len(available)

    def neg_objective(weights):
        return -float(weights @ conviction)

    # Vol constraint uses effective_vol_cap (dynamic, fast-model driven)
    # rather than static max_portfolio_vol
    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
        {"type": "ineq",
         "fun": lambda w: effective_vol_cap ** 2 - _compute_portfolio_var(w, cov_matrix)},
        *[
            {"type": "ineq",
             "fun": lambda w, i=i: max_risk_contribution - _compute_risk_contribution(w, cov_matrix)[i]}
            for i in range(n)
        ]
    ]

    result = minimize(
        neg_objective, x0=x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 1000, "ftol": 1e-9}
    )

    optimized_weights = result.x if result.success else kelly_weights

    port_vol = np.sqrt(_compute_portfolio_var(optimized_weights, cov_matrix))
    risk_contribs = _compute_risk_contribution(optimized_weights, cov_matrix)

    ticker_to_weight = dict(zip(available, optimized_weights))
    ticker_to_risk = dict(zip(available, risk_contribs))
    ticker_to_corr = {t: corr_matrix[t].to_dict() for t in available}

    updated_trends = []
    for item in trends:
        if item["sym"] in ticker_to_weight:
            w = ticker_to_weight[item["sym"]]
            dollar_amount = round(w * portfolio_value, 2)
            top_corr = dict(list({
                k: round(v, 2)
                for k, v in sorted(
                    ticker_to_corr[item["sym"]].items(),
                    key=lambda x: abs(x[1]), reverse=True
                )
                if k != item["sym"]
            }.items())[:3])
            item = {
                **item,
                "pos_size": f"${dollar_amount:,.0f}",
                "dollar_amount": dollar_amount,
                "weight_pct": round(w * 100, 1),
                "risk_contribution": round(ticker_to_risk[item["sym"]] * 100, 1),
                "top_correlations": top_corr,
            }
        else:
            item = {**item, "weight_pct": 0.0, "risk_contribution": 0.0,
                    "top_correlations": {}}
        updated_trends.append(item)

    return updated_trends, {
        "total_allocated": round(float(optimized_weights.sum()) * 100, 1),
        "portfolio_vol": round(port_vol * 100, 1),
        "n_positions": int((optimized_weights > 0.001).sum()),
        "max_risk_contributor": available[int(np.argmax(risk_contribs))],
        "optimization_success": result.success,
        "regime_scalar": round(regime_scalar, 2),
        "vol_cap": round(effective_vol_cap * 100, 1),   # expose for dashboard display
    }


# =========================
# VI. Kelly Sizing
# FIX: slope / 1000 corrects dimensionality of daily_return_pct
# NEW: accepts ml_conf_slow and divergence_discount separately
# =========================
def _compute_kelly_size(price, slope, atr, ml_conf_slow, r2,
                        portfolio_value=500000,
                        projection_days=63,
                        atr_stop_multiplier=2.5,
                        kelly_fraction=0.25,
                        max_allocation=0.15,
                        delta_slope=0.0,
                        divergence_discount=0.0):
    """
    Kelly position sizing using slow model probability.

    Args:
        ml_conf_slow:        Slow model confidence (0-100). Used as Kelly win probability.
        divergence_discount: Fraction to reduce kelly_fraction when fast/slow diverge.
                             Computed in ml_engine.get_dual_ml_confidence_for_kelly().
    """
    zero = {
        'dollar_amount': 0.0, 'shares': 0,
        'stop': round(price - (atr * atr_stop_multiplier), 2),
        'target': round(price, 2), 'rr_ratio': 0.0,
        'risk_dollar': 0.0, 'exp_return': 0.0
    }

    # Gate: slow model must clear 50% and slope/r2 must be positive quality
    if slope <= 0 or ml_conf_slow <= 50 or r2 < 0.15:
        return zero

    # Adjust kelly_fraction for momentum direction
    if delta_slope > 3:
        kelly_fraction = min(kelly_fraction * 1.25, 0.40)
    elif delta_slope < 3:
        kelly_fraction = kelly_fraction * 0.75

    # Apply divergence discount — reduce sizing when fast/slow disagree
    kelly_fraction = kelly_fraction * (1.0 - divergence_discount)

    stop_price = price - (atr * atr_stop_multiplier)
    risk_per_share = price - stop_price

    if stop_price >= price or risk_per_share <= 0:
        return {**zero, 'stop': round(stop_price, 2)}

    # FIX: slope is in units of scale*100 from _trend_stats(window=10, scale=10)
    # slope = log_slope * 10 * 100 = 1000 * daily_log_return
    # So daily_return_pct = slope / 1000
    daily_return_pct = slope / 1000

    projected_price = price * ((1 + daily_return_pct) ** projection_days)
    target_price = price + (projected_price - price) * r2
    reward_per_share = target_price - price
    rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

    if rr_ratio < 1.0:
        return {**zero, 'stop': round(stop_price, 2),
                'target': round(target_price, 2), 'rr_ratio': round(rr_ratio, 2)}

    # Kelly formula using slow model probability
    p = ml_conf_slow / 100.0
    q = 1.0 - p
    b = rr_ratio
    kelly = (b * p - q) / b
    fractional_kelly = kelly * kelly_fraction
    final_allocation = max(0, min(fractional_kelly, max_allocation))

    position_value = portfolio_value * final_allocation
    shares = int(position_value / price)
    actual_investment = shares * price
    risk_per_position = shares * risk_per_share
    reward_per_position = shares * reward_per_share
    expected_return = reward_per_position * p - risk_per_position * q

    return {
        'dollar_amount': round(actual_investment, 2),
        'shares': shares,
        'stop': round(stop_price, 2),
        'target': round(target_price, 2),
        'rr_ratio': round(rr_ratio, 2),
        'risk_dollar': round(risk_per_position, 2),
        'exp_return': round(expected_return, 2)
    }


# =========================
# VII. Trends (Main Entry Point)
# =========================
@ttl_cache(30)
def get_trends():
    global _portfolio_summary
    from macro.ml_engine import get_dual_ml_confidence_for_kelly

    shared_data = _get_shared_market_data()
    results = []
    symbols = list(TREND_ASSETS.keys())

    for sym, name in TREND_ASSETS.items():
        try:
            if len(symbols) > 1:
                df = shared_data.xs(sym, level=1, axis=1).dropna()
            else:
                df = shared_data.dropna()

            if df.empty:
                continue

            c = df["Close"].squeeze()
            ma_50 = c.rolling(50, min_periods=1).mean()
            ma_200 = c.rolling(200, min_periods=1).mean()

            slope, r2 = _trend_stats(c, 10, 10)

            # Get dual model components
            dual = get_dual_ml_confidence_for_kelly(df)
            ml_conf_slow = dual['slow']
            ml_conf_fast = dual['fast']
            ml_conf = dual['blended']      # display value
            divergence = dual['divergence']
            divergence_discount = dual['divergence_discount']
            regime = dual['regime']

            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            s50 = float(ma_50.iloc[-1])
            s200 = float(ma_200.iloc[-1])

            rsi14 = float(compute_RSI(c, 14).iloc[-1])

            # Hurst exponent — trend persistence probability
            # Uses full available price history (up to 1y from shared data)
            hurst = _hurst_exponent(c, max_lag=40)

            # Z-Score calculation
            c_len = len(c)
            start_idx = max(0, c_len - 60)
            hist_slopes = []
            for i in range(start_idx + 10, c_len + 1):
                window_slice = c.iloc[i - 10:i]
                if len(window_slice) >= 5:
                    s_val, _ = _trend_stats(window_slice, 10, 10)
                    hist_slopes.append(s_val)

            if hist_slopes:
                slope_mean = np.mean(hist_slopes)
                slope_std = np.std(hist_slopes)
                slope_z = (slope - slope_mean) / slope_std if slope_std > 0 else 0
            else:
                slope_z = 0

            delta_slope = _compute_delta_slope(c, window=20)

            # Stop hit probability — probability price reaches stop before target
            # Computed before Kelly so it can discount position sizing
            # Uses placeholder stop/target from ATR — refined after Kelly call
            atr_stop_prelim = last - (atr * 2.5)
            slope_prelim, _ = _trend_stats(c, 10, 10)
            daily_return_prelim = slope_prelim / 1000
            target_prelim = last * ((1 + daily_return_prelim) ** 63)
            p_stop = _stop_hit_probability(
                last, atr_stop_prelim, target_prelim, slope, atr
            )

            # Hurst-adjusted divergence discount
            # Mean-reverting series (H < 0.45) compounds the divergence penalty
            # Trending series (H > 0.55) reduces it slightly
            hurst_discount = 0.0
            if hurst < 0.45:
                hurst_discount = (0.45 - hurst) * 0.5   # up to +22.5% discount
            elif hurst > 0.55:
                hurst_discount = -(hurst - 0.55) * 0.2  # up to -10% discount (bonus)

            # Stop probability discount — reduce Kelly when geometry is unfavorable
            # p_stop > 0.40 means nearly a coin flip on stop vs target
            # Scale discount linearly from 0 at p_stop=0.40 to 0.50 at p_stop=0.70
            p_stop_discount = max(0.0, (p_stop - 0.40) / 0.30 * 0.50) if p_stop > 0.40 else 0.0

            # Combined divergence discount: original + hurst + p_stop geometry
            combined_discount = float(np.clip(
                divergence_discount + hurst_discount + p_stop_discount,
                -0.20,
                0.80
            ))

            # Kelly sizing uses slow model and combined discount
            position = _compute_kelly_size(
                last, slope, atr, ml_conf_slow, r2,
                delta_slope=delta_slope,
                divergence_discount=combined_discount
            )
            pos_size = position['dollar_amount']

            # Recompute p_stop with final Kelly stop/target for accurate display
            p_stop = _stop_hit_probability(
                last, position['stop'], position['target'], slope, atr
            )

            # =============================================
            # Decision Logic
            # Priority order matters — earlier conditions win
            # =============================================
            # sell
            if last < position['stop']:
                status = "SELL (STOP)"

            elif last < s50 and slope < 0:
                # Price below MA50 AND slope negative → confirmed downtrend
                status = "SELL (MA50)"

            elif slope_z > 2.0 and ml_conf_slow > 55 and r2 > 0.7 and rsi14 < 75 and slope > 0:
                # Momentum breakout — all three confirm
                status = "BUY (BREAKOUT)"

            elif slope_z > 2.0 and r2 > 0.8 and rsi14 > 75:
                # Slope very extended but ML not confirming → trim
                status = "TRIM (EXTENDED)"

            elif slope_z > 1.5 and ml_conf_slow < 45:
                # Momentum elevated but slow model losing conviction
                status = "TRIM (FADING MOMENTUM)"

            elif ml_conf_slow < 45 and last > s50:
                # Slow model has lost conviction — above MA50 but deteriorating
                status = "TRIM (ML FADE)"

            elif slope < -2:
                # Slope significantly negative
                status = "TRIM (NEGATIVE SLOPE)"

            elif pos_size == 0:
                status = "TRIM (POSITION SIZE)"

            # buy/hold zone
            elif (last > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
                # Strong uptrend — determine entry quality
                if ml_conf_slow > 60 and ml_conf_fast < 45:
                    # Slope below its own mean — momentum pullback within uptrend
                    # Good entry point (buying the dip)
                    status = "BUY (PULLBACK)"
                elif ml_conf_slow > 55 and ml_conf_fast > 60:
                    status = "BUY (BULL)"
                elif ml_conf_slow > 50:
                    status = "BUY"
                elif ml_conf_slow < 50 and ml_conf_fast > 60:
                    status = "HOLD (Recovering)"
                else:
                    status = "HOLD"

            else:
                status = "HOLD"

            results.append({
                "sym": sym,
                "name": name,
                "price": round(last, 2),
                "status": status,
                "r2": round(r2, 2),
                "ml_conf": ml_conf_slow,           # blended — for display
                "ml_conf_slow": ml_conf_slow,       # slow — for Kelly
                "ml_conf_fast": ml_conf_fast,       # fast — for pattern
                "divergence": divergence,
                "regime": regime,
                "rsi14": round(rsi14, 1),
                "slope": round(slope, 2),
                "slope_z": round(slope_z, 2),
                "delta_slope": round(delta_slope, 4),
                "hurst": hurst,                     # trend persistence [0,1]
                "p_stop": p_stop,                   # probability of stop before target
                "stop": position['stop'],
                "target": position['target'],
                "rr_ratio": position['rr_ratio'],
                "pos_size": f"${position['dollar_amount']:,.0f}",
                "dollar_amount": position['dollar_amount'],
                "shares": position['shares'],
                "risk_dollar": position['risk_dollar'],
                "exp_return": position['exp_return'],
                "weight_pct": 0.0,
                "risk_contribution": 0.0,
                "top_correlations": {},
            })

        except Exception as e:
            print(f"Error in trend loop for {sym}: {e}")
            continue

    # Sort by composite score: slope * r2 * ml_conf_slow
    sorted_results = sorted(
        results,
        key=lambda x: x["slope"] * x["r2"] * x.get("ml_conf_slow", x["ml_conf"]) / 100,
        reverse=True
    )

    # Get regime once — passed to optimizer for dynamic vol cap
    regime = get_risk_regime()
    scalar = get_regime_scalar(regime)

    optimized_results, summary = _kelly_covariance_optimizer(
        sorted_results, shared_data,
        portfolio_value=500000,
        max_single=0.15,
        max_risk_contribution=0.30,
        min_portfolio_vol=0.08,     # floor: 8%
        max_portfolio_vol=0.15,     # ceiling: 15% — Kelly fraction already fractionalizes
        regime_scalar=scalar,
        conviction_threshold=0.5,
        regime=regime,              # pass regime for dynamic vol cap
    )

    _portfolio_summary = summary
    return optimized_results
