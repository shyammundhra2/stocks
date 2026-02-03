import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import threading
import time
from functools import wraps, lru_cache

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets


# TTL cache with cleanup
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
                # Cleanup old entries (prevent memory leak)
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
# SHARED DATA MANAGER (BIGGEST OPTIMIZATION)
# =========================
@ttl_cache(30)
def _get_shared_market_data():
    """
    Single bulk download for all tickers used across functions.
    This eliminates redundant yfinance calls and is the biggest performance win.
    """
    # Collect all unique tickers from all functions
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

    # Single bulk download with threading
    raw = yf.download(all_tickers, period="1y", progress=False,
                      auto_adjust=False, threads=True, group_by='column')

    return raw


@ttl_cache(30)
def _get_extended_data():
    """
    Extended data for risk regime (needs 300d history for ML)
    """
    macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
    risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
    all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))

    raw = yf.download(all_tickers, period="300d", progress=False,
                      auto_adjust=False, threads=True)

    return raw, risk_tickers


# =========================
# Internal Math Helpers (Optimized)
# =========================
@lru_cache(maxsize=128)
def _safe_r2_cached(y_tuple, coeffs_tuple):
    """Cached R2 calculation for repeated calls with same data"""
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
    """Wrapper to use cached version"""
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


# =========================
# I. Risk Regime (Uses Extended Data)
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        # Get extended data for ML history
        raw, risk_tickers = _get_extended_data()
        data = raw['Close'].ffill()

        # 2. ML Logic: Current & 20-Point History
        # We use the logic from your predict.py by iterating over the last 20 valid trading dates
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

        current_conf = history_points[-1]
        is_risk_on_ml = current_conf > 50

        # 3. Vectorized macro calculations
        last_vals = data.iloc[-1]
        ma50 = data.rolling(50).mean().iloc[-1]

        # Credit Stress
        credit_ratio = data['HYG'] / data['IEF']
        credit_pass = bool(last_vals['HYG'] / last_vals['IEF'] > credit_ratio.rolling(50).mean().iloc[-1])

        # Yield Curve
        curve_spread = last_vals['^TNX'] - last_vals['^IRX']
        curve_pass = bool(curve_spread > 0)

        # Carry Trade (vectorized volatility calculation)
        jpy_ret = data['JPY=X'].pct_change()
        jpy_vol = jpy_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
        carry_pass = bool(last_vals['JPY=X'] > ma50['JPY=X'] and jpy_vol < 0.15)

        # Existing Technicals
        spy_ma200 = data['SPY'].rolling(200).mean().iloc[-1]
        spy_trend = bool(last_vals['SPY'] > spy_ma200)
        vix_low = bool(last_vals['^VIX'] < 20 and last_vals['^MOVE'] < 110)

        # Breadth Logic: RSP/SPY Ratio vs its 50-day Moving Average
        rsp_spy_ratio = data['RSP'] / data['SPY']
        breadth_pass = bool(rsp_spy_ratio.iloc[-1] > rsp_spy_ratio.rolling(50).mean().iloc[-1])

        details = [
            {"label": "Trend (SPY > 200MA)", "pass": spy_trend},
            {"label": "Fear (VIX/MOVE Low)", "pass": vix_low},
            {"label": "Breadth (RSP/SPY > 50MA)", "pass": breadth_pass},
            {"label": "Credit (HYG/IEF Ratio)", "pass": credit_pass},
            {"label": "Curve (10Y-3M Spread)", "pass": curve_pass},
            {"label": "Carry (JPY Weak/Stable)", "pass": carry_pass},
        ]

        return {
            "status": "RISK-ON" if is_risk_on_ml else "RISK-OFF",
            "confidence": current_conf,
            "history": history_points,
            "details": details
        }
    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {"status": "ERROR", "confidence": 0, "history": [], "details": []}


# =========================
# II. ML Asset Predictions (Uses Shared Data)
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
        return {
            "top": top,
            "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Sector Error: {e}")
        return {"top": {"name": "N/A", "ticker": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_country_prediction():
    try:
        tickers = list(COUNTRIES.keys()) + ML_MACRO_TICKERS + ['SPY']  # SPY needed by compute_country_features()
        res = predict_assets("country_model.joblib", tickers, COUNTRIES, "country")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {
            "top": top,
            "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Country Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


@ttl_cache(30)
def get_ml_commodity_prediction():
    try:
        res = predict_assets("commodity_model.joblib",
                             list(COMMODITIES.keys()) + ML_MACRO_TICKERS,
                             COMMODITIES, "commodity")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {
            "top": top,
            "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Commodity Error: {e}")
        return {"top": {"name": "N/A", "confidence": 0}, "all": []}


# =========================
# III. Traditional Technical Signals (Optimized with Shared Data)
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        # Use shared data instead of separate download
        shared_data = _get_shared_market_data()
        vix = shared_data['Close']['^VIX'].dropna()

        # Single pass calculations
        vix_tail = vix.tail(50)
        v_last = float(vix.iloc[-1])
        vix_mean = vix_tail.mean()
        vix_std = vix_tail.std()
        z = float((v_last - vix_mean) / vix_std)

        # 2. Breadth Calculations (RSP/SPY Ratio)
        rsp = shared_data['Close']['RSP']
        spy = shared_data['Close']['SPY']
        ratio = rsp / spy

        current_ratio = float(ratio.iloc[-1])
        ma50_ratio = float(ratio.tail(50).mean())
        breadth_failing = current_ratio < ma50_ratio

        # FIX: flattened signal chain — the trim/hold checks were previously
        # nested inside the z > 1.0 block where they could never trigger.
        if z > 2.0:
            signal = "AGGRESSIVE_BUY"
        elif z > 1.0:
            signal = "SCALE_IN"
        elif z < -1.5:
            signal = "AGGRESSIVE_TRIM"
        elif z < -1.0 and breadth_failing:
            signal = "CAUTIOUS_TRIM"
        else:
            signal = "HOLD"

        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": signal}
    except:
        return {"vix": 0, "z": 0, "signal": "ERROR"}


@ttl_cache(30)
def get_mean_reversion():
    try:
        # Use shared data instead of separate download
        shared_data = _get_shared_market_data()
        c = shared_data['Close']['QQQ'].dropna()

        # Parallel calculations
        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        price = float(c.iloc[-1])
        s200 = float(c.rolling(200, min_periods=1).mean().iloc[-1])

        # Signal determination
        if rsi2 >= 70:
            signal = "EXIT"
        elif price < s200:
            signal = "RISK OFF"
        elif rsi2 <= 10:
            signal = "BUY"
        else:
            signal = "HOLD"

        return {"price": round(price, 2), "rsi2": round(rsi2, 1), "signal": signal}
    except:
        return {"price": 0, "rsi2": 0, "signal": "ERROR"}


# =========================
# IV. Rotation & Trends (Uses Shared Data)
# =========================
import math


def _compute_gradient(series, window=5, slice_len=10, scale=1.0):
    """Optimized gradient calculation"""
    if len(series) < window + slice_len:
        return 0.0

    # Current week
    slope_now, r2_now = _trend_stats(series.tail(window), window, scale)
    # Last week
    slope_prev, r2_prev = _trend_stats(series.tail(window + 5).iloc[:-5], window, scale)

    dx = slope_now - slope_prev
    dy = r2_now - r2_prev

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return round(angle_deg, 1)


def _compute_slope_change(series, window=20):
    """Compute slope change from last week"""
    if len(series) < window + 5:
        return 0.0
    slope_this_week, _ = _trend_stats(series.tail(window), window, window)
    slope_last_week, _ = _trend_stats(series.tail(window + 5)[:-5], window, window)
    return round(slope_this_week - slope_last_week, 2)


@ttl_cache(30)
def get_sector_rotation():
    try:
        # Use shared data - no separate download needed
        shared_data = _get_shared_market_data()
        data = shared_data['Close']

        # Vectorized relative strength calculation
        sector_data = data[list(SECTORS.keys())]
        rel = sector_data.div(data["SPY"], axis=0).pct_change(63).iloc[-1]

        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            gradient = _compute_gradient(data[t].tail(20))
            slope_change = _compute_slope_change(data[t])
            rel_gain = rel[t]
            out.append({
                "name": name,
                "gain": f"{rel_gain:+.2%}",
                "is_positive": slope > 0,
                "r2": r2,
                "slope": slope,
                "gradient": gradient,
                "slope_change": slope_change
            })

        # Sort by slope instead of gain
        return {"all_ranked": sorted(out, key=lambda x: x["slope"], reverse=True)}
    except:
        return {"all_ranked": []}

@ttl_cache(30)
def get_country_rotation():
    try:
        # Use shared data
        shared_data = _get_shared_market_data()
        data = shared_data['Close']

        results = []
        for s, n in COUNTRIES.items():
            slope, r2 = _trend_stats(data[s], 60, 60)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({
                "sym": s,
                "name": n,
                "slope": slope,
                "r2": r2,
                "gradient": gradient,
                "slope_change": slope_change
            })
        return results
    except:
        return []


@ttl_cache(30)
def get_commodity_rotation():
    try:
        # Use shared data
        shared_data = _get_shared_market_data()
        data = shared_data['Close']

        results = []
        for s, n in COMMODITIES.items():
            slope, r2 = _trend_stats(data[s], 60, 60)
            gradient = _compute_gradient(data[s].tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(data[s])
            results.append({
                "sym": s,
                "name": n,
                "slope": slope,
                "r2": r2,
                "gradient": gradient,
                "slope_change": slope_change
            })
        return results
    except:
        return []


@ttl_cache(30)
def get_currency_rotation():
    try:
        # Use shared data
        shared_data = _get_shared_market_data()
        data = shared_data['Close']

        invert_set = {"EURUSD=X", "GBPUSD=X", "AUDUSD=X"}
        results = []

        for s, n in CURRENCIES.items():
            c = data[s].dropna().tail(60)
            if s not in invert_set:
                c = 1 / c
            slope, r2 = _trend_stats(c, 60, 60)
            gradient = _compute_gradient(c.tail(20), window=5, slice_len=10, scale=60)
            slope_change = _compute_slope_change(c)
            results.append({
                "sym": s,
                "name": n,
                "slope": round(slope, 4),
                "r2": r2,
                "gradient": gradient,
                "slope_change": slope_change
            })

        return results
    except:
        return []


@ttl_cache(30)
def get_trends():
    from macro.ml_engine import get_ml_confidence

    # Use shared data - no separate download needed!
    shared_data = _get_shared_market_data()

    results = []
    symbols = list(TREND_ASSETS.keys())

    for sym, name in TREND_ASSETS.items():
        try:
            # Extract symbol data from shared download
            if len(symbols) > 1:
                df = shared_data.xs(sym, level=1, axis=1).dropna()
            else:
                df = shared_data.dropna()

            if df.empty:
                continue

            c = df["Close"].squeeze()

            # Pre-compute rolling windows once
            c_tail_20 = c.tail(20)
            ma_50 = c.rolling(50, min_periods=1).mean()
            ma_200 = c.rolling(200, min_periods=1).mean()

            # Parallel metric calculations
            slope, r2 = _trend_stats(c, 10, 10)
            ml_conf = get_ml_confidence(df)
            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])

            # Structural levels
            stop = float(c_tail_20.max()) - (2.5 * atr)
            s50 = float(ma_50.iloc[-1])
            s200 = float(ma_200.iloc[-1])

            # Vectorized gradient Z-score
            c_len = len(c)
            start_idx = max(0, c_len - 60)

            # Pre-allocate array for speed
            hist_slopes = np.empty(c_len - start_idx - 10)
            for idx, i in enumerate(range(start_idx + 10, c_len)):
                hist_slopes[idx] = _trend_stats(c.iloc[i - 10:i], 10, 10)[0]

            slope_mean = np.mean(hist_slopes)
            slope_std = np.std(hist_slopes)
            slope_z = (slope - slope_mean) / slope_std if slope_std > 0 else 0

            # Decision logic
            if last < stop:
                status = "SELL (STOP)"
            elif last < s50:
                status = "SELL (MA50)"
            elif slope_z > 2.0 and r2 > 0.8:
                status = "TRIM (GRADIENT)"
            elif ml_conf < 45 and last > s50:
                status = "TRIM (ML FADE)"
            elif (s50 > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
                status = "STRONG BUY" if ml_conf > 75 else "BUY"
            else:
                status = "HOLD"

            pos_size = _compute_kelly_size(ml_conf, last, stop)
            rsi14 = float(compute_RSI(c, 14).iloc[-1])

            results.append({
                "sym": sym,
                "name": name,
                "price": round(last, 2),
                "status": status,
                "r2": round(r2, 2),
                "ml_conf": ml_conf,
                "rsi14": round(rsi14, 1),
                "slope": round(slope, 2),
                "slope_z": round(slope_z, 1),
                "stop": round(stop, 2),
                "pos_size": f"${pos_size:,.0f}"
            })

        except Exception as e:
            print(f"Error in trend loop for {sym}: {e}")
            continue

    return sorted(results, key=lambda x: x["slope"], reverse=True)


def _compute_kelly_size(ml_conf, price, stop, portfolio_value=100000):
    """
    Quarter Kelly sizing logic: f* = (p - q/b) / 4
    b (odds) is set to 2.0 based on a conservative 2:1 Reward/Risk.
    """
    if ml_conf <= 50:
        return 0.0

    p = ml_conf / 100.0
    q = 1.0 - p
    b = 2.0

    full_kelly = p - (q / b)
    fractional_kelly = full_kelly / 4.0
    final_allocation = min(fractional_kelly, 0.15)

    return round(portfolio_value * max(0, final_allocation), 2)