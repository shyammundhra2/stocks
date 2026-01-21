import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import threading
import time
from functools import wraps

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets

# TTL cache
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
            # Compute outside lock to avoid blocking other threads
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, now)
            return result

        return wrapper
    return decorator

# =========================
# Internal Math Helpers (Optimized)
# =========================
def _safe_r2(y, coeffs):
    if len(y) < 2:
        return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum(np.square(y - y_hat))  # Faster than ** 2
    y_mean = np.mean(y)
    ss_tot = np.sum(np.square(y - y_mean))
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def _trend_stats(series, window, scale):
    c = series.dropna().tail(window)
    if len(c) < 5:
        return 0.0, 0.0
    y = np.log(c.values)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0]) * scale * 100
    r2 = _safe_r2(y, coeffs)
    return round(slope, 2), round(r2, 2)


import numpy as np
import yfinance as yf
import pandas as pd

import numpy as np
import yfinance as yf
import pandas as pd


@ttl_cache(30)
def get_risk_regime():
    try:
        # 1. Define Macro Proxies & Tickers
        # ^TNX/^IRX = Yield Curve, HYG/IEF = Credit Stress, JPY=X = Carry
        macro_proxies = ['HYG', 'IEF', '^TNX', '^IRX', 'JPY=X', 'HG=F', 'GC=F']
        risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
        all_tickers = list(set(risk_tickers + macro_proxies + ['^VIX', '^MOVE']))

        # Download data for both ML history and Heuristic calculations
        # We need roughly 20 trading days + 200 days for MAs
        raw = yf.download(all_tickers, period="300d", progress=False, auto_adjust=False)
        data = raw['Close'].ffill()

        # 2. ML Logic: Current & 20-Point History
        # We use the logic from your predict.py by iterating over the last 20 valid trading dates
        history_points = []
        recent_dates = data.index[-5:]

        for ts in recent_dates:
            # We call the real prediction logic for each date in the window
            # Passing the specific date to predict_assets to ensure no data leakage
            ml_res = predict_assets(
                model_path="risk_model.joblib",
                tickers=risk_tickers,
                friendly_names={},
                model_type="risk",
                as_of_date=ts
            )
            probs = ml_res.get('probabilities', {})
            # Capture the "Class 1" (Risk-On) probability for the history plot
            conf_val = round(probs.get('Class 1', 0) * 100, 1)
            history_points.append(conf_val)

        # Current state for main return
        current_conf = history_points[-1]
        is_risk_on_ml = current_conf > 50

        # 3. New Macro Breadth Logic (Heuristics)
        # A. Credit Stress Proxy (High Yield Price / Treasury Price)
        credit_ratio = data['HYG'] / data['IEF']
        credit_pass = bool(credit_ratio.iloc[-1] > credit_ratio.rolling(50).mean().iloc[-1])

        # B. Yield Curve (10Y Yield - 3M Bill Yield)
        curve_spread = data['^TNX'] - data['^IRX']
        curve_pass = bool(curve_spread.iloc[-1] > 0)

        # C. Carry Trade (JPY weakness relative to 50MA + Low Realized Vol)
        jpy_ret = data['JPY=X'].pct_change()
        jpy_vol = jpy_ret.rolling(20).std() * np.sqrt(252)
        carry_pass = bool(
            data['JPY=X'].iloc[-1] > data['JPY=X'].rolling(50).mean().iloc[-1] and jpy_vol.iloc[-1] < 0.15)

        # D. Global Growth (Copper/Gold Ratio)
        cu_au_ratio = data['HG=F'] / data['GC=F']
        growth_pass = bool(cu_au_ratio.iloc[-1] > cu_au_ratio.rolling(50).mean().iloc[-1])

        # E. Existing Technicals
        spy_trend = bool(data['SPY'].iloc[-1] > data['SPY'].rolling(200).mean().iloc[-1])
        vix_low = bool(data['^VIX'].iloc[-1] < 20 and data['^MOVE'].iloc[-1] < 110)

        # 4. Formatted Output (Backward Compatible)
        details = [
            {"label": "Trend (SPY > 200MA)", "pass": spy_trend},
            {"label": "Fear (VIX/MOVE Low)", "pass": vix_low},
            {"label": "Credit (HYG/IEF Ratio)", "pass": credit_pass},
            {"label": "Curve (10Y-3M Spread)", "pass": curve_pass},
            {"label": "Carry (JPY Weak/Stable)", "pass": carry_pass},
            {"label": "Growth (Cu/Au Ratio)", "pass": growth_pass}
        ]

        pass_count = sum(1 for d in details if d['pass'])
        macro_score = (pass_count / len(details)) * 100

        return {
            "status": "RISK-ON" if (is_risk_on_ml and macro_score >= 50) else "RISK-OFF",
            "confidence": current_conf,
            "history": history_points,  # 20 real data points from your ML model
            "details": details
        }
    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {"status": "ERROR", "confidence": 0, "history": [], "details": []}

# =========================
# II. ML Asset Predictions (Optimized)
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
        tickers = list(COUNTRIES.keys()) + ML_MACRO_TICKERS
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
        res = predict_assets("commodity_model.joblib", list(COMMODITIES.keys()) + ML_MACRO_TICKERS, COMMODITIES,
                             "commodity")
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
# III. Traditional Technical Signals
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False, auto_adjust=False)['Close'].squeeze()
        vix_tail = vix.tail(50)
        v_last = float(vix.iloc[-1])
        vix_mean = vix_tail.mean()
        vix_std = vix_tail.std()
        z = float((v_last - vix_mean) / vix_std)

        # Determine signal with single pass
        if z > 2.0:
            signal = "AGGRESSIVE_BUY"
        elif z > 1.0:
            signal = "SCALE_IN"
        elif z < -1.5:
            signal = "TRIM_PROFITS"
        else:
            signal = "NEUTRAL"

        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": signal}
    except:
        return {"vix": 0, "z": 0, "signal": "ERROR"}


@ttl_cache(30)
def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=False, progress=False)
        c = df["Close"].squeeze()
        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        price = float(c.iloc[-1])
        s200 = float(c.rolling(200, min_periods=1).mean().iloc[-1])

        # Determine signal with single pass
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
# IV. Rotation & Trends (Optimized)
# =========================
import math

# -------------------------
# Helper to compute gradient angle
# -------------------------
def _compute_gradient(series, window=5, slice_len=10, scale=1.0):
    """
    Compute gradient (angle in degrees) of R2 vs slope over last `window` slices.
    - x-axis: slope
    - y-axis: R2
    Returns 0-360 degrees.
    """
    if len(series) < window + slice_len:
        return 0.0

    slopes = []
    r2s = []

    # Compute slope and R2 for rolling slices
    """
       Compute movement vector (slope, R²) from last week.
       - series: price series (pd.Series)
       - returns: gradient angle in degrees 0-360
       """
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

# =========================
# Rotation Functions with Gradient & Slope Change
# =========================

def _compute_slope_change(series, window=20):
    """
    Compute slope change from last week.
    """
    if len(series) < window + 5:
        return 0.0
    slope_this_week, _ = _trend_stats(series.tail(window), window, window)
    slope_last_week, _ = _trend_stats(series.tail(window + 5)[:-5], window, window)
    return round(slope_this_week - slope_last_week, 2)


@ttl_cache(30)
def get_sector_rotation():
    try:
        tickers = list(SECTORS.keys()) + ["SPY"]
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=False)['Close']

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
                "is_positive": rel_gain > 0,
                "r2": r2,
                "slope": slope,
                "gradient": gradient,
                "slope_change": slope_change
            })

        return {"all_ranked": sorted(out, key=lambda x: float(x["gain"].strip("%+")), reverse=True)}
    except:
        return {"all_ranked": []}


@ttl_cache(30)
def get_country_rotation():
    try:
        data = yf.download(list(COUNTRIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
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
        data = yf.download(list(COMMODITIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
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
        data = yf.download(list(CURRENCIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
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

# Trends
@ttl_cache(30)
def get_trends():
    from macro.ml_engine import get_ml_confidence
    results = []

    # --- Bulk Download Optimization ---
    symbols = list(TREND_ASSETS.keys())
    # Download all symbols at once; keep auto_adjust=False as per original
    bulk_data = yf.download(symbols, period="1y", progress=False, auto_adjust=False, group_by='column')

    for sym, name in TREND_ASSETS.items():
        try:
            # Extract symbol-specific dataframe from bulk object
            # Handle both single-ticker and multi-ticker return structures
            if len(symbols) > 1:
                df = bulk_data.xs(sym, level=1, axis=1).dropna()
            else:
                df = bulk_data.dropna()

            if df.empty:
                continue

            c = df["Close"].squeeze()

            # --- Technical Metrics ---
            slope, r2 = _trend_stats(c, 10, 10)
            ml_conf = get_ml_confidence(df)
            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])

            # --- Structural Levels ---
            stop = float(c.tail(20).max()) - (2.5 * atr)
            s50 = float(c.rolling(50, min_periods=1).mean().iloc[-1])
            s200 = float(c.rolling(200, min_periods=1).mean().iloc[-1])

            # --- Gradient Z-Score (Trend Acceleration) ---
            c_len = len(c)
            start_idx = max(0, c_len - 60)
            hist_slopes = [_trend_stats(c.iloc[i - 10:i], 10, 10)[0] for i in range(start_idx + 10, c_len)]

            slope_mean = np.mean(hist_slopes)
            slope_std = np.std(hist_slopes)
            slope_z = (slope - slope_mean) / slope_std if slope_std > 0 else 0

            # --- Multi-Tiered Decision Logic ---
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

            # --- Position Sizing ---
            pos_size = compute_kelly_size(ml_conf, last, stop)
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

def compute_kelly_size(ml_conf, price, stop, portfolio_value=100000):
    """
    Quarter Kelly sizing logic: f* = (p - q/b) / 4
    b (odds) is set to 2.0 based on a conservative 2:1 Reward/Risk.
    """
    if ml_conf <= 50:
        return 0.0

    # 1. Winning Probability
    p = ml_conf / 100.0
    q = 1.0 - p

    # 2. Risk/Reward (b)
    b = 2.0

    # 3. Kelly Calculation
    full_kelly = p - (q / b)

    # 4. Apply Quarter Kelly Fraction & Cap at 15% per position
    fractional_kelly = full_kelly / 4.0
    final_allocation = min(fractional_kelly, 0.15)

    return round(portfolio_value * max(0, final_allocation), 2)