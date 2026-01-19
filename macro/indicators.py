import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import time
from functools import wraps, lru_cache

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets


# =========================
# TTL Cache Decorator
# =========================
def ttl_cache(ttl_seconds=30):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return value
            result = func(*args, **kwargs)
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


# =========================
# I. Risk Regime (ML Core + Heuristic Display)
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        # Single combined download for all needed tickers
        all_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY', '^VIX', '^MOVE'] + list(SECTOR_NAMES.keys())))
        raw = yf.download(all_tickers, period="300d", progress=False, auto_adjust=False)

        # 1. ML Logic via predict.py
        risk_tickers = list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys())))
        ml_res = predict_assets(
            model_path="risk_model.joblib",
            tickers=risk_tickers,
            friendly_names={},
            model_type="risk"
        )

        probs = ml_res.get('probabilities', {})
        is_risk_on = probs.get('Class 1', 0) > probs.get('Class 0', 0)
        confidence = round(max(probs.values()) * 100, 1) if probs else 0

        # 2. Heuristic Logic for UI Display (using already downloaded data)
        data = raw['Close'].ffill()

        # Pre-calculate values
        spy = data['SPY']
        spy_last = spy.iloc[-1]
        spy_200ma = spy.rolling(200, min_periods=1).mean().iloc[-1]

        vix_last = data['^VIX'].iloc[-1]
        move_last = data['^MOVE'].iloc[-1]

        breadth_ratio = data['RSP'] / spy
        breadth_ma = breadth_ratio.rolling(50, min_periods=1).mean().iloc[-1]
        breadth_last = breadth_ratio.iloc[-1]

        # Boolean conversions
        trend_pass = bool(spy_last > spy_200ma)
        fear_pass = bool(vix_last < 20 and move_last < 110)
        breadth_pass = bool(breadth_last > breadth_ma)

        return {
            "status": "RISK-ON" if is_risk_on else "RISK-OFF",
            "confidence": confidence,
            "details": [
                {"label": "Trend (SPY > 200MA)", "pass": trend_pass},
                {"label": "Fear (VIX/MOVE Low)", "pass": fear_pass},
                {"label": "Breadth (RSP > MA)", "pass": breadth_pass}
            ]
        }
    except Exception as e:
        print(f"Risk Regime Error: {e}")
        return {"status": "ERROR", "details": []}


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
@ttl_cache(30)
def get_sector_rotation():
    try:
        tickers = list(SECTORS.keys()) + ["SPY"]
        data = yf.download(tickers, period="1y", progress=False, auto_adjust=False)['Close']

        # Vectorized relative performance calculation
        sector_data = data[list(SECTORS.keys())]
        rel = sector_data.div(data["SPY"], axis=0).pct_change(63).iloc[-1]

        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            rel_gain = rel[t]
            out.append({
                "name": name,
                "gain": f"{rel_gain:+.2%}",
                "is_positive": rel_gain > 0,
                "r2": r2,
                "slope": slope
            })

        # Single sort operation
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
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2})
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
            results.append({"sym": s, "name": n, "slope": slope, "r2": r2})
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
            results.append({"sym": s, "name": n, "slope": round(slope, 4), "r2": r2})

        return results
    except:
        return []


@ttl_cache(30)
def get_trends():
    from macro.ml_engine import get_ml_confidence
    results = []

    for sym, name in TREND_ASSETS.items():
        try:
            df = yf.download(sym, period="1y", progress=False, auto_adjust=False)
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
            # Optimized: calculate slopes in vectorized manner where possible
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