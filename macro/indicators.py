import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import time
from functools import wraps

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
# Internal Math Helpers
# =========================
def _safe_r2(y, coeffs):
    if len(y) < 2: return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

def _trend_stats(series, window, scale):
    c = series.dropna().tail(window)
    if len(c) < 5: return 0.0, 0.0
    y = np.log(c.values)
    coeffs = np.polyfit(np.arange(len(y)), y, 1)
    slope = float(coeffs[0]) * scale * 100
    r2 = _safe_r2(y, coeffs)
    return round(slope, 2), round(r2, 2)

# =========================
# I. Risk Regime (ML Core + Heuristic Display)
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
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

        # 2. Heuristic Logic for UI Display
        raw = yf.download(['SPY', 'RSP', '^VIX', '^MOVE'], period="300d", progress=False, auto_adjust=False)
        data = raw['Close'].ffill()

        # Trend Check
        spy_last = data['SPY'].iloc[-1]
        spy_200ma = data['SPY'].rolling(200).mean().iloc[-1]
        trend_pass = bool(spy_last > spy_200ma)

        # Fear Check
        fear_pass = bool(data['^VIX'].iloc[-1] < 20 and data['^MOVE'].iloc[-1] < 110)

        # Breadth Check
        breadth_ratio = data['RSP'] / data['SPY']
        breadth_ma = breadth_ratio.rolling(50).mean().iloc[-1]
        breadth_pass = bool(breadth_ratio.iloc[-1] > breadth_ma)

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
# II. ML Asset Predictions
# =========================
@ttl_cache(30)
def get_ml_sector_prediction():
    try:
        res = predict_assets("sector_model.joblib", list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS, SECTOR_NAMES, "sector")
        ranked = [{"ticker": k, "name": SECTOR_NAMES.get(k, k), "confidence": round(v*100, 1)} for k, v in res["probabilities"].items()]
        return {"top": ranked[0], "bottom": ranked[-1], "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Sector Error: {e}"); return {"top": {"name": "N/A", "ticker": "N/A", "confidence": 0}, "all": []}

@ttl_cache(30)
def get_ml_country_prediction():
    try:
        res = predict_assets("country_model.joblib", list(COUNTRIES.keys()) + ML_MACRO_TICKERS, COUNTRIES, "country")
        ranked = [{"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v*100, 1)} for k, v in res["probabilities"].items()]
        return {"top": ranked[0], "bottom": ranked[-1], "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Country Error: {e}"); return {"top": {"name": "N/A", "confidence": 0}, "all": []}

@ttl_cache(30)
def get_ml_commodity_prediction():
    try:
        res = predict_assets("commodity_model.joblib", list(COMMODITIES.keys()) + ML_MACRO_TICKERS, COMMODITIES, "commodity")
        ranked = [{"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v*100, 1)} for k, v in res["probabilities"].items()]
        return {"top": ranked[0], "bottom": ranked[-1], "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Commodity Error: {e}"); return {"top": {"name": "N/A", "confidence": 0}, "all": []}

# =========================
# III. Traditional Technical Signals
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False, auto_adjust=False)['Close'].squeeze()
        v_last = float(vix.iloc[-1])
        z = float((v_last - vix.tail(50).mean()) / vix.tail(50).std())
        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": "AGGRESSIVE_BUY" if z > 2.0 else "SCALE_IN" if z > 1.0 else "TRIM_PROFITS" if z < -1.5 else "NEUTRAL"}
    except: return {"vix": 0, "z": 0, "signal": "ERROR"}

@ttl_cache(30)
def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=False, progress=False)
        c = df["Close"].squeeze()
        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        price, s200 = float(c.iloc[-1]), float(c.rolling(200).mean().iloc[-1])
        return {"price": round(price, 2), "rsi2": round(rsi2, 1), "signal": "EXIT" if rsi2 >= 70 else "RISK OFF" if price < s200 else "BUY" if rsi2 <= 10 else "HOLD"}
    except: return {"price": 0, "rsi2": 0, "signal": "ERROR"}

# =========================
# IV. Rotation & Trends
# =========================
@ttl_cache(30)
def get_sector_rotation():
    try:
        data = yf.download(list(SECTORS.keys()) + ["SPY"], period="1y", progress=False, auto_adjust=False)['Close']
        rel = data[list(SECTORS.keys())].div(data["SPY"], axis=0).pct_change(63).iloc[-1]
        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            out.append({"name": name, "gain": f"{rel[t]:+.2%}", "is_positive": rel[t] > 0, "r2": r2, "slope": slope})
        return {"all_ranked": sorted(out, key=lambda x: float(x["gain"].strip("%")), reverse=True)}
    except: return {"all_ranked": []}

@ttl_cache(30)
def get_country_rotation():
    try:
        data = yf.download(list(COUNTRIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
        return [{"sym": s, "name": n, "slope": _trend_stats(data[s], 60, 60)[0], "r2": _trend_stats(data[s], 60, 60)[1]} for s, n in COUNTRIES.items()]
    except: return []

@ttl_cache(30)
def get_commodity_rotation():
    try:
        data = yf.download(list(COMMODITIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
        return [{"sym": s, "name": n, "slope": _trend_stats(data[s], 60, 60)[0], "r2": _trend_stats(data[s], 60, 60)[1]} for s, n in COMMODITIES.items()]
    except: return []

@ttl_cache(30)
def get_currency_rotation():
    try:
        data = yf.download(list(CURRENCIES.keys()), period="1y", progress=False, auto_adjust=False)['Close']
        out = []
        for s, n in CURRENCIES.items():
            c = data[s].dropna().tail(60)
            c = c if s in ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"] else 1 / c
            slope, r2 = _trend_stats(c, 60, 60)
            out.append({"sym": s, "name": n, "slope": round(slope, 4), "r2": r2})
        return out
    except: return []

@ttl_cache(30)
def get_trends():
    from macro.ml_engine import get_ml_confidence
    results = []
    for sym, name in TREND_ASSETS.items():
        try:
            df = yf.download(sym, period="1y", progress=False, auto_adjust=False)
            c = df["Close"].squeeze()
            slope, r2 = _trend_stats(c, 10, 10)
            ml_conf = get_ml_confidence(df)
            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            stop = float(c.tail(20).max()) - (2.5 * atr)
            s50, s200 = float(c.rolling(50).mean().iloc[-1]), float(c.rolling(200).mean().iloc[-1])
            
            if last < stop or r2 < 0.3: status = "SELL"
            elif (s50 > s200) and (last > s50) and (slope > 0) and (r2 > 0.6): status = "STRONG BUY" if ml_conf > 75 else "BUY"
            else: status = "HOLD"
            
            results.append({
                "sym": sym, "name": name, "price": round(last, 2), "status": status, 
                "r2": round(r2, 2), "ml_conf": ml_conf, "rsi14": round(float(compute_RSI(c, 14).iloc[-1]), 1), 
                "slope": round(slope, 2), "stop": round(stop, 2)
            })
        except: continue
    return sorted(results, key=lambda x: x["ml_conf"], reverse=True)
