# macro/indicators.py
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

# =========================
# TTL Cache Decorator
# =========================
def ttl_cache(ttl_seconds=30):
    """Simple TTL cache decorator."""
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
# Internal Helpers
# =========================
def _safe_r2(y, coeffs):
    if len(y) < 2:
        return 0.0
    y_hat = np.polyval(coeffs, np.arange(len(y)))
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


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
# ML Sector Prediction
# =========================
@ttl_cache(30)
def get_ml_sector_prediction():
    try:
        model_path = 'sector_model.joblib'
        if not os.path.exists(model_path):
            return {
                "top": {"name": "N/A", "ticker": "N/A", "confidence": 0},
                "bottom": {"name": "N/A", "ticker": "N/A", "confidence": 0},
                "spread": 0,
                "all": []
            }

        bundle = joblib.load(model_path)
        model = bundle['model']
        scaler = bundle['scaler']

        raw = yf.download(
            ML_MACRO_TICKERS,
            period="20d",
            progress=False,
            multi_level_index=False
        )['Close'].ffill()

        X = pd.DataFrame(index=[raw.index[-1]])
        X['DXY_mom'] = float(raw['DX-Y.NYB'].pct_change(10).iloc[-1])
        X['VIX_level'] = float(raw['^VIX'].iloc[-1])
        X['MOVE_level'] = float(raw['^MOVE'].iloc[-1])
        X['Yield_Curve'] = float(raw['^TYX'].iloc[-1] - raw['^TNX'].iloc[-1])
        X['Credit_Spread'] = float(raw['LQD'].iloc[-1] / raw['HYG'].iloc[-1])
        X['TNX_vol'] = float(raw['^TNX'].rolling(10).std().iloc[-1])

        probs = model.predict_proba(scaler.transform(X))[0]
        classes = model.classes_

        ranked = sorted(
            [
                {
                    "ticker": t,
                    "name": SECTOR_NAMES.get(t, t),
                    "confidence": round(p * 100, 1)
                }
                for t, p in zip(classes, probs)
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )

        return {
            "top": ranked[0],
            "bottom": ranked[-1],
            "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"], 1),
            "all": ranked
        }

    except Exception as e:
        print(f"ML Sector Error: {e}")
        return {
            "top": {"name": "Error", "ticker": "ERR", "confidence": 0},
            "bottom": {"name": "Error", "ticker": "ERR", "confidence": 0},
            "spread": 0,
            "all": []
        }


# =========================
# Risk Regime
# =========================
@ttl_cache(30)
def get_risk_regime():
    try:
        data = yf.download(
            ["SPY", "^VIX", "RSP"],
            period="300d",
            progress=False,
            multi_level_index=False
        )['Close']

        spy = data["SPY"]
        vix = data["^VIX"]
        rsp = data["RSP"]

        m1 = float(spy.iloc[-1]) > float(spy.rolling(200).mean().iloc[-1])
        m2 = float(vix.iloc[-1]) < 20
        ratio = rsp / spy
        m3 = float(ratio.iloc[-1]) > float(ratio.rolling(50).mean().iloc[-1])

        status = "RISK-ON" if sum([m1, m2, m3]) >= 2 else "RISK-OFF"

        return {
            "status": status,
            "details": [
                {"label": "Trend", "pass": bool(m1)},
                {"label": "Fear", "pass": bool(m2)},
                {"label": "Breadth", "pass": bool(m3)}
            ]
        }
    except:
        return {"status": "UNKNOWN", "details": []}


# =========================
# VIX Signal
# =========================
@ttl_cache(30)
def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False, multi_level_index=False)['Close'].squeeze()
        v_last = float(vix.iloc[-1])
        z = float((v_last - vix.tail(50).mean()) / vix.tail(50).std())

        if z > 2.0:
            sig = "AGGRESSIVE_BUY"
        elif z > 1.0:
            sig = "SCALE_IN"
        elif z < -1.5:
            sig = "TRIM_PROFITS"
        else:
            sig = "NEUTRAL"

        return {"vix": round(v_last, 2), "z": round(z, 2), "signal": sig}
    except:
        return {"vix": 0, "z": 0, "signal": "ERROR"}


# =========================
# Mean Reversion
# =========================
@ttl_cache(30)
def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=True, progress=False, multi_level_index=False)
        c = df["Close"].squeeze()

        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        price = float(c.iloc[-1])
        s200 = float(c.rolling(200).mean().iloc[-1])

        if rsi2 >= 70:
            sig = "EXIT"
        elif price < s200:
            sig = "RISK OFF"
        elif rsi2 <= 10:
            sig = "BUY"
        else:
            sig = "HOLD"

        return {"price": round(price, 2), "rsi2": round(rsi2, 1), "signal": sig}
    except:
        return {"price": 0, "rsi2": 0, "signal": "ERROR"}


# =========================
# Sector Rotation
# =========================
@ttl_cache(30)
def get_sector_rotation():
    try:
        data = yf.download(list(SECTORS.keys()) + ["SPY"], period="1y", progress=False, multi_level_index=False)['Close']
        rel = data[list(SECTORS.keys())].div(data["SPY"], axis=0).pct_change(63).iloc[-1]

        out = []
        for t, name in SECTORS.items():
            slope, r2 = _trend_stats(data[t], 20, 20)
            out.append({
                "name": name,
                "gain": f"{rel[t]:+.2%}",
                "is_positive": rel[t] > 0,
                "r2": r2,
                "slope": slope
            })

        return {"all_ranked": sorted(out, key=lambda x: float(x["gain"].strip("%")), reverse=True)}
    except:
        return {"all_ranked": []}


# =========================
# Country Rotation
# =========================
@ttl_cache(30)
def get_country_rotation():
    try:
        data = yf.download(list(COUNTRIES.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        return [
            {
                "sym": sym,
                "name": name,
                "slope": _trend_stats(data[sym], 60, 60)[0],
                "r2": _trend_stats(data[sym], 60, 60)[1]
            }
            for sym, name in COUNTRIES.items()
        ]
    except:
        return []


# =========================
# Commodity Rotation
# =========================
@ttl_cache(30)
def get_commodity_rotation():
    try:
        data = yf.download(list(COMMODITIES.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        return [
            {
                "sym": sym,
                "name": name,
                "slope": _trend_stats(data[sym], 60, 60)[0],
                "r2": _trend_stats(data[sym], 60, 60)[1]
            }
            for sym, name in COMMODITIES.items()
        ]
    except:
        return []


# =========================
# Currency Rotation
# =========================
@ttl_cache(30)
def get_currency_rotation():
    try:
        data = yf.download(list(CURRENCIES.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        out = []
        for sym, name in CURRENCIES.items():
            c = data[sym].dropna().tail(60)
            c = c if sym in ["EURUSD=X", "GBPUSD=X", "AUDUSD=X"] else 1 / c
            slope, r2 = _trend_stats(c, 60, 60)
            out.append({"sym": sym, "name": name, "slope": round(slope, 4), "r2": r2})
        return out
    except:
        return []


# =========================
# Trends
# =========================
@ttl_cache(30)
def get_trends():
    from macro.ml_engine import get_ml_confidence
    results = []

    for sym, name in TREND_ASSETS.items():
        try:
            df = yf.download(sym, period="1y", progress=False, multi_level_index=False)
            c = df["Close"].squeeze()

            slope, r2 = _trend_stats(c, 10, 10)
            ml_conf = get_ml_confidence(df)

            atr = float(compute_ATR(df, 14).iloc[-1])
            last = float(c.iloc[-1])
            stop = float(c.tail(20).max()) - (2.5 * atr)

            s50 = float(c.rolling(50).mean().iloc[-1])
            s200 = float(c.rolling(200).mean().iloc[-1])

            bullish = (s50 > s200) and (last > s50)
            strong = (slope > 0) and (r2 > 0.6)

            if last < stop or r2 < 0.3:
                status = "SELL"
            elif bullish and strong:
                status = "STRONG BUY" if ml_conf > 75 else "BUY"
            else:
                status = "HOLD"

            results.append({
                "sym": sym,
                "name": name,
                "price": round(last, 2),
                "status": status,
                "r2": round(r2, 2),
                "ml_conf": ml_conf,
                "rsi14": round(float(compute_RSI(c, 14).iloc[-1]), 1),
                "slope": round(slope, 2),
                "stop": round(stop, 2)
            })
        except:
            continue

    return sorted(results, key=lambda x: x["ml_conf"], reverse=True)


# =========================
# ML Country Prediction (FIXED)
# =========================
@ttl_cache(30)
def get_ml_country_prediction():
    from predict import predict_assets
    try:
        result = predict_assets(
            model_path="country_model.joblib",
            tickers=list(COUNTRIES.keys()) + ML_MACRO_TICKERS,
            friendly_names=COUNTRIES,
            model_type="country",
            top_n=3
        )
        ranked = [{"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v*100,1)} 
                  for k,v in result["probabilities"].items()]
        return {
            "top": ranked[0],
            "bottom": ranked[-1],
            "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"],1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Country Error: {e}")
        return {
            "top": {"name": "N/A", "ticker": "N/A", "confidence": 0},
            "bottom": {"name": "N/A", "ticker": "N/A", "confidence": 0},
            "spread": 0,
            "all": []
        }


# =========================
# ML Commodity Prediction (FIXED)
# =========================
@ttl_cache(30)
def get_ml_commodity_prediction():
    from predict import predict_assets
    try:
        result = predict_assets(
            model_path="commodity_model.joblib",
            tickers=list(COMMODITIES.keys()) + ML_MACRO_TICKERS,
            friendly_names=COMMODITIES,
            model_type="commodity",
            top_n=3
        )
        ranked = [{"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v*100,1)} 
                  for k,v in result["probabilities"].items()]
        return {
            "top": ranked[0],
            "bottom": ranked[-1],
            "spread": round(ranked[0]["confidence"] - ranked[-1]["confidence"],1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Commodity Error: {e}")
        return {
            "top": {"name": "N/A", "ticker": "N/A", "confidence": 0},
            "bottom": {"name": "N/A", "ticker": "N/A", "confidence": 0},
            "spread": 0,
            "all": []
        }

