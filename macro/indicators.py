# macro/indicators.py
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import os
import time
from functools import wraps
from macro.helpers import compute_RSI, compute_ATR

# --- TTL Cache Decorator ---
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

# --- ML Sector Prediction ---
@ttl_cache(30)
def get_ml_sector_prediction():
    try:
        model_path = 'sector_model.joblib'
        if not os.path.exists(model_path):
            return {"top": {"name": "N/A", "ticker": "N/A", "confidence": 0}, 
                    "bottom": {"name": "N/A", "ticker": "N/A", "confidence": 0}, 
                    "spread": 0}

        data_bundle = joblib.load(model_path)
        model = data_bundle['model']
        scaler = data_bundle['scaler']
        
        # Fetch live macro data
        macro_tickers = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD'] 
        raw_data = yf.download(macro_tickers, period="20d", progress=False, multi_level_index=False)['Close']
        raw_data = raw_data.ffill()
        
        # Feature Engineering
        X_live = pd.DataFrame(index=[raw_data.index[-1]])
        X_live['DXY_mom'] = float(raw_data['DX-Y.NYB'].pct_change(10).iloc[-1])
        X_live['VIX_level'] = float(raw_data['^VIX'].iloc[-1])
        X_live['MOVE_level'] = float(raw_data['^MOVE'].iloc[-1])
        X_live['Yield_Curve'] = float(raw_data['^TYX'].iloc[-1] - raw_data['^TNX'].iloc[-1])
        X_live['Credit_Spread'] = float(raw_data['LQD'].iloc[-1] / raw_data['HYG'].iloc[-1])
        X_live['TNX_vol'] = float(raw_data['^TNX'].rolling(10).std().iloc[-1])

        # Predict Probabilities
        X_scaled = scaler.transform(X_live)
        probs = model.predict_proba(X_scaled)[0]
        classes = model.classes_
        
        sector_names = {
            'VDE':'Energy', 'XLB':'Materials', 'VIS':'Industrials', 
            'XLY':'Discr', 'XLF':'Financials', 'XLC':'Comm Serv', 
            'VGT':'Tech', 'XLV':'Health', 'XLP':'Staples', 
            'XLU':'Utilities', 'XLRE':'Real Estate'
        }

        results = []
        for i, prob in enumerate(probs):
            ticker = classes[i]
            results.append({
                "ticker": ticker,
                "name": sector_names.get(ticker, ticker),
                "confidence": round(prob * 100, 1)
            })
        
        ranked = sorted(results, key=lambda x: x['confidence'], reverse=True)

        return {
            "top": ranked[0],
            "bottom": ranked[-1],
            "spread": round(ranked[0]['confidence'] - ranked[-1]['confidence'], 1),
            "all": ranked
        }
    except Exception as e:
        print(f"ML Sector Error: {e}")
        return {"top": {"name": "Error", "ticker": "ERR", "confidence": 0}, 
                "bottom": {"name": "Error", "ticker": "ERR", "confidence": 0}, 
                "spread": 0}

# --- Risk Regime ---
@ttl_cache(30)
def get_risk_regime():
    try:
        data = yf.download(["SPY", "^VIX", "RSP"], period="300d", progress=False, multi_level_index=False)['Close']
        spy_last = float(data["SPY"].iloc[-1])
        spy_ma = float(data["SPY"].rolling(200).mean().iloc[-1])
        vix_last = float(data["^VIX"].iloc[-1])
        m1 = spy_last > spy_ma
        m2 = vix_last < 20
        ratio = data["RSP"] / data["SPY"]
        m3 = float(ratio.iloc[-1]) > float(ratio.rolling(50).mean().iloc[-1])
        status = "RISK-ON" if sum([m1, m2, m3]) >= 2 else "RISK-OFF"
        return {"status": status, "details": [{"label": "Trend", "pass": bool(m1)}, {"label": "Fear", "pass": bool(m2)}, {"label": "Breadth", "pass": bool(m3)}]}
    except: return {"status": "UNKNOWN", "details": []}

# --- VIX Signal ---
@ttl_cache(30)
def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False, multi_level_index=False)['Close'].squeeze()
        vix_last = float(vix.iloc[-1])
        z = float((vix_last - vix.tail(50).mean()) / vix.tail(50).std())
        sig = "NEUTRAL"
        if z > 2.0: sig = "AGGRESSIVE_BUY"
        elif z > 1.0: sig = "SCALE_IN"
        elif z < -1.5: sig = "TRIM_PROFITS"
        return {"vix": round(vix_last, 2), "z": round(z, 2), "signal": sig}
    except: return {"vix": 0, "z": 0, "signal": "ERROR"}

# --- Mean Reversion ---
@ttl_cache(30)
def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=True, progress=False, multi_level_index=False)
        c = df["Close"].squeeze()
        rsi2 = float(compute_RSI(c, 2).iloc[-1])
        p = float(c.iloc[-1])
        s200 = float(c.rolling(200).mean().iloc[-1])
        sig = "HOLD"
        if rsi2 >= 70: sig = "EXIT"
        elif p < s200: sig = "RISK OFF"
        elif rsi2 <= 10: sig = "BUY"
        return {"price": round(p, 2), "rsi2": round(rsi2, 1), "signal": sig}
    except: return {"price": 0, "rsi2": 0, "signal": "ERROR"}

# --- Sector Rotation ---
@ttl_cache(30)
def get_sector_rotation():
    sectors = {"XLC":"Comm Serv","XLY":"Discr","XLP":"Staples","XLE":"Energy","XLF":"Financials","XLV":"Health","XLI":"Industrials","XLB":"Materials","XLRE":"Real Estate","XLK":"Tech","XLU":"Utilities"}
    try:
        data = yf.download(list(sectors.keys()) + ["SPY"], period="1y", progress=False, multi_level_index=False)['Close']
        rel_mom = data[list(sectors.keys())].div(data["SPY"], axis=0).pct_change(63).iloc[-1]
        all_r = []
        for t, label in sectors.items():
            y = np.log(data[t].dropna().tail(20).values)
            coeffs = np.polyfit(np.arange(len(y)), y, 1)
            r2 = 1 - (np.sum((y - np.polyval(coeffs, np.arange(len(y))))**2)/np.sum((y - np.mean(y))**2))
            all_r.append({"name": label, "gain": f"{rel_mom[t]:+.2%}", "is_positive": rel_mom[t] > 0, "r2": round(float(r2),2), "slope": round(float(coeffs[0])*20*100,2)})
        return {"all_ranked": sorted(all_r, key=lambda x: float(x["gain"].strip("%")), reverse=True)}
    except: return {"all_ranked": []}

# --- Country Rotation ---
@ttl_cache(30)
def get_country_rotation():
    countries = {"SPY":"USA","EFA":"Dev ex-US","EEM":"Emerging","EWJ":"Japan","EWZ":"Brazil","INDA":"India","FXI":"China","EWU":"UK","EWG":"Germany"}
    results = []
    try:
        data = yf.download(list(countries.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        for sym, name in countries.items():
            c = data[sym].dropna().tail(60)
            y = np.log(c.values)
            coeffs = np.polyfit(np.arange(len(y)), y, 1)
            r2 = 1 - (np.sum((y - np.polyval(coeffs, np.arange(len(y))))**2)/np.sum((y - np.mean(y))**2))
            results.append({"sym": sym, "name": name, "slope": round(float(coeffs[0])*60*100,2), "r2": round(float(r2),2)})
        return results
    except: return []

# --- Commodity Rotation ---
@ttl_cache(30)
def get_commodity_rotation():
    commodities = {"DBC":"Broad Commodities","GC=F":"Gold","SI=F":"Silver","HG=F":"Copper","ALI=F":"Aluminium","PL=F":"Platinum","PA=F":"Palladium","CL=F":"Crude Oil (WTI)","BZ=F":"Brent Oil","NG=F":"Natural Gas","ZS=F":"Soybeans","ZC=F":"Corn","ZW=F":"Wheat","KC=F":"Coffee","SB=F":"Sugar","CT=F":"Cotton","LE=F":"Live Cattle","HE=F":"Lean Hogs","LBR=F":"Lumber"}
    results = []
    try:
        data = yf.download(list(commodities.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        for sym, name in commodities.items():
            c = data[sym].dropna().tail(60)
            y = np.log(c.values)
            coeffs = np.polyfit(np.arange(len(y)), y, 1)
            r2 = 1 - (np.sum((y - np.polyval(coeffs, np.arange(len(y))))**2)/np.sum((y - np.mean(y))**2))
            results.append({"sym": sym, "name": name, "slope": round(float(coeffs[0])*60*100,2), "r2": round(float(r2),2)})
        return results
    except: return []

# --- Currency Rotation ---
@ttl_cache(30)
def get_currency_rotation():
    currencies = {"EURUSD=X":"EUR","JPY=X":"JPY","GBPUSD=X":"GBP","AUDUSD=X":"AUD", "USDINR=X":"INR"}
    results = []
    try:
        data = yf.download(list(currencies.keys()), period="1y", progress=False, multi_level_index=False)['Close']
        for sym, curr in currencies.items():
            c = data[sym].dropna().tail(60)
            c_usd = c if sym in ["EURUSD=X","GBPUSD=X","AUDUSD=X"] else 1/c
            y = np.log(c_usd.values)
            coeffs = np.polyfit(np.arange(len(y)), y, 1)
            r2 = 1 - (np.sum((y - np.polyval(coeffs, np.arange(len(y))))**2)/np.sum((y - np.mean(y))**2))
            results.append({"sym": sym, "name": curr, "slope": round(float(coeffs[0])*60*100,4), "r2": round(float(r2),2)})
        return results
    except: return []

# --- Trends ---
@ttl_cache(30)
def get_trends():
    assets = {"VGT":"Tech","VDE":"Energy","VIS":"Industrials","XME":"Metals","GLD":"Gold","IBIT":"Bitcoin","TLT":"30yr Bond"}
    results = []
    from macro.ml_engine import get_ml_confidence
    for sym, name in assets.items():
        try:
            df = yf.download(sym, period="1y", progress=False, multi_level_index=False)
            c = df["Close"].squeeze()
            y = np.log(c.tail(10).values)
            coeffs = np.polyfit(np.arange(len(y)), y, 1)
            r2 = float(1 - (np.sum((y - np.polyval(coeffs, np.arange(len(y))))**2)/np.sum((y - np.mean(y))**2)))
            ml_conf = get_ml_confidence(df)
            atr = float(compute_ATR(df, 14).iloc[-1])
            last_c = float(c.iloc[-1])
            trailing_stop = float(c.tail(20).max()) - (2.5 * atr)
            s50 = float(c.rolling(50).mean().iloc[-1])
            s200 = float(c.rolling(200).mean().iloc[-1])
            is_bullish = (s50 > s200) and (last_c > s50)
            is_strong = (float(coeffs[0]) > 0) and (r2 > 0.6)
            if last_c < trailing_stop or r2 < 0.3: status = "SELL"
            elif is_bullish and is_strong: status = "STRONG BUY" if ml_conf > 75 else "BUY"
            else: status = "HOLD"
            results.append({"sym": sym, "name": name, "price": round(last_c, 2), "status": status, "r2": round(r2, 2), "ml_conf": ml_conf, "rsi14": round(float(compute_RSI(c, 14).iloc[-1]), 1), "slope": round(float(coeffs[0])*10*100, 2), "stop": round(trailing_stop, 2)})
        except: continue
    return sorted(results, key=lambda x: x["ml_conf"], reverse=True)

