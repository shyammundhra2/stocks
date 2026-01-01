from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- HELPERS ---
def compute_RSI(series, period=2):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50)

def compute_ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# --- MODULES ---
def get_risk_regime():
    try:
        data = yf.download(["SPY", "^VIX", "RSP"], period="300d", progress=False)['Close']
        spy_last, spy_200ma = data["SPY"].iloc[-1], data["SPY"].rolling(200).mean().iloc[-1]
        m1 = spy_last > spy_200ma
        vix_last = data["^VIX"].iloc[-1]
        m2 = vix_last < 20
        ratio = data["RSP"] / data["SPY"]
        m3 = ratio.iloc[-1] > ratio.rolling(50).mean().iloc[-1]
        status = "RISK-ON" if (sum([m1, m2, m3]) >= 2) else "RISK-OFF"
        return {"status": status, "details": [{"label": "Trend", "pass": bool(m1)}, {"label": "Fear", "pass": bool(m2)}, {"label": "Breadth", "pass": bool(m3)}]}
    except: return {"status": "UNKNOWN", "details": []}

def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False)['Close']
        vix_last = float(vix.iloc[-1])
        vix_ma, vix_std = float(vix.tail(50).mean()), float(vix.tail(50).std())
        z = (vix_last - vix_ma) / vix_std
        if z > 2.0: sig = "AGGRESSIVE_BUY"
        elif z > 1.0: sig = "SCALE_IN"
        elif z < -1.5: sig = "TRIM_PROFITS"
        else: sig = "NEUTRAL"
        return {"vix": round(vix_last, 2), "z": round(z, 2), "signal": sig}
    except: return {"vix": 0, "z": 0, "signal": "ERROR"}

def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=True, progress=False)
        rsi2 = float(compute_RSI(df["Close"], 2).iloc[-1])
        p, s200 = float(df["Close"].iloc[-1]), float(df["Close"].rolling(200).mean().iloc[-1])
        if rsi2 >= 70: sig = "EXIT"
        elif p < s200: sig = "RISK_OFF"
        elif rsi2 <= 10: sig = "STRONG_BUY"
        else: sig = "HOLD"
        return {"price": round(p, 2), "rsi2": round(rsi2, 1), "signal": sig}
    except: return {"price": 0, "rsi2": 0, "signal": "ERROR"}

def get_sector_rotation():
    try:
        sectors = {"XLC": "Comm Serv", "XLY": "Discr", "XLP": "Staples", "XLE": "Energy", "XLF": "Financials", "XLV": "Health", "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Tech", "XLU": "Utilities"}
        data = yf.download(list(sectors.keys()) + ["SPY"], period="6mo", progress=False)['Close']
        rel_mom = data[list(sectors.keys())].div(data["SPY"], axis=0).pct_change(20).iloc[-1]
        ranked = rel_mom.sort_values(ascending=False)
        all_r = [{"name": sectors[t], "gain": f"{rel_mom[t]:+.2%}", "is_positive": rel_mom[t] > 0} for t in ranked.index]
        return {"top_3": all_r[:3], "all_ranked": all_r}
    except: return {"top_3": [], "all_ranked": []}

def get_trends():
    assets = {"VGT": "Tech", "VDE": "Energy", "VIS": "Industrials", "XME": "Metals", "GLD": "Gold", "IBIT": "Bitcoin", "TLT": "30yr Bond"}
    results = []
    for sym, name in assets.items():
        try:
            df = yf.download(sym, period="1y", progress=False)
            c = df['Close']
            s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
            rsi14, atr = compute_RSI(c, 14), compute_ATR(df, 14)
            last_c, last_rsi = float(c.iloc[-1]), float(rsi14.iloc[-1])
            stop = float(c.rolling(50).max().iloc[-1]) - (10 * float(atr.iloc[-1]))
            buy = (float(s50.iloc[-1]) > float(s200.iloc[-1])) and (last_rsi < 85) and (last_c > float(s50.iloc[-1]))
            status = "BUY" if buy else ("SELL" if last_c < float(s50.iloc[-1]) or last_c < stop else "HOLD")
            results.append({"sym": sym, "name": name, "price": round(last_c, 2), "stop": round(stop, 2), "status": status, "rsi14": round(last_rsi, 1)})
        except: continue
    return sorted(results, key=lambda x: x['rsi14'], reverse=True)

@app.route('/')
def index():
    return render_template('dashboard.html', 
                           regime=get_risk_regime(), 
                           vix_mr=get_vix_signal(), 
                           mr=get_mean_reversion(), 
                           sr=get_sector_rotation(), 
                           trends=get_trends())

if __name__ == "__main__":
    app.run(debug=True)
