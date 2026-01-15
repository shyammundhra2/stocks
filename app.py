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
        elif p < s200: sig = "RISK OFF"
        elif rsi2 <= 10: sig = "BUY"
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
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            c = df['Close'].dropna()
            if len(c) < 50: continue 

            # --- 1. Momentum Stats (10-Day Window) ---
            window = 10
            y = np.log(c.tail(window).values)
            x = np.arange(len(y))
            coeffs = np.polyfit(x, y, 1)
            slope, r_squared = coeffs[0], 1 - (np.sum((y - np.polyval(coeffs, x))**2) / np.sum((y - np.mean(y))**2))
            pct_slope = slope * window 

            # --- 2. Macro & Volatility Indicators ---
            s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
            atr = compute_ATR(df, 14).iloc[-1]
            last_c = float(c.iloc[-1])
            
            # --- 3. Smart Exit Logic (The Trailing Stop) ---
            # Chandelier Exit: Highest high of last 20 days minus 2.5x ATR
            recent_high = c.tail(20).max()
            trailing_stop = recent_high - (2.5 * atr)

            # --- 4. Signal Generation ---
            # BUY: Golden Cross + Price > 50MA + Strong 10-day R2
            is_bullish_ma = (s50.iloc[-1] > s200.iloc[-1]) and (last_c > s50.iloc[-1])
            is_strong_mom = (slope > 0) and (r_squared > 0.6)
            
            buy_signal = is_bullish_ma and is_strong_mom

            # SELL Logic:
            # - Price breaks the 50-day MA (Macro Exit)
            # - Price hits the ATR Trailing Stop (Profit Protection)
            # - R2 collapses < 0.3 (Momentum Decay)
            if last_c < s50.iloc[-1] or last_c < trailing_stop:
                status = "SELL"
            elif r_squared < 0.3:
                status = "HOLD (Weakening)"
            elif buy_signal:
                status = "BUY"
            else:
                status = "HOLD"

            results.append({
                "sym": sym, "name": name, "price": round(last_c, 2),
                "status": status, "r2": round(r_squared, 2),
                "slope": round(pct_slope * 100, 2), "stop": round(trailing_stop, 2)
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
            continue
    return sorted(results, key=lambda x: x['r2'], reverse=True)

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
