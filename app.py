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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def compute_ATR(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# --- MODULE 1: RISK REGIME ---
def get_risk_regime():
    try:
        data = yf.download(["SPY", "^VIX", "RSP"], period="300d", progress=False)['Close']
        spy_last = float(data["SPY"].iloc[-1])
        spy_200ma = float(data["SPY"].rolling(200).mean().iloc[-1])
        m1 = spy_last > spy_200ma
        vix_last = float(data["^VIX"].iloc[-1])
        m2 = vix_last < 20
        ratio = data["RSP"] / data["SPY"]
        m3 = float(ratio.iloc[-1]) > float(ratio.rolling(50).mean().iloc[-1])
        status = "RISK-ON" if (sum([m1, m2, m3]) >= 2) else "RISK-OFF"
        return {"status": status, "details": [{"label": "Trend", "pass": m1}, {"label": "Fear", "pass": m2}, {"label": "Breadth", "pass": m3}]}
    except:
        return {"status": "UNKNOWN", "details": []}

# --- MODULE 2: MEAN REVERSION (QQQ) ---
def get_mean_reversion():
    try:
        df = yf.download("QQQ", period="400d", auto_adjust=True, progress=False)
        rsi2_series = compute_RSI(df["Close"], 2)
        p = float(df["Close"].iloc[-1])
        s200 = float(df["Close"].rolling(200).mean().iloc[-1])
        rsi2 = float(rsi2_series.iloc[-1])
        if rsi2 >= 70: sig = "EXIT"
        elif p < s200: sig = "RISK_OFF"
        elif rsi2 <= 10: sig = "STRONG_BUY"
        else: sig = "HOLD"
        return {"price": round(p, 2), "rsi2": round(rsi2, 1), "signal": sig}
    except:
        return {"price": 0, "rsi2": 0, "signal": "ERROR"}

# --- MODULE 5: VIX MEAN REVERSION (SPY) ---
def get_vix_signal():
    try:
        vix = yf.download("^VIX", period="100d", progress=False)['Close']
        vix_last = float(vix.iloc[-1])
        vix_ma = float(vix.tail(50).mean())
        vix_std = float(vix.tail(50).std())
        z = (vix_last - vix_ma) / vix_std
        if z > 2.0: sig = "AGGRESSIVE_BUY"
        elif z > 1.0: sig = "SCALE_IN"
        elif z < -1.5: sig = "TRIM_PROFITS"
        else: sig = "NEUTRAL"
        return {"vix": round(vix_last, 2), "z": round(z, 2), "signal": sig}
    except:
        return {"vix": 0, "z": 0, "signal": "ERROR"}

# --- MODULE 3: SECTOR ROTATION ---
def get_sector_rotation():
    try:
        sectors = {"XLC": "Comm Services", "XLY": "Consumer Discr", "XLP": "Consumer Staples", "XLE": "Energy", 
                   "XLF": "Financials", "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials", "XLRE": "Real Estate", "XLK": "Technology", "XLU": "Utilities"}
        data = yf.download(list(sectors.keys()) + ["SPY"], period="6mo", progress=False)['Close']
        rel_mom = data[list(sectors.keys())].div(data["SPY"], axis=0).pct_change(20).iloc[-1]
        ranked = rel_mom.sort_values(ascending=False)
        all_r = [{"name": sectors[t], "gain": f"{rel_mom[t]:+.2%}", "is_positive": rel_mom[t] > 0} for t in ranked.index]
        return {"top_3": all_r[:3], "all_ranked": all_r}
    except:
        return {"top_3": [], "all_ranked": []}

# --- MODULE 4: TREND FOLLOWING ---
def get_trends():
    assets = {"VGT": "Tech", "VDE": "Energy", "VIS": "Industrials", "XME": "Metals", "GLD": "Gold", "IBIT": "Bitcoin", "TLT": "30 yr Treasuries"}
    results = []
    for sym, name in assets.items():
        try:
            df = yf.download(sym, period="1y", progress=False)
            close_series = df['Close']
            sma50, sma200 = close_series.rolling(50).mean(), close_series.rolling(200).mean()
            rsi14, atr14, atr6m = compute_RSI(close_series, 14), compute_ATR(df, 14), compute_ATR(df, 126)
            last_Close, last_SMA50, last_SMA200 = float(close_series.iloc[-1]), float(sma50.iloc[-1]), float(sma200.iloc[-1])
            last_RSI14, last_ATR14, last_ATR6M = float(rsi14.iloc[-1]), float(atr14.iloc[-1]), float(atr6m.iloc[-1])
            recent_high = float(close_series.rolling(50).max().iloc[-1])
            buy_cond = (last_SMA50 > last_SMA200) and (last_ATR14 < 2 * last_ATR6M) and (last_RSI14 < 70) and (last_Close > last_SMA50)
            atr_stop = recent_high - (10 * last_ATR14)
            sell_cond = last_Close < last_SMA50 or last_Close < atr_stop
            if buy_cond: status, reason = "BUY", "SMA Cross | Vol OK | RSI OK"
            elif sell_cond: status, reason = "SELL", "Below SMA50" if last_Close < last_SMA50 else "Stop Hit"
            else: status, reason = "HOLD", "Trend Neutral"
            results.append({"sym": sym, "name": name, "price": round(last_Close, 2), "stop": round(atr_stop, 2), "status": status, "reason": reason})
        except: continue
    return results

@app.route('/')
def index():
    return render_template('dashboard.html', 
                           regime=get_risk_regime(), 
                           mr=get_mean_reversion(), 
                           sr=get_sector_rotation(), 
                           trends=get_trends(),
                           vix_mr=get_vix_signal())

if __name__ == "__main__":
    app.run(debug=True)
