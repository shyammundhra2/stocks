"""
S&P 500 stock-selection ranker.

REWRITTEN 2026-08-25 around the only rule that survived an honest walk-forward
backtest (backtest_stock_xsection.py / _combo.py / _bufgate.py, 2005-2026):

  Cross-sectional 12-1 momentum (12-month return skipping the last month),
  hold the TOP QUINTILE equal-weight, GATED to cash when SPY < its 200-DMA.

Full-cycle vs the fair benchmark (equal-weight universe): Sharpe 1.22 vs 0.99,
maxDD -18.6% vs -47%. The edge is survival/drawdown control (the 200-DMA gate
avoids the 2008 momentum crash), NOT return alpha - momentum SELECTION alone
~ties equal-weight; the gate is the driver. Same architecture as the GSS book.

The prior macro-ML model (macro_model_2027.json) was RETIRED: ~all its features
were market-wide (identical across stocks on a date), so it couldn't discriminate
cross-sectionally, and it had no holdout / overlapping 1y labels. Kept on disk
for reference; no longer loaded.

HONEST CAVEATS baked into the reporting: survivorship bias (current-membership
S&P 500 inflates absolute returns; the relative edge is the trustworthy part),
10bps costs assumed, exact (not buffered) gate - buffered LOST here because
momentum's tail is sharp crashes where a lagged exit/entry costs more than the
whipsaw it saves.
"""
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
LOCAL_FILE = "sp500.csv"
SECTOR_FILE = "sp500_sectors.csv"
OUTPUT_CSV = "macro_2027_ranking_enhanced.csv"

MOM_LOOKBACK = 252      # 12 months
MOM_SKIP = 21           # skip last month (12-1 momentum)
# Concentration: hold the top TOP_N momentum names equal-weight (this is a small
# ~1%-of-net-worth / $50k RETURN sleeve, so we concentrate for return - drawdown
# is tolerable at this size). backtest_stock_concentration.py (2005-26, gated):
# N=20 Sharpe 1.29 / CAGR 26% / maxDD -26% - Sharpe PLATEAUS by N=20, so tighter
# (N=10) buys +7% CAGR at zero Sharpe gain = pure added risk AND the most
# survivorship-flattered number. N=20 keeps each name at 5% (one blowup != gutted).
# Knob: 10 = max return / more blowup+survivorship risk; 30 = more diversified.
TOP_N = 20
# Sector cap: at most this many BUYs per GICS sector (backtest_stock_sectorcap.py).
# Sectors are a cleaner diversification axis than trailing correlation: on the
# 10-name book the <=3/sector cap edged out naive top-10 (full Sharpe 1.32 vs
# 1.30, 2020-26 1.22 vs 1.14, lower vol) at trivial CAGR cost, and beat the
# correlation-constrained pick (1.27). Scaled to TOP_N here (~ceil(N/5)) so no
# single sector dominates the sleeve. Set to None to disable. (For a tight
# 10-name book use TOP_N=10, MAX_PER_SECTOR=3.)
MAX_PER_SECTOR = 4
DMA_200 = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
BOOK_NOTIONAL = 50_000   # sleeve size (~1% of net worth)

BATCH_SIZE = 50
RETRY_DELAY = 2


# ---------------- INDICATORS ----------------
def calculate_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_rsi(close, period=14):
    if len(close) < period + 1:
        return pd.Series([np.nan] * len(close), index=close.index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / loss))


def calculate_slope_r2(close, period=21):
    """1-month log-price OLS slope (%/day) and R^2 - display only."""
    if len(close) < period:
        return np.nan, np.nan
    y = np.log(close.tail(period).values.astype(float))
    x = np.arange(len(y))
    m = ~np.isnan(y)
    if m.sum() < 2:
        return np.nan, np.nan
    x, y = x[m], y[m]
    xc = x - x.mean()
    slope = float((xc @ (y - y.mean())) / (xc @ xc))
    yhat = slope * x + (y.mean() - slope * x.mean())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - yhat) ** 2).sum()) / sst if sst > 0 else 0.0
    return round(slope * 100, 3), round(max(min(r2, 1.0), 0.0), 3)


def momentum_12_1(close):
    """12-1 momentum: return from t-252 to t-21 (skip last month)."""
    if len(close) < MOM_LOOKBACK + 1:
        return np.nan
    p_now = close.iloc[-1 - MOM_SKIP]
    p_then = close.iloc[-MOM_LOOKBACK]
    if not (np.isfinite(p_now) and np.isfinite(p_then)) or p_then <= 0:
        return np.nan
    return float(p_now / p_then - 1.0)


# ---------------- DATA ----------------
def download_stocks_batch(tickers, start, end, batch_size=50):
    all_results = []
    total = (len(tickers) + batch_size - 1) // batch_size
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"📦 batch {(i // batch_size) + 1}/{total} ...")
        for _ in range(3):
            try:
                if i > 0:
                    time.sleep(RETRY_DELAY)
                data = yf.download(batch, start=start, end=end, auto_adjust=True,
                                   progress=False, group_by='ticker', threads=True)
                for t in batch:
                    try:
                        sd = data[t] if len(batch) > 1 else data
                        if not sd.empty:
                            all_results.append({'ticker': t, 'data': sd})
                    except Exception:
                        continue
                break
            except Exception as e:
                print(f"   retry ({str(e)[:40]})"); time.sleep(RETRY_DELAY)
    return all_results


def load_company_names():
    try:
        df = pd.read_csv(LOCAL_FILE)
        return dict(zip(df['Symbol'], df['Company']))
    except Exception:
        return {}


def load_sectors():
    """Ticker -> GICS sector (cached CSV). Missing names get their own ticker as
    'sector' so they're never capped away."""
    try:
        df = pd.read_csv(SECTOR_FILE)
        return dict(zip(df['Symbol'], df['Sector']))
    except Exception:
        return {}


# ---------------- INFERENCE ----------------
def infer_today(top_n=25):
    print("🚀 Momentum(12-1) + SPY-200DMA gate ranker\n")
    end = datetime.now()
    start = end - timedelta(days=650)          # ~450 trading days > 273 needed
    print(f"📅 {start.date()} → {end.date()}")

    # SPY 200-DMA gate (exact threshold - buffered lost in backtest)
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    spy = pd.Series(np.asarray(spy).ravel(), index=spy.index)
    spy_200 = spy.rolling(DMA_200).mean()
    gate_risk_off = bool(spy.iloc[-1] < spy_200.iloc[-1]) if np.isfinite(spy_200.iloc[-1]) else False
    gate_txt = "RISK-OFF (SPY<200DMA → book to CASH)" if gate_risk_off else "RISK-ON (SPY>200DMA)"
    print(f"🚦 Gate: {gate_txt}   SPY {spy.iloc[-1]:.2f} vs 200DMA {spy_200.iloc[-1]:.2f}\n")

    names = load_company_names()
    sectors = load_sectors()
    symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()
    results = download_stocks_batch(symbols, start, end, BATCH_SIZE)
    print(f"✅ {len(results)}/{len(symbols)} stocks\n")

    rows = []
    for r in results:
        t, data = r['ticker'], r['data']
        if 'Close' not in data or len(data) < MOM_LOOKBACK + 1:
            continue
        close = data['Close']
        mom = momentum_12_1(close)
        if not np.isfinite(mom):
            continue
        sma200 = close.rolling(DMA_200).mean().iloc[-1]
        slope1m, r2_1m = calculate_slope_r2(close, 21)
        rows.append({
            'Ticker': t, 'Name': names.get(t, t),
            'Mom_12_1': round(mom, 4),
            'Current_Price': round(float(close.iloc[-1]), 2),
            'SMA_200': round(float(sma200), 2) if np.isfinite(sma200) else np.nan,
            'RSI': round(float(calculate_rsi(close, RSI_PERIOD).iloc[-1]), 1),
            'ATR': round(float(calculate_atr(data['High'], data['Low'], close, ATR_PERIOD).iloc[-1]), 2),
            'Slope_1M': slope1m, 'R2_1M': r2_1m,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("❌ no valid stocks"); return df

    # Cross-sectional momentum rank (percentile) -> repurposes 'Prob' for
    # downstream compatibility. BUY = the TOP_N momentum names (matches the
    # validated top-N rule); the SPY-200DMA gate handles crash protection.
    df['Prob'] = df['Mom_12_1'].rank(pct=True).round(3)
    df['Sector'] = df['Ticker'].map(lambda t: sectors.get(t, t))
    df = df.sort_values('Mom_12_1', ascending=False).reset_index(drop=True)

    # Momentum-ranked walk, admitting a name only if its sector is under the cap.
    buy_set, _sec_cnt = set(), {}
    if not gate_risk_off:
        for _, r in df.iterrows():
            s = r['Sector']
            if MAX_PER_SECTOR is None or _sec_cnt.get(s, 0) < MAX_PER_SECTOR:
                buy_set.add(r['Ticker']); _sec_cnt[s] = _sec_cnt.get(s, 0) + 1
            if len(buy_set) == TOP_N:
                break

    def signal(row):
        if gate_risk_off:
            return "GATE-CASH"                       # market gate overrides all
        if row['Ticker'] in buy_set:
            return "BUY"                             # top-N momentum
        return "HOLD"

    df['Signal'] = df.apply(signal, axis=1)
    n_buy = int((df['Signal'] == "BUY").sum())
    per_name = (1.0 / n_buy) if n_buy else 0.0       # equal-weight the top-N
    df['Size_Allocation'] = np.where(df['Signal'] == "BUY",
                                     round(per_name * BOOK_NOTIONAL, 2), 0.0)

    df = df.sort_values('Prob', ascending=False)
    out_cols = ['Ticker', 'Name', 'Sector', 'Prob', 'Mom_12_1', 'Signal', 'Slope_1M',
                'R2_1M', 'RSI', 'Current_Price', 'Size_Allocation', 'ATR', 'SMA_200']
    df['Gate'] = gate_txt

    print("=" * 120)
    print(f"TOP MOMENTUM (12-1) HOLDS   [{gate_txt}]")
    print("=" * 120)
    print(df[out_cols].head(top_n).to_string(index=False))

    df[out_cols + ['Gate']].to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ saved → {OUTPUT_CSV}")

    print(f"\n📊 {len(df)} ranked | BUY {n_buy} (top-{TOP_N}, <={MAX_PER_SECTOR}/sector) | "
          f"HOLD {(df['Signal']=='HOLD').sum()} | "
          f"{'ALL CASH (gate risk-off)' if gate_risk_off else 'gate risk-on'}")
    if n_buy:
        mix = df[df['Signal'] == 'BUY']['Sector'].value_counts().to_dict()
        print(f"   equal-weight {per_name:.1%} each (${per_name*BOOK_NOTIONAL:,.0f} on ${BOOK_NOTIONAL:,})")
        print(f"   sector mix: {mix}")
    return df


if __name__ == "__main__":
    infer_today()
