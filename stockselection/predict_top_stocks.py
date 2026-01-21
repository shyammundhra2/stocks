import yfinance as yf
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
LOCAL_FILE = "sp500.csv"
MODEL_NAME = "macro_model_2027.json"

MACRO_TICKERS = ['SPY','RSP','^VIX','^MOVE','DX-Y.NYB',
                 '^IRX','^TNX','HYG','LQD']

SECTOR_TICKERS = ['XLK','XLF','XLI','XLY','XLE','XLV',
                  'XLP','XLU','XLB','XLRE','XLC']

# Lookback for 3M momentum (trading days)
LOOKBACK_3M = 63

# Extra days buffer for weekends/holidays
BUFFER_DAYS = LOOKBACK_3M * 2

def infer_today(top_n=15, avoid_n=15):
    # ---------------- LOAD MODEL ----------------
    model = xgb.XGBClassifier()
    model.load_model(MODEL_NAME)

    # ---------------- DATE RANGE ----------------
    end = datetime.now()
    start = end - timedelta(days=BUFFER_DAYS)

    # ---------------- MACRO & SECTOR ----------------
    ms = yf.download(MACRO_TICKERS + SECTOR_TICKERS,
                     start=start, end=end,
                     auto_adjust=True)['Close']

    m = pd.DataFrame(index=ms.index)
    m['Curve_Inversion'] = ms['^IRX'] - ms['^TNX']
    m['Bond_Stress'] = ms['^MOVE']
    m['Liquidity'] = ms['HYG'] / ms['LQD']
    m['Concentration'] = ms['SPY'] / ms['RSP']

    for s in SECTOR_TICKERS:
        m[f'{s}_3M'] = ms[s].pct_change(LOOKBACK_3M, fill_method=None)

    m = m.shift(1).dropna()
    macro_latest = m.iloc[-1].astype(float)

    # ---------------- STOCK FEATURES ----------------
    symbols = pd.read_csv(LOCAL_FILE)['Symbol'].tolist()
    rows = []

    for t in symbols:
        s = yf.download(t, start=start, end=end,
                        auto_adjust=True, progress=False)
        if s.empty or len(s) < LOOKBACK_3M:
            continue

        try:
            mom_3m = s['Close'].pct_change(LOOKBACK_3M, fill_method=None).iloc[-1].item()
        except (IndexError, TypeError, ValueError):
            continue

        if pd.isna(mom_3m):
            continue

        # Get company name safely
        try:
            name = yf.Ticker(t).info.get('shortName', t)
        except Exception:
            name = t

        row = macro_latest.copy()
        row['Stock_Mom_3M'] = mom_3m
        row['Ticker'] = t
        row['Name'] = name
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------- FEATURES ----------------
    feats = ['Curve_Inversion','Bond_Stress','Liquidity',
             'Concentration','Stock_Mom_3M'] + \
            [f'{s}_3M' for s in SECTOR_TICKERS]

    df[feats] = df[feats].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=feats)

    if df.empty:
        raise ValueError("No valid rows for inference. Check symbol list or Yahoo data.")

    # ---------------- PREDICTIONS ----------------
    df['Prob'] = model.predict_proba(df[feats])[:,1]
    df['Margin'] = model.predict(df[feats], output_margin=True)

    df['Confidence'] = pd.qcut(df['Prob'], q=[0,0.33,0.66,1],
                               labels=['Avoid','Medium','Strong'])

    df = df.sort_values(['Prob','Margin'], ascending=False)

    # ---------------- OUTPUT ----------------
    print("\n" + "="*80)
    print("TOP 2027 STRATEGIC HOLDS")
    print("="*80)
    print(df[['Ticker','Name','Prob','Confidence']].head(top_n).to_string(index=False))

    print("\n" + "="*80)
    print("AVOID / UNDERWEIGHT")
    print("="*80)
    print(df[['Ticker','Name','Prob','Confidence']].tail(avoid_n).to_string(index=False))

    # Optional CSV output
    df[['Ticker','Name','Prob','Confidence']].to_csv('macro_2027_ranking_fast_with_names.csv', index=False)
    print("\n✅ Ranking CSV saved → macro_2027_ranking_fast_with_names.csv")


if __name__ == "__main__":
    infer_today()
