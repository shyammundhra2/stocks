"""Adds commodity (DBC) and country (EWY) sleeves: long when close > 200SMA (lagged 1d), else cash; 5 bps on turnover.
NOTE: DBC/EWY were chosen because they top today's ranking -> hindsight selection bias; treat as optimistic."""
import numpy as np, pandas as pd, yfinance as yf
R="/Users/riddhisiddhi/stocks/research/kelly/sleeve_monthly_excess.csv"
d=pd.read_csv(R,index_col=0)
raw=yf.download(["DBC","EWY","^IRX"],start="2004-01-01",end="2026-11-26",progress=False,auto_adjust=True)["Close"].ffill()
cash=(raw["^IRX"]/100/252).fillna(0)
def sl(t):
    c=raw[t].dropna(); f=(c>c.rolling(200,min_periods=200).mean()).astype(float).shift(1).fillna(0)
    r=f*c.pct_change()+(1-f)*cash.reindex(c.index)-f.diff().abs().fillna(0)*5/1e4
    m=(1+r).groupby(pd.Grouper(freq="ME")).prod()-1; cm=(1+cash).groupby(pd.Grouper(freq="ME")).prod()-1
    s=(m-cm.reindex(m.index)); s.index=s.index.to_period("M"); return s
d["commodity_dbc"]=sl("DBC"); d["country_ewy"]=sl("EWY")
d=d.dropna(subset=["commodity_dbc","country_ewy"]); d.to_csv(R.replace(".csv","_7.csv")); print(d.index[0],d.index[-1],len(d))
