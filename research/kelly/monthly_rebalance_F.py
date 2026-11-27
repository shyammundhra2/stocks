"""Variant F with EVERY sleeve rebalanced only at month-end (position decided at month-end close, held next month)."""
import sys, numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0,"/Users/riddhisiddhi/stocks")
from macro.helpers import compute_RSI
R="/Users/riddhisiddhi/stocks/research/kelly/"
d=pd.read_csv(R+"sleeve_monthly_excess_7.csv",index_col=0)
ETF=pd.read_csv("/Users/riddhisiddhi/stocks/trendselection/trend_etf_ranking.csv")["Ticker"].tolist()
raw=yf.download(ETF+["SPY","QQQ","^VIX","^IRX"],start="2004-01-01",end="2026-11-26",progress=False,auto_adjust=True)["Close"].ffill()
irx=raw["^IRX"]; spy=raw.SPY; vix=raw["^VIX"]; cash_d=(irx/100/252).fillna(0)
me=raw.resample("ME").last().index; ix=raw.index; ME=[ix[ix.searchsorted(m,side="right")-1] for m in me]
ME=pd.DatetimeIndex(ME).unique()
def mret(s):  # month-end to next month-end return series indexed by the START month-end
    return (s.loc[ME].shift(-1)/s.loc[ME]-1)
cm=((1+cash_d).cumprod().loc[ME].shift(-1)/(1+cash_d).cumprod().loc[ME]-1)
px=raw[[t for t in ETF if t in raw.columns]]; px=px.loc[:,px.notna().sum()>800]
# trend ETF: equal-weight of those > 200SMA at month-end
on=(px>px.rolling(200,min_periods=200).mean()).astype(float).where(px.notna(),0.0); n=px.notna().sum(axis=1).clip(lower=1)
wt=on.div(n,axis=0).loc[ME]; ret=px.loc[ME].shift(-1)/px.loc[ME]-1
tr=(wt*ret).sum(axis=1)+(1-wt.sum(axis=1))*cm-(wt.diff().abs().sum(axis=1).fillna(0))*5/1e4
ew_on=(px.EWY>px.EWY.rolling(200,min_periods=200).mean()).astype(float).loc[ME]
ewy=ew_on*ret.EWY+(1-ew_on)*cm-ew_on.diff().abs().fillna(0)*5/1e4
# capitulation ladder at month-end
z=(vix-vix.rolling(50).mean())/vix.rolling(50).std(); r14=compute_RSI(spy,14); dde=spy/spy.rolling(252,min_periods=60).max().shift(1)-1
df=pd.concat({"z":z,"r":r14,"d":dde},axis=1).dropna(); dep,hw,tg=False,0.0,[]
for zz,rr,dd in df.itertuples(index=False):
    if dep and dd>0: dep,hw=False,0.0
    else:
        if not dep and zz>2 and rr<30: dep,hw=True,zz
        elif dep: hw=max(hw,zz)
    tg.append((1000+min(max((hw-2)/2,0),1)*99000)/100000 if dep else 0.0)
f=pd.Series(tg,index=df.index).reindex(ix).ffill().fillna(0).loc[ME]
cap=f*mret(spy)+(1-f)*cm-f.diff().abs().fillna(0)*5/1e4
# QQQ RSI2 state at month-end
q=raw.QQQ; rsi=compute_RSI(q,2); sma=q.rolling(200,min_periods=200).mean(); pos,cur=[],0
for p,s,r in zip(q,sma,rsi):
    if np.isnan(s) or p<s: cur=0
    elif r<=5: cur=1
    elif r>=70: cur=0
    pos.append(cur)
qp=pd.Series(pos,index=q.index).loc[ME]
qq=qp*mret(q)+(1-qp)*cm-qp.diff().abs().fillna(0)*5/1e4
m=pd.DataFrame({"trend_etf":tr-cm,"country_ewy":ewy-cm,"vix_capitulation":cap-cm,"qqq_rsi2":qq-cm}).dropna()
m.index=pd.DatetimeIndex(m.index).to_period("M")+1   # return realised in the following month, align to d's month index
m.index=m.index.astype(str)
D=d.copy(); 
for c in m.columns: D[c+"_m"]=m[c]
D=D.dropna(subset=[c+"_m" for c in m.columns])
W={"stock_sieve":.2,"etf_12_1":.1,"trend_etf":.2,"vix_capitulation":.2,"qqq_rsi2":.05,"country_ewy":.02}
def st(ex,lbl):
    tot=ex+D.cash_m; eq=np.cumprod(1+tot); dd=(eq/np.maximum.accumulate(eq)-1).min()
    print(f"{lbl:34s} annEx {ex.mean()*1200:5.1f}% vol {ex.std()*12**.5*100:4.1f}% Sharpe {ex.mean()/ex.std()*12**.5:4.2f} maxDD {dd*100:6.1f}%")
daily=sum(D[k]*w for k,w in W.items()); st(daily,"F, daily-signal sleeves (as before)")
mon=sum((D[k+"_m"] if k+"_m" in D else D[k])*w for k,w in W.items()); st(mon,"F, ALL monthly rebalance")
print("\nper-sleeve Sharpe daily-signal -> monthly:")
for k in m.columns: print(f"  {k:18s} {D[k].mean()/D[k].std()*12**.5:5.2f} -> {D[k+'_m'].mean()/D[k+'_m'].std()*12**.5:5.2f}   annEx {D[k].mean()*1200:5.1f}% -> {D[k+'_m'].mean()*1200:5.1f}%")
print("QQQ RSI2 invested at month-ends: %.1f%% ; capitulation deployed at month-ends: %.1f%%"%(qp.mean()*100,(f>0).mean()*100))
mix=sum(D[k]*W[k] for k in ("vix_capitulation","qqq_rsi2"))+sum((D[k+"_m"] if k+"_m" in D else D[k])*W[k] for k in ("stock_sieve","etf_12_1","trend_etf","country_ewy"))
st(mix,"F, QQQ+VIX daily, rest monthly")
dd2=D.copy(); dd2["stock_sieve"]*=.4
mix2=sum(dd2[k]*W[k] for k in ("vix_capitulation","qqq_rsi2"))+sum((dd2[k+"_m"] if k+"_m" in dd2 else dd2[k])*W[k] for k in ("stock_sieve","etf_12_1","trend_etf","country_ewy")); st(mix2,"  same, sieve returns x0.4")
h=len(mix)//2; print("  halves Sharpe: %.2f / %.2f"%(mix[:h].mean()/mix[:h].std()*12**.5,mix[h:].mean()/mix[h:].std()*12**.5))

# ================= BASELINES (same vol as F's mixed-cadence mix) =================
def aligned(s):  # series indexed by month-end START -> index like D (return month)
    s=s.copy(); s.index=(pd.DatetimeIndex(s.index).to_period("M")+1).astype(str); return s.reindex(D.index)
spy_ret=aligned(mret(spy)); cmm=aligned(cm)
spy_ex=spy_ret-cmm
sma=spy.rolling(200,min_periods=200).mean(); g=(spy>sma).astype(float).loc[ME]
gated=aligned(g*mret(spy)+(1-g)*cm-g.diff().abs().fillna(0)*5/1e4)-cmm
def sh(x): return x.mean()/x.std()*12**.5
def scaled(x,vol): return x*(vol/x.std())
def stats2(x):
    tot=x+D.cash_m; eq=np.cumprod(1+tot); dd=(eq/np.maximum.accumulate(eq)-1).min()
    yr=tot.groupby(tot.index.str[:4]).apply(lambda v:(1+v).prod()-1); h=len(x)//2
    return x.mean()*1200,x.std()*12**.5*100,sh(x),dd*100,yr.min()*100,yr.idxmin(),sh(x[:h]),sh(x[h:])
vol=mix.std(); rows={"F (QQQ+VIX daily, rest monthly)":mix,"SPY buy&hold, unscaled":spy_ex,"SPY buy&hold, scaled to F vol":scaled(spy_ex,vol),
 "SPY 200DMA-gated, unscaled":gated,"SPY 200DMA-gated, scaled to F vol":scaled(gated,vol),
 "Trend-ETF sleeve alone, scaled to F vol":scaled(D["trend_etf_m"],vol),
 "F with sieve x0.4 (survivorship haircut)":mix2}
print(f"\n{'':44s} annEx   vol Sharpe  maxDD  worst yr        halves")
for k,x in rows.items():
    a=stats2(x); print(f"{k:44s} {a[0]:5.1f}% {a[1]:4.1f}% {a[2]:5.2f} {a[3]:6.1f}% {a[4]:6.1f}% ({a[5]})  {a[6]:.2f}/{a[7]:.2f}")
print("\nlev needed for gated SPY to match F vol: %.2fx ; SPY b&h: %.2fx"%(vol/gated.std(),vol/spy_ex.std()))
# paired difference F - scaled gated SPY: is it distinguishable from noise?
diff=mix-scaled(gated,vol); print("F minus scaled gated-SPY: mean %.2f%%/yr, t-stat (monthly) %.2f"%(diff.mean()*1200,diff.mean()/diff.std()*np.sqrt(len(diff))))
