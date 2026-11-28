"""Bond ETF buy-signal research (walk-forward, monthly). Pre-declared design, nothing tuned on the test:
 target: next-month excess-over-cash return of each bond ETF (total return, auto_adjust), signals known at month-end.
 features (all causal, expanding-window z-scored): trend200 (+1/-1), 12-1 & 3m momentum, carry (trailing-12m
   distribution yield - cash), Baa-Treasury spread level & 3m change, curve slope (10y-3m), real 10y yield, SPY>200DMA, VIX z.
 Bayesian model: conjugate ridge regression (Normal prior on betas, prior sd = 0.25*sd(y_train)/feature-sd, noise var
   from training residuals) -> posterior predictive mean/sd -> P(excess>0). BUY if P>=0.60 (declared up front).
 Comparators: buy&hold, 200DMA trend, exposure-matched static (same average exposure), random-exposure null (same
   exposure fraction, 300 draws), both sub-periods, 2x cost."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import norm
import pandas_datareader.data as web
OUT="/Users/riddhisiddhi/stocks/research/bonds/"
ETF={"SHY":"UST 1-3y","IEI":"UST 3-7y","IEF":"UST 7-10y","TLH":"UST 10-20y","TLT":"UST 20y+","TIP":"TIPS","LQD":"IG corp","VCSH":"IG corp short","HYG":"High yield","JNK":"High yield (SPDR)","EMB":"EM USD debt","AGG":"Agg bond"}
tk=list(ETF)+["SPY","^VIX","^IRX"]
raw=yf.download(tk,start="2002-06-01",end="2026-11-28",progress=False,auto_adjust=True)["Close"].ffill()
adj=yf.download(list(ETF),start="2002-06-01",end="2026-11-28",progress=False,auto_adjust=False)["Close"].ffill()
yld={}
for t in ETF:
    d=yf.Ticker(t).dividends
    if len(d)==0: continue
    d.index=d.index.tz_localize(None); ttm=d.rolling("365D").sum(); yld[t]=(ttm.reindex(adj.index,method="ffill")/adj[t]).where(adj[t].notna())
yld=pd.DataFrame(yld)
f=web.DataReader(["BAA10Y","DGS10","DGS3MO","DFII10"],"fred","2002-01-01").ffill()
cash=(raw["^IRX"]/100/252).fillna(0)
spy=raw["SPY"]; vix=raw["^VIX"]
me=raw.resample("ME").last().index; ix=raw.index; ME=pd.DatetimeIndex(sorted({ix[ix.searchsorted(m,side="right")-1] for m in me}))
cashm=((1+cash).cumprod().loc[ME].shift(-1)/(1+cash).cumprod().loc[ME]-1)
def z_exp(s,minp=36): return (s-s.expanding(minp).mean())/s.expanding(minp).std()
rows=[]
macro=pd.DataFrame(index=ME)
macro["baa"]=f["BAA10Y"].reindex(ME,method="ffill"); macro["baa_chg3"]=macro.baa.diff(3)
macro["curve"]=(f["DGS10"]-f["DGS3MO"]).reindex(ME,method="ffill"); macro["real10"]=f["DFII10"].reindex(ME,method="ffill")
macro["spy_up"]=((spy>spy.rolling(200).mean())*2-1).reindex(ME).astype(float)
macro["vixz"]=(((vix-vix.rolling(252).mean())/vix.rolling(252).std())).reindex(ME)
for c in ("baa","baa_chg3","curve","real10"): macro[c+"_z"]=z_exp(macro[c])
data={}
for t in ETF:
    p=raw[t].dropna(); 
    if len(p)<900: continue
    pm=p.reindex(ME,method="ffill"); ret=pm.shift(-1)/pm-1
    d=pd.DataFrame({"y":ret-cashm})
    d["trend"]=((p>p.rolling(200).mean())*2-1).reindex(ME,method="ffill").astype(float)
    d["mom12_1"]=z_exp((p.shift(21)/p.shift(252)-1).reindex(ME,method="ffill"))
    d["mom3"]=z_exp((p/p.shift(63)-1).reindex(ME,method="ffill"))
    cm_ann=(cash*252).rolling(21).mean().reindex(ME,method="ffill")
    d["carry"]=z_exp((yld[t].reindex(ME,method="ffill")-cm_ann)) if t in yld else np.nan
    for c in ("baa_z","baa_chg3_z","curve_z","real10_z","spy_up","vixz"): d[c]=macro[c]
    d["cash_m"]=cashm; data[t]=d.dropna(subset=["trend","mom12_1","mom3","carry","baa_z","baa_chg3_z","curve_z","real10_z","vixz"]).dropna(subset=["y"])
FEATS=["trend","mom12_1","mom3","carry","baa_z","baa_chg3_z","curve_z","real10_z","spy_up","vixz"]
def bayes_ridge(Xtr,ytr,x_new,tau_scale=0.25):
    xm,ym=Xtr.mean(0),ytr.mean(); sx=Xtr.std(0)+1e-9; Xs=(Xtr-xm)/sx; yc=ytr-ym
    tau=tau_scale*ytr.std()                      # prior sd of each standardized-feature beta
    s2=ytr.var()
    A=Xs.T@Xs/s2+np.eye(Xs.shape[1])/tau**2; S=np.linalg.inv(A); b=S@Xs.T@yc/s2
    for _ in range(3):                           # refine noise var from residuals (empirical-Bayes-lite)
        s2=max(((yc-Xs@b)**2).mean(),1e-8); A=Xs.T@Xs/s2+np.eye(Xs.shape[1])/tau**2; S=np.linalg.inv(A); b=S@Xs.T@yc/s2
    xs=(x_new-xm)/sx; mu=ym+xs@b; var=s2+xs@S@xs
    return mu,np.sqrt(var),b
MINTR=60; THR=0.60
res={}
for t,d in data.items():
    n=len(d); P=np.full(n,np.nan); MU=np.full(n,np.nan); SD=np.full(n,np.nan)
    X=d[FEATS].values; y=d["y"].values
    for i in range(MINTR,n):
        mu,sd,_=bayes_ridge(X[:i],y[:i],X[i]); MU[i],SD[i],P[i]=mu,sd,norm.cdf(mu/sd)
    d["P"],d["MU"],d["SD"]=P,MU,SD; res[t]=d
def perf(x,cm):
    x=pd.Series(x).dropna()
    if len(x)<24 or x.std()==0: return (np.nan,)*3
    tot=x+cm.reindex(x.index); eq=np.cumprod(1+tot); return x.mean()/x.std()*12**.5,(eq/np.maximum.accumulate(eq)-1).min()*100,x.mean()*1200
rng=np.random.default_rng(0); rows=[]
for t,d in res.items():
    d=d.dropna(subset=["P"]); cm=d["cash_m"]; y=d["y"]; h=len(d)//2
    sigs={"buy&hold":pd.Series(1.0,index=d.index),"trend200":(d.trend>0).astype(float),"bayes P>=.60":(d.P>=THR).astype(float),"bayes & trend":((d.P>=THR)&(d.trend>0)).astype(float)}
    for nm,s in sigs.items():
        r=s*y-s.diff().abs().fillna(0)*0.0005*1  # 5bps one-way on turnover (ETF spreads tighter; 2x below)
        r2=s*y-s.diff().abs().fillna(0)*0.0010
        e=s.mean(); sh,dd,ann=perf(r,cm); sh1=perf(r[:h],cm)[0]; sh2=perf(r[h:],cm)[0]; sh2x=perf(r2,cm)[0]
        # exposure-matched static and random-exposure null
        shm=perf(e*y,cm)[0]; nulls=[]
        if nm not in("buy&hold",):
            for _ in range(300):
                rs=(rng.random(len(d))<e).astype(float); nulls.append(perf(rs*y,cm)[0])
        rows.append(dict(etf=t,rule=nm,exp=e,sharpe=sh,maxdd=dd,annex=ann,sh_h1=sh1,sh_h2=sh2,sh_2xcost=sh2x,sh_matched_static=shm,
                         null_mean=np.mean(nulls) if nulls else np.nan,null95=np.percentile(nulls,95) if nulls else np.nan,n=len(d),start=str(d.index[0].date())))
R=pd.DataFrame(rows); R.to_csv(OUT+"bond_walkforward_results.csv",index=False)
for t,d in res.items(): d.to_csv(OUT+f"posterior_{t}.csv")
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(R.round(2).to_string(index=False))
