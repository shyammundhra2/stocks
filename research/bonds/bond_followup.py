import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
O="/Users/riddhisiddhi/stocks/research/bonds/"
ETFs=["SHY","IEI","IEF","TLH","TLT","TIP","LQD","VCSH","HYG","JNK","EMB","AGG"]
D={t:pd.read_csv(O+f"posterior_{t}.csv",index_col=0,parse_dates=True) for t in ETFs}
# (a) calibration / AUC of the Bayesian probability, pooled and per ETF
from itertools import chain
allp=pd.concat([d.dropna(subset=["P"]).assign(etf=t) for t,d in D.items()]); allp["up"]=(allp.y>0).astype(int)
def auc(p,u):
    r=pd.Series(p).rank().values; n1=u.sum(); n0=len(u)-n1; return (r[u==1].sum()-n1*(n1+1)/2)/(n1*n0)
print("AUC of P(excess>0) pooled: %.3f  (0.5 = no skill); base rate up=%.2f ; mean P=%.2f"%(auc(allp.P.values,allp.up.values),allp.up.mean(),allp.P.mean()))
print("per ETF AUC:",{t:round(auc(g.P.values,g.up.values),2) for t,g in allp.groupby("etf")})
allp["bin"]=pd.cut(allp.P,[0,.3,.4,.5,.6,.7,1]); print(allp.groupby("bin").agg(n=("up","size"),hit=("up","mean"),mean_y=("y",lambda s:s.mean()*100)).round(3).to_string())
# rank-IC of posterior mean vs realised return
ic=allp.groupby("etf").apply(lambda g:g.MU.corr(g.y,method="spearman")); print("rank IC (posterior mean vs realised) per ETF:",ic.round(2).to_dict()," avg %.3f"%ic.mean())
# (b) credit-stress gate for credit ETFs: hold only if SPY>200DMA (spy_up>0) and Baa-spread 3m change z < 1
print(f"\n{'etf':5s} rule                                   exp  Sharpe maxDD  h1/h2    null95")
rng=np.random.default_rng(1)
def sh(x,cm): 
    x=x.dropna(); tot=x+cm.reindex(x.index); eq=np.cumprod(1+tot); return x.mean()/x.std()*12**.5,(eq/np.maximum.accumulate(eq)-1).min()*100
for t in ("LQD","HYG","JNK","EMB","VCSH"):
    d=D[t].copy(); cm=d.cash_m; y=d.y; h=len(d)//2
    rules={"buy&hold":pd.Series(1.0,index=d.index),"credit gate (SPY>200 & dBaa z<1)":((d.spy_up>0)&(d.baa_chg3_z<1)).astype(float),
           "SPY>200DMA only":(d.spy_up>0).astype(float),"trend200 & credit gate":((d.trend>0)&(d.spy_up>0)&(d.baa_chg3_z<1)).astype(float)}
    for nm,s in rules.items():
        r=s*y-s.diff().abs().fillna(0)*0.0005; e=s.mean(); a=sh(r,cm)
        n95=np.percentile([sh((rng.random(len(d))<e).astype(float)*y,cm)[0] for _ in range(300)],95) if nm!="buy&hold" else np.nan
        print(f"{t:5s} {nm:38s} {e:4.2f} {a[0]:5.2f} {a[1]:6.1f}%  {sh(r[:h],cm)[0]:.2f}/{sh(r[h:],cm)[0]:.2f}  {n95:5.2f}")
