"""Per-ETF evidence table for the Bonds tab: buy&hold vs the defensive rule (trend200, plus SPY>200DMA & Baa-spread
3m-change z<1 for credit ETFs). Uses the walk-forward feature/posterior files from bond_research.py."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
O="/Users/riddhisiddhi/stocks/research/bonds/"
CREDIT={"LQD","VCSH","HYG","JNK","EMB"}; rng=np.random.default_rng(2); rows=[]
def sh(x,cm):
    x=x.dropna(); tot=x+cm.reindex(x.index); eq=np.cumprod(1+tot); return x.mean()/x.std()*12**.5,(eq/np.maximum.accumulate(eq)-1).min()*100,x.mean()*1200
for t in ["SHY","IEI","IEF","TLH","TLT","TIP","AGG","LQD","VCSH","HYG","JNK","EMB"]:
    d=pd.read_csv(O+f"posterior_{t}.csv",index_col=0,parse_dates=True).dropna(subset=["y"]); d=d.dropna(subset=["spy_up","baa_chg3_z"]); cm=d.cash_m; y=d.y; h=len(d)//2
    s=((d.trend>0)&(d.spy_up>0)&(d.baa_chg3_z<1)).astype(float) if t in CREDIT else (d.trend>0).astype(float)
    r=s*y-s.diff().abs().fillna(0)*0.0005; e=s.mean(); b=sh(y,cm); a=sh(r,cm)
    n95=np.percentile([sh((rng.random(len(d))<e).astype(float)*y,cm)[0] for _ in range(300)],95)
    rows.append(dict(etf=t,rule="trend200 + credit gate" if t in CREDIT else "trend200",start=str(d.index[0].date()),n_months=len(d),exposure=e,
        bh_sharpe=b[0],bh_maxdd=b[1],bh_annex=b[2],rule_sharpe=a[0],rule_maxdd=a[1],rule_annex=a[2],
        rule_h1=sh(r[:h],cm)[0],rule_h2=sh(r[h:],cm)[0],bh_h1=sh(y[:h],cm)[0],bh_h2=sh(y[h:],cm)[0],null95=n95,
        sharpe_2x_cost=sh(s*y-s.diff().abs().fillna(0)*0.0010,cm)[0]))
E=pd.DataFrame(rows); E.to_csv(O+"bond_rule_evidence.csv",index=False); print(E.round(2).to_string(index=False))
