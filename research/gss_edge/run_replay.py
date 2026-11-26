import sys, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, live_replay as L
window = int(sys.argv[1]); L.install()
idx = L.CL.index; spy = L.CL["SPY"].values; m200 = pd.Series(spy).rolling(200).mean().values
off = False; flips = []
for i in range(200, len(idx)):
    prev = off
    if off and spy[i] > m200[i] * 1.04: off = False
    elif (not off) and spy[i] < m200[i] * 0.96: off = True
    if off != prev: flips.append(idx[i])
ds = pd.Series(idx, index=idx); me = ds.groupby(ds.index.to_period("M")).last().values
dates = sorted(set(pd.DatetimeIndex(me)) | set(flips)); dates = [d for d in dates if d >= pd.Timestamp("2007-01-01")]
out = {}
for k, d in enumerate(dates):
    try:
        w, s = L.book_at(d, window=window, overlays=False); out[d] = dict(w=w, off=(s.get("risk_gate") == "RISK-OFF"))
    except Exception as e:
        out[d] = dict(w={}, off=False, err=str(e))
    if k % 25 == 0: print(k, len(dates), d.date(), flush=True)
pickle.dump(out, open(f"weights_w{window}.pkl", "wb")); print("done", len(out))
