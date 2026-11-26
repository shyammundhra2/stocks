"""Replay the live get_trends() with an alternative MOM eligibility (T6 idea). usage: run_replay_t6.py <mode> <window>"""
import sys, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, live_replay as L
import macro.indicators.portfolio as P
mode, window = sys.argv[1], int(sys.argv[2]); L.install(); P.MOM_ELIGIBILITY = mode
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
    if k % 25 == 0: print(mode, k, len(dates), d.date(), flush=True)
pickle.dump(out, open(f"weights_{mode}_w{window}.pkl", "wb")); print("done", len(out), "errors", sum('err' in v for v in out.values()))
