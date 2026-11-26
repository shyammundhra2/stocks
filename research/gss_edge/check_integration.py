"""(a) get_trends WITH overlays enabled == offline book_overlay on the raw replayed book (same pure code) on sampled dates;
(b) informational size sensitivity of the overlays (NOT a tuned claim)."""
import pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, live_replay as L, replay_sim as S
from macro.indicators import overlays as OV
L.install(); W = pickle.load(open("weights_w1000.pkl", "rb")); ks = sorted(W)
sample = [ks[i] for i in range(10, len(ks), 18)][:14]; bad = 0
for d in sample:
    w, s = L.book_at(d, window=1000, overlays=True); ov = s.get("overlay") or {}
    t = S.idx.get_loc(d); ct = S.cap_targets()[t]
    hv = OV.pick_haven(S.CL["IEF"].iloc[:t + 1], S.CL["SHY"].iloc[:t + 1], S.CL["BIL"].iloc[:t + 1])
    off = W[d]["off"]; exp = OV.book_overlay(W[d]["w"], off, ct, hv)["weights"]; got = s["weights"]
    diff = max([abs(exp.get(k, 0) - got.get(k, 0)) for k in set(exp) | set(got)] or [0])
    bad += diff > 0.006
    print(d.date(), "off" if off else "on ", "haven", ov.get("haven"), "reserve%", ov.get("reserve_spy_pct"), "max|Δw| %.4f" % diff)
print("dates with mismatch >0.6pp:", bad, "of", len(sample))
print("\nINFORMATIONAL size sensitivity (window 1000):")
base = S.evaluate(S.sim(W))
for lab, hf, rf, kw in [("haven 25%", .25, .2, dict(haven=True)), ("reserve 10%", .5, .10, dict(reserve=True)), ("haven 25% + reserve 10%", .25, .10, dict(haven=True, reserve=True))]:
    OV.HAVEN_FRAC, OV.RESERVE_FRAC = hf, rf; ev = S.evaluate(S.sim(W, **kw))
    print(f"{lab:26s}" + " | ".join(f"{p}: Sh {ev[p][0]:.3f} (b {base[p][0]:.3f}) DD {ev[p][2]*100:.1f}% (b {base[p][2]*100:.1f}%)" for p in ev))
