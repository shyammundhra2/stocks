"""Turn replayed LIVE-optimizer books into a return stream with the user's rules (next-day fill, 5bps, cash=BIL),
applying the SAME pure overlay code the live get_trends uses (macro/indicators/overlays.py)."""
import sys, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.indicators import overlays as OV
from macro.indicators.signals import capitulation_state
D = pd.read_pickle("/Users/riddhisiddhi/stocks/research/gss_edge/ohlcv.pkl"); CL = D["Close"]; idx = CL.index
RET = CL.pct_change(fill_method=None).fillna(0.0)
cash = CL["BIL"].pct_change(fill_method=None).fillna(CL["SHY"].pct_change(fill_method=None)).fillna(0.0).values
COST = 5e-4; PERIODS = [("2007-2015", "2007-01-01", "2015-12-31"), ("2016-now", "2016-01-01", "2100-01-01")]
_cap = None
def cap_targets():
    global _cap
    if _cap is None:
        v, s = CL["^VIX"].ffill(), CL["SPY"]; out = np.zeros(len(idx))
        for i in range(300, len(idx)):
            try: out[i] = capitulation_state(v.iloc[max(0, i - 1500):i + 1], s.iloc[max(0, i - 1500):i + 1]).get("target", 0)
            except Exception: out[i] = 0
        _cap = out
    return _cap

def sim(W, haven=False, reserve=False):
    ks = sorted(W); first = idx.searchsorted(ks[0]); ct = cap_targets()
    held = {}; pend = None; cur = None; hv = None; out = np.zeros(len(idx)); kptr = 0
    for t in range(first, len(idx)):
        d = idx[t]
        r = sum(w * RET[s].values[t] for s, w in held.items() if s in RET.columns); r += (1 - sum(held.values())) * cash[t]
        if pend is not None:
            keys = set(pend) | set(held); r -= sum(abs(pend.get(k, 0) - held.get(k, 0)) for k in keys) * COST; held, pend = pend, None
        out[t] = r
        # (loop above advanced via explicit pointer below)
        if kptr < len(ks) and ks[kptr] <= d:
            pass
        # advance decision pointer
        while kptr < len(ks) and ks[kptr] <= d:
            cur = W[ks[kptr]]; kptr += 1
            hv = OV.pick_haven(CL["IEF"].iloc[:t + 1], CL["SHY"].iloc[:t + 1], CL["BIL"].iloc[:t + 1]) if haven else None
        if cur is None: continue
        o = OV.book_overlay(cur["w"], cur["off"], ct[t] if reserve else 0.0, hv if haven else None, reserve_on=reserve)
        pend = o["weights"]
    return pd.Series(out, index=idx)

def stats(r, lo, hi):
    x = r.loc[lo:hi]; c = pd.Series(cash, index=idx).loc[lo:hi]; ex = x - c
    eq = (1 + x).cumprod(); return ex.mean() / ex.std() * np.sqrt(252), float(eq.iloc[-1] ** (252 / len(x)) - 1), float((eq / eq.cummax() - 1).min())
def evaluate(r): return {p: stats(r, lo, hi) for p, lo, hi in PERIODS}
