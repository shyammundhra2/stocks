"""Replay the LIVE get_trends() (real router, sizing, covariance optimizer, gate, throttle) at historical dates.
Only data feeds are substituted: shared market data -> trailing window of ohlcv.pkl ending at the as-of date,
VIX term structure -> ^VIX/^VIX3M at that date, ML risk regime -> stub (ml_slow no longer moves capital per
mathstats._effective_vol_cap). Outputs per-date raw book weights, gate state -> weights_<tag>.pkl"""
import sys, io, time, contextlib, warnings, pickle; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
import macro.indicators.portfolio as P
from macro.indicators import overlays as OV

DATA = pd.read_pickle("/Users/riddhisiddhi/stocks/research/gss_edge/ohlcv.pkl")
CL = DATA["Close"]; STATE = {}
ts_full = (CL["^VIX"] / CL["^VIX3M"]).dropna()

def install():
    P._get_shared_market_data = lambda: STATE["shared"]
    P.get_risk_regime = lambda: {"details": [], "ml_slow": 50}
    P.get_vix_term_structure = lambda: STATE["ts"]
    P.vix_ts_deploy_throttle = lambda floor=0.30: float(np.clip(1.0 - 2.5 * (float(STATE["ts"].iloc[-1]) - 0.90), floor, 1.0)) if len(STATE["ts"]) else 1.0

def book_at(d, window=1000, overlays=False):
    sub = DATA.loc[:d].tail(window)
    STATE["shared"] = sub; STATE["ts"] = ts_full.loc[:d]
    OV.OVERLAYS_ENABLED = overlays
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = P.get_trends.__wrapped__()
    summ = P.get_portfolio_summary()
    w = {r["sym"]: r["weight_pct"] / 100.0 for r in res if r.get("weight_pct", 0) > 0}
    return w, summ

if __name__ == "__main__":
    install()
    for d in ["2008-09-30", "2020-03-31", "2024-06-28"]:
        t = time.time(); w, s = book_at(pd.Timestamp(d))
        print(d, "gate", s.get("risk_gate"), "n", len(w), "sum %.2f" % sum(w.values()), "%.1fs" % (time.time() - t), {k: round(v, 3) for k, v in list(w.items())[:5]})
