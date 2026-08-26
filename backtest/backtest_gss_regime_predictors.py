"""
What can PREDICT the regime (choppy/calm/crisis) with LEADING edge - not just
label the present? The regime is trend + volatility. Forward TREND/return is
unpredictable (backtest_gss_hmm_eval). Forward VOLATILITY is partly predictable
(vol clusters), so the real question is: what leads a rise INTO a higher-vol
regime before VIX/term-structure (which fire coincidentally, at the bottom)?

Test candidate LEADING features at time t against:
  fwd_vol    forward 20d realized vol (the regime's vol axis)
  d_vol      fwd_vol - current_vol  (the TRANSITION - leading edge lives here,
             beyond trivial persistence)
  fwd_dd     forward 20d max drawdown (how bad the path gets)

Spearman corr reported. A feature with LEADING edge shows meaningful positive
corr with d_vol (predicts vol RISING) - that's what could give the HMM a
forward tilt. Coincident features corr with fwd_vol but ~0 with d_vol.

Candidates (all from data the app already fetches):
  cur_vol    current 20d realized vol (persistence baseline)
  ts         VIX/VIX3M term structure (known coincident - the control)
  credit_mom HYG/IEF 20d return (credit weakening -> stress ahead?)
  curve      10y-3m (^TNX-^IRX) level (inversion leads stress, long lag)
  move_z     ^MOVE z-score (bond vol - leads equity vol?)
  breadth    RSP/SPY 20d return (equal-wt lagging cap-wt = narrowing/top warn)
"""
import sys
import ssl
import io
import time
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr


def cboe(sym):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=ctx, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw)); df.columns = [c.strip().upper() for c in df.columns]
    dc = next(c for c in df.columns if "DATE" in c); cc = next(c for c in df.columns if "CLOSE" in c)
    df[dc] = pd.to_datetime(df[dc]); return df.set_index(dc)[cc].sort_index()


def col(raw, t):
    c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    s = c[t] if t in c.columns else c
    return pd.Series(np.asarray(s).ravel(), index=c.index)


def main():
    t0 = time.time()
    tk = ["SPY", "RSP", "HYG", "IEF", "^TNX", "^IRX", "^MOVE", "^VIX"]
    print("Downloading ...")
    raw = yf.download(tk, start="2009-06-01", end="2026-08-21", progress=False, auto_adjust=True)
    spy = col(raw, "SPY"); rsp = col(raw, "RSP"); hyg = col(raw, "HYG"); ief = col(raw, "IEF")
    tnx = col(raw, "^TNX"); irx = col(raw, "^IRX"); move = col(raw, "^MOVE"); vix = col(raw, "^VIX")
    ts = (cboe("VIX") / cboe("VIX3M")).reindex(spy.index).ffill()

    dr = np.log(spy / spy.shift(1))
    cur_vol = dr.rolling(20).std() * np.sqrt(252)
    # forward 20d realized vol: rolling(20) at i+20 covers returns i+1..i+20
    fwd_vol = (dr.rolling(20).std() * np.sqrt(252)).shift(-20)
    # forward 20d max drawdown
    fwd_dd = pd.Series(index=spy.index, dtype=float)
    sv = spy.values
    for i in range(len(sv) - 1):
        w = sv[i + 1:i + 21]
        if len(w) > 1:
            fwd_dd.iloc[i] = float((w / np.maximum.accumulate(w) - 1).min())
    d_vol = fwd_vol - cur_vol

    feats = {
        "cur_vol": cur_vol,
        "ts": ts,
        "credit_mom": -(hyg / ief).pct_change(20),        # sign: credit WEAKENING = +
        "curve_inv": -(tnx - irx),                        # sign: inversion = +
        "move_z": (move - move.rolling(126).mean()) / move.rolling(126).std(),
        "breadth": -(rsp / spy).pct_change(20),           # sign: equal-wt lagging = +
    }
    df = pd.DataFrame({**feats, "fwd_vol": fwd_vol, "d_vol": d_vol, "fwd_dd": fwd_dd}).dropna()
    print(f"obs {len(df)}  {df.index[0].date()} -> {df.index[-1].date()}")
    print("(features signed so + = more stress expected)\n")
    print(f"{'feature':11s} {'corr fwd_vol':>12s} {'corr d_vol':>11s} {'corr fwd_dd':>12s}   read")
    print("-" * 70)
    for k in feats:
        c_fv = spearmanr(df[k], df["fwd_vol"]).correlation
        c_dv = spearmanr(df[k], df["d_vol"]).correlation
        c_dd = spearmanr(df[k], df["fwd_dd"]).correlation      # fwd_dd negative; +corr = predicts deeper dd
        tag = "LEADS vol-up" if c_dv > 0.08 else ("coincident" if c_fv > 0.2 else "weak")
        print(f"{k:11s} {c_fv:>+12.3f} {c_dv:>+11.3f} {c_dd:>+12.3f}   {tag}")
    print("\n(d_vol = forward vol MINUS current vol: leading edge for the")
    print(" calm->crisis TRANSITION lives here. fwd_dd is negative, so a")
    print(" NEGATIVE corr with fwd_dd means the feature predicts DEEPER drawdowns.)")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
