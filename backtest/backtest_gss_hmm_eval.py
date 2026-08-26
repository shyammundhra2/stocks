"""
Does the regime HMM get BETTER with the VIX term structure, and does it have ANY
predictive edge (vs just labeling the present)? Fit 3-feature and 4-feature HMMs
2010-26, decode states, and test each against FORWARD SPY returns. Crucially,
compare to the RAW term-structure signal (backwardation, computed causally, no
model) - if the HMM doesn't beat the raw ratio, the predictive edge lives in the
leading feature, not the HMM machinery.

Metrics per model:
  switch%   daily state-switch rate (lower = more persistent regimes)
  fwd20 by state   mean SPY next-20d return | crisis / choppy / calm
                   (a real detector: crisis precedes LOW returns, calm HIGH)
  spread    calm fwd20 - crisis fwd20 (bigger = better forward separation)

Honest note: state params are fit in-sample (optimistic absolute levels); the
RELATIVE comparison (3 vs 4 feat, HMM vs raw causal signal) is the real signal.
"""
import sys
import ssl
import io
import time
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from hmmlearn import hmm
except Exception:
    hmm = None


def cboe(sym):
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, context=ctx, timeout=30).read()
    df = pd.read_csv(io.BytesIO(raw)); df.columns = [c.strip().upper() for c in df.columns]
    dc = next(c for c in df.columns if "DATE" in c); cc = next(c for c in df.columns if "CLOSE" in c)
    df[dc] = pd.to_datetime(df[dc]); return df.set_index(dc)[cc].sort_index()


def fit_label(X, ncols):
    tp = np.ones((3, 3)); np.fill_diagonal(tp, 10.0)
    m = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100,
                        random_state=42, transmat_prior=tp)
    m.fit(X)
    means = m.means_
    score = {i: means[i, 0] - 0.5 * means[i, 1] - 0.5 * means[i, 2] -
             (0.5 * means[i, 3] if ncols > 3 else 0.0) for i in range(3)}
    order = sorted(score, key=score.get)
    lab = {order[0]: "crisis", order[1]: "choppy", order[2]: "calm"}
    states = m.predict(X)
    return np.array([lab[s] for s in states]), (m.transmat_, means, lab)


def main():
    if hmm is None:
        print("hmmlearn not available"); return
    t0 = time.time()
    spy = yf.download("SPY", start="2009-06-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    vix = yf.download("^VIX", start="2009-06-01", end="2026-08-21", progress=False, auto_adjust=True)["Close"]
    spy = pd.Series(np.asarray(spy).ravel(), index=spy.index)
    vix = pd.Series(np.asarray(vix).ravel(), index=vix.index)
    vix3m = cboe("VIX3M"); vixc = cboe("VIX")
    ts = (vixc / vix3m).reindex(spy.index).ffill()

    dr = np.log(spy / spy.shift(1))
    ret = np.log(spy / spy.shift(21))
    vol = dr.rolling(20).std() * np.sqrt(252)
    vol_z = (vol - vol.rolling(126).mean()) / vol.rolling(126).std()
    vixr = vix.reindex(spy.index).ffill()
    vix_z = (vixr - vixr.rolling(50).mean()) / vixr.rolling(50).std()
    ts_z = (ts - ts.rolling(50).mean()) / ts.rolling(50).std()
    fwd20 = spy.shift(-20) / spy - 1.0

    base = pd.DataFrame({"ret": ret, "vol_z": vol_z, "vix_z": vix_z, "ts_z": ts_z,
                         "ts": ts, "fwd20": fwd20}).dropna()
    print(f"obs {len(base)}  {base.index[0].date()} -> {base.index[-1].date()}\n")

    for name, cols in [("3-feature (current)", ["ret", "vol_z", "vix_z"]),
                       ("4-feature (+ts)", ["ret", "vol_z", "vix_z", "ts_z"])]:
        X = base[cols].values
        states, _ = fit_label(X, len(cols))
        sw = float(np.mean(states[1:] != states[:-1]))
        f = base["fwd20"].values
        by = {s: f[states == s].mean() for s in ("crisis", "choppy", "calm")}
        share = {s: float(np.mean(states == s)) for s in ("crisis", "choppy", "calm")}
        spread = by["calm"] - by["crisis"]
        print(f"{name}")
        print(f"  switch% {sw:5.1%}   fwd20:  crisis {by['crisis']:+.2%} ({share['crisis']:.0%})   "
              f"choppy {by['choppy']:+.2%} ({share['choppy']:.0%})   calm {by['calm']:+.2%} ({share['calm']:.0%})")
        print(f"  calm-crisis spread {spread:+.2%}\n")

    # Raw causal baseline: no model, just the term-structure ratio
    bw = base["ts"] > 1.0
    print("RAW term structure (no model, causal):")
    print(f"  backwardation(ts>1) share {bw.mean():.0%}   fwd20:  ts>1 {base['fwd20'][bw].mean():+.2%}   "
          f"ts<=1 {base['fwd20'][~bw].mean():+.2%}   spread {base['fwd20'][~bw].mean()-base['fwd20'][bw].mean():+.2%}")
    # continuous: correlation of ts with forward return
    c = np.corrcoef(base["ts"].values, base["fwd20"].values)[0, 1]
    print(f"  corr(ts, fwd20) = {c:+.3f}   (negative = backwardation predicts weak forward returns)")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
