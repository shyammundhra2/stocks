"""Bonds tab engine: live signals for Treasury (by duration), TIPS, investment-grade, high-yield and EM bond ETFs.

WHAT THE SIGNAL IS (and is not) - research/bonds/ (walk-forward, monthly, excess over cash):
  * The BUY/AVOID signal is a DEFENSIVE overlay: trend (price > 200DMA) for Treasuries/TIPS/Agg, and trend AND
    SPY>200DMA AND Baa-spread 3m-change z < 1 for credit (IG/HY/EM). It cut max drawdown 40-60% at roughly the same
    Sharpe as buy&hold. It did NOT beat random timing at the same exposure (null 95th pct) - it is not alpha.
  * The Bayesian view is a conjugate ridge regression (Normal prior on feature effects) giving a posterior
    distribution for next month's excess return. Out of sample its P(excess>0) had AUC 0.506 and avg rank-IC -0.04:
    NO predictive skill. It is shown for honest uncertainty (wide credible intervals) and an evidence breakdown,
    not as a forecast to trade.
Display only - research, not advice.
"""
import os
import json
import time
import threading
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, ".cache", "bonds_live.json")
_EVID = os.path.join(_ROOT, "research", "bonds", "bond_rule_evidence.csv")
_TTL = 6 * 3600
_lock = threading.Lock()

# approx effective durations (years) - labelled approximate on the page
ETFS = {
    "SHY": ("Treasury 1-3y", "Treasury", 1.9), "IEI": ("Treasury 3-7y", "Treasury", 4.5),
    "IEF": ("Treasury 7-10y", "Treasury", 7.5), "TLH": ("Treasury 10-20y", "Treasury", 13.0),
    "TLT": ("Treasury 20y+", "Treasury", 17.0), "TIP": ("TIPS (inflation-linked)", "TIPS", 6.5),
    "AGG": ("US Aggregate", "Aggregate", 6.0), "VCSH": ("IG corporate short", "Investment-grade", 2.7),
    "LQD": ("IG corporate", "Investment-grade", 8.5), "HYG": ("High yield (iShares)", "High yield", 3.5),
    "JNK": ("High yield (SPDR)", "High yield", 3.3), "EMB": ("EM USD sovereign", "Emerging", 7.0),
}
CREDIT = {"LQD", "VCSH", "HYG", "JNK", "EMB"}
FEATS = ["trend", "mom12_1", "mom3", "carry", "baa_z", "baa_chg3_z", "curve_z", "real10_z", "spy_up", "vixz"]
FEAT_LABEL = {"trend": "Trend (above 200DMA)", "mom12_1": "12-1 momentum", "mom3": "3-month momentum",
              "carry": "Carry (yield - cash)", "baa_z": "Credit spread level", "baa_chg3_z": "Credit spread widening (3m)",
              "curve_z": "Curve slope (10y-3m)", "real10_z": "Real 10y yield", "spy_up": "Equities above 200DMA",
              "vixz": "VIX (vs 1y)"}
THR_BUY, THR_AGAINST = 0.60, 0.40
MIN_TRAIN = 60


def _z_exp(s, minp=36):
    return (s - s.expanding(minp).mean()) / s.expanding(minp).std()


def _bayes_ridge(Xtr, ytr, x_new, tau_scale=0.25):
    xm, ym = Xtr.mean(0), ytr.mean()
    sx = Xtr.std(0) + 1e-9
    Xs, yc = (Xtr - xm) / sx, ytr - ym
    tau = tau_scale * ytr.std()
    s2 = ytr.var()
    for _ in range(4):
        S = np.linalg.inv(Xs.T @ Xs / s2 + np.eye(Xs.shape[1]) / tau ** 2)
        b = S @ Xs.T @ yc / s2
        s2 = max(((yc - Xs @ b) ** 2).mean(), 1e-8)
    xs = (x_new - xm) / sx
    return ym + xs @ b, float(np.sqrt(s2 + xs @ S @ xs)), b, xs


def _compute():
    import yfinance as yf
    import pandas_datareader.data as web
    from macro.helpers import compute_RSI  # noqa: F401  (kept for parity with other tabs)

    tk = list(ETFS) + ["SPY", "^VIX", "^IRX"]
    raw = yf.download(tk, start="2002-06-01", progress=False, auto_adjust=True)["Close"].ffill()
    unadj = yf.download(list(ETFS), start="2002-06-01", progress=False, auto_adjust=False)["Close"].ffill()
    asof = raw.index[-1]
    cash = (raw["^IRX"] / 100 / 252).fillna(0)
    cash_ann = (cash * 252).rolling(21).mean()
    spy, vix = raw["SPY"], raw["^VIX"]

    errs = []
    try:
        fr = web.DataReader(["BAA10Y", "DGS3MO", "DGS1", "DGS2", "DGS5", "DGS10", "DGS30", "DFII10", "T10YIE"],
                            "fred", "2002-01-01").ffill()
    except Exception as e:
        fr = None
        errs.append(f"FRED unavailable ({e}); Bayesian view and credit-spread gate degraded")

    yld = {}
    for t in ETFS:
        try:
            d = yf.Ticker(t).dividends
            if len(d):
                d.index = d.index.tz_localize(None)
                yld[t] = (d.rolling("365D").sum().reindex(unadj.index, method="ffill") / unadj[t]).where(unadj[t].notna())
        except Exception:
            pass
    yld = pd.DataFrame(yld)

    ix = raw.index
    me = sorted({ix[ix.searchsorted(m, side="right") - 1] for m in raw.resample("ME").last().index})
    ME = pd.DatetimeIndex(me)
    if ME[-1] != asof:
        ME = ME.append(pd.DatetimeIndex([asof]))
    cumc = (1 + cash).cumprod()
    cashm = cumc.loc[ME].shift(-1) / cumc.loc[ME] - 1

    macro = pd.DataFrame(index=ME)
    macro["spy_up"] = ((spy > spy.rolling(200).mean()) * 2 - 1).reindex(ME).astype(float)
    macro["vixz"] = ((vix - vix.rolling(252).mean()) / vix.rolling(252).std()).reindex(ME)
    if fr is not None:
        macro["baa"] = fr["BAA10Y"].reindex(ME, method="ffill")
        macro["baa_chg3"] = macro.baa.diff(3)
        macro["curve"] = (fr["DGS10"] - fr["DGS3MO"]).reindex(ME, method="ffill")
        macro["real10"] = fr["DFII10"].reindex(ME, method="ffill")
        for c in ("baa", "baa_chg3", "curve", "real10"):
            macro[c + "_z"] = _z_exp(macro[c])

    out_etf = {}
    for t, (nm, cat, dur) in ETFS.items():
        p = raw[t].dropna()
        if len(p) < 300:
            continue
        pm = p.reindex(ME, method="ffill")
        d = pd.DataFrame({"y": pm.shift(-1) / pm - 1 - cashm})
        sma = p.rolling(200).mean()
        d["trend"] = ((p > sma) * 2 - 1).reindex(ME, method="ffill").astype(float)
        d["mom12_1"] = _z_exp((p.shift(21) / p.shift(252) - 1).reindex(ME, method="ffill"))
        d["mom3"] = _z_exp((p / p.shift(63) - 1).reindex(ME, method="ffill"))
        d["carry"] = _z_exp(yld[t].reindex(ME, method="ffill") - cash_ann.reindex(ME, method="ffill")) if t in yld else np.nan
        for c in ("baa_z", "baa_chg3_z", "curve_z", "real10_z", "spy_up", "vixz"):
            d[c] = macro[c] if c in macro else np.nan

        last = float(p.iloc[-1])
        r = {"ticker": t, "name": nm, "category": cat, "duration": dur, "price": round(last, 2),
             "sma200": round(float(sma.iloc[-1]), 2), "above_200": bool(last > sma.iloc[-1]),
             "dist_200_pct": round((last / float(sma.iloc[-1]) - 1) * 100, 1),
             "mom1": round(float(p.iloc[-1] / p.iloc[-22] - 1) * 100, 1), "mom3": round(float(p.iloc[-1] / p.iloc[-64] - 1) * 100, 1),
             "mom12": round(float(p.iloc[-22] / p.iloc[-253] - 1) * 100, 1) if len(p) > 253 else None,
             "dd_pct": round(float(p.iloc[-1] / p.rolling(252, min_periods=60).max().iloc[-1] - 1) * 100, 1),
             "yield_pct": round(float(yld[t].dropna().iloc[-1]) * 100, 2) if t in yld and yld[t].notna().any() else None}
        r["carry_pct"] = round(r["yield_pct"] - float(cash_ann.iloc[-1]) * 100, 2) if r["yield_pct"] is not None else None
        # ---- defensive signal ----
        spy_ok = bool(macro["spy_up"].iloc[-1] > 0)
        baa_ok = True
        if "baa_chg3_z" in macro and np.isfinite(macro["baa_chg3_z"].iloc[-1]):
            baa_ok = bool(macro["baa_chg3_z"].iloc[-1] < 1)
        reasons = [("Price above 200DMA", r["above_200"])]
        on = r["above_200"]
        if t in CREDIT:
            reasons += [("Equities above 200DMA", spy_ok), ("Credit spreads not widening fast", baa_ok)]
            on = on and spy_ok and baa_ok
        r["signal"] = "BUY" if on else "AVOID"
        r["reasons"] = [{"text": a, "ok": bool(b)} for a, b in reasons]
        # ---- Bayesian posterior ----
        r["bayes"] = None
        if fr is not None and d["carry"].notna().any():
            X = d[FEATS]
            train = d.dropna(subset=["y"] + FEATS)
            live = X.iloc[-1]
            if len(train) >= MIN_TRAIN and live.notna().all():
                mu, sd, b, xs = _bayes_ridge(train[FEATS].values, train["y"].values, live.values.astype(float))
                prior = float((train["y"] > 0).mean())
                post = float(norm.cdf(mu / sd))
                contrib = sorted(zip(FEATS, b * xs), key=lambda z: -abs(z[1]))
                r["bayes"] = {"prior_p": round(prior, 3), "post_p": round(post, 3), "mu_pct": round(mu * 100, 2),
                              "lo_pct": round((mu - 1.645 * sd) * 100, 2), "hi_pct": round((mu + 1.645 * sd) * 100, 2),
                              "sd_pct": round(sd * 100, 2), "n_train": int(len(train)),
                              "view": "favors" if post >= THR_BUY else "against" if post <= THR_AGAINST else "neutral",
                              "drivers": [{"feature": FEAT_LABEL[f], "effect_pct": round(c * 100, 2)} for f, c in contrib[:4]]}
        out_etf[t] = r

    # ---- curve & credit analysis ----
    analysis = {"errors": errs}
    if fr is not None:
        def lvl(sid, back=0):
            s = fr[sid].dropna()
            if back == 0:
                return float(s.iloc[-1])
            tgt = s.index[-1] - pd.Timedelta(days=back)
            return float(s[:tgt].iloc[-1]) if len(s[:tgt]) else None
        analysis["curve"] = [{"label": lb, "now": lvl(sid), "m1": lvl(sid, 30), "y1": lvl(sid, 365)}
                             for lb, sid in (("3M", "DGS3MO"), ("1Y", "DGS1"), ("2Y", "DGS2"), ("5Y", "DGS5"), ("10Y", "DGS10"), ("30Y", "DGS30"))]
        analysis["slope_10y3m"] = round(lvl("DGS10") - lvl("DGS3MO"), 2)
        analysis["slope_10y2y"] = round(lvl("DGS10") - lvl("DGS2"), 2)
        analysis["real10"] = round(lvl("DFII10"), 2)
        analysis["breakeven10"] = round(lvl("T10YIE"), 2)
        analysis["baa_spread"] = round(lvl("BAA10Y"), 2)
        analysis["baa_spread_chg3m"] = round(float(macro["baa_chg3"].iloc[-1]), 2) if "baa_chg3" in macro else None
        analysis["baa_spread_z"] = round(float(macro["baa_z"].iloc[-1]), 2) if "baa_z" in macro else None
        analysis["baa_pctile"] = round(float((fr["BAA10Y"].dropna() <= lvl("BAA10Y")).mean() * 100), 0)
    analysis["spy_above_200"] = bool(macro["spy_up"].iloc[-1] > 0)
    analysis["vix"] = round(float(vix.iloc[-1]), 1)
    analysis["vix_z"] = round(float(macro["vixz"].iloc[-1]), 2) if np.isfinite(macro["vixz"].iloc[-1]) else None
    if "HYG" in raw and "IEF" in raw:
        analysis["hy_vs_treasury_3m"] = round(float((raw["HYG"].iloc[-1] / raw["HYG"].iloc[-64]) / (raw["IEF"].iloc[-1] / raw["IEF"].iloc[-64]) - 1) * 100, 2)
    analysis["cash_yield"] = round(float(cash_ann.iloc[-1]) * 100, 2)

    return {"asof": str(asof.date()), "built": time.strftime("%Y-%m-%d %H:%M"), "etfs": list(out_etf.values()), "analysis": analysis}


def _evidence():
    try:
        e = pd.read_csv(_EVID)
        return {r["etf"]: {k: (None if pd.isna(v) else (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v))
                           for k, v in r.items()} for _, r in e.iterrows()}
    except Exception:
        return {}


def _refresh():
    if not _lock.acquire(blocking=False):
        return
    try:
        data = _compute()
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        tmp = _CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, _CACHE)
    except Exception as e:
        print(f"bonds refresh error: {e}")
    finally:
        _lock.release()


def get_bonds(block_if_empty=True):
    """Cached (6h) stale-while-revalidate. First ever call computes synchronously (~30-60s)."""
    data, fresh = None, False
    try:
        with open(_CACHE) as f:
            data = json.load(f)
        fresh = (time.time() - os.path.getmtime(_CACHE)) < _TTL
    except Exception:
        pass
    if data is None:
        if block_if_empty:
            _refresh()
            try:
                with open(_CACHE) as f:
                    data = json.load(f)
            except Exception:
                return {"error": "Bond data unavailable (first build failed - check network/FRED).", "etfs": [], "analysis": {}, "evidence": _evidence()}
        else:
            return {"error": "Computing...", "etfs": [], "analysis": {}, "evidence": _evidence()}
    elif not fresh:
        threading.Thread(target=_refresh, daemon=True).start()
        data["refreshing"] = True
    data["evidence"] = _evidence()
    return data


if __name__ == "__main__":
    d = _compute()
    print(d["asof"], d["analysis"])
    for r in d["etfs"]:
        b = r["bayes"] or {}
        print(f"{r['ticker']:5s} {r['signal']:5s} P={b.get('post_p')} mu={b.get('mu_pct')}% [{b.get('lo_pct')},{b.get('hi_pct')}] yld={r['yield_pct']} mom12={r['mom12']} dd={r['dd_pct']}")
