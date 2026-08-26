"""
Metro housing ranker - REWRITTEN 2026-08-26 to be honest and VALIDATED.

The old version trained a separate RF/GBM per metro on ~60-120 overlapping-target
observations with NO out-of-sample check (TimeSeriesSplit was imported but never
used) and forecast 5 years out - textbook overfit. This version:

  1. POOLS all metros into one cross-sectional panel (metro x month) - thousands
     of rows instead of ~100, so the model actually generalizes.
  2. 12-MONTH horizon - where housing momentum is strong (annual autocorr +0.75,
     the one asset class where trend forecasting genuinely works).
  3. WALK-FORWARD validation that reports the real skill: does the model's metro
     RANKING predict the actual forward ranking OOS (cross-sectional rank IC), and
     do its top metros beat its bottom metros out of sample?

Keyless FRED via pandas_datareader (no API key). Case-Shiller 20 metros (monthly).
Features are metro-specific (momentum, valuation vs trend, vol) + national macro
(mortgage, curve) as common factors. Output: current ranking + the honest OOS skill.
"""
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

CS_METROS = {
    "ATXRNSA": "Atlanta", "BOXRNSA": "Boston", "CHXRNSA": "Chicago", "SDXRNSA": "San Diego",
    "DAXRNSA": "Dallas", "DNXRNSA": "Denver", "DEXRNSA": "Detroit", "LXXRNSA": "Los Angeles",
    "MIXRNSA": "Miami", "MNXRNSA": "Minneapolis", "NYXRNSA": "New York", "PHXRNSA": "Phoenix",
    "POXRNSA": "Portland", "LVXRNSA": "Las Vegas", "SFXRNSA": "San Francisco",
    "SEXRNSA": "Seattle", "TPXRNSA": "Tampa", "WDXRNSA": "Washington DC", "CEXRNSA": "Cleveland",
}
MACRO = {"MORTGAGE30US": "mortgage", "T10Y2Y": "curve", "PERMIT": "permits", "UNRATE": "unrate"}
HORIZON = 12   # months
FEATS = ["mom12", "mom6", "px_vs_ma36", "px_vs_ma_long", "vol12", "mortgage_chg", "curve", "permits_yoy", "unrate_chg"]


def build_panel():
    print("fetching Case-Shiller metros + macro (keyless FRED) ...")
    mac = web.DataReader(list(MACRO), "fred", "1990-01-01").rename(columns=MACRO).resample("MS").last().ffill()
    mac_f = pd.DataFrame(index=mac.index)
    mac_f["mortgage_chg"] = mac["mortgage"].diff(12)
    mac_f["curve"] = mac["curve"]
    mac_f["permits_yoy"] = mac["permits"].pct_change(12) * 100
    mac_f["unrate_chg"] = mac["unrate"].diff(6)

    rows = []
    for sid, name in CS_METROS.items():
        try:
            s = web.DataReader(sid, "fred", "1990-01-01").iloc[:, 0].resample("MS").last().dropna()
        except Exception:
            continue
        if len(s) < 180:
            continue
        d = pd.DataFrame({"px": s})
        d["mom12"] = s.pct_change(12) * 100
        d["mom6"] = s.pct_change(6) * 100
        d["px_vs_ma36"] = (s / s.rolling(36).mean() - 1) * 100
        d["px_vs_ma_long"] = (s / s.rolling(120).mean() - 1) * 100
        d["vol12"] = s.pct_change().rolling(12).std() * 100
        d["target"] = (s.shift(-HORIZON) / s - 1) * 100     # forward 12m return
        d = d.join(mac_f)
        d["metro"] = name
        rows.append(d.reset_index().rename(columns={"index": "date", "DATE": "date"}))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=FEATS + ["target"])
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values("date").reset_index(drop=True)


def _gbm():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=30, random_state=42)


def compute_metro_ranking():
    """Returns {asof, skill:{rank_ic, ic_pos, top_bot}, ranking:[{metro,fwd12,mom12,vs36,vol}]}.
    Walk-forward validated; current ranking from a model trained on all matured data."""
    panel = build_panel()
    oof = []
    for yr in range(2012, 2026):
        cut = pd.Timestamp(f"{yr}-01-01") - pd.DateOffset(months=HORIZON + 1)
        tr = panel[panel["date"] <= cut]
        te = panel[(panel["date"] >= pd.Timestamp(f"{yr}-01-01")) & (panel["date"] < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(tr) < 500 or len(te) < 20:
            continue
        m = _gbm(); m.fit(tr[FEATS], tr["target"])
        te = te.copy(); te["pred"] = m.predict(te[FEATS]); oof.append(te)
    oof = pd.concat(oof)
    ics = [spearmanr(g["pred"], g["target"]).correlation for _, g in oof.groupby("date") if len(g) >= 5]
    ics = [x for x in ics if x == x]
    oof["bucket"] = oof.groupby("date")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["bot", "mid", "top"]))
    tb = oof.groupby("bucket")["target"].mean()
    skill = {"rank_ic": round(float(np.mean(ics)), 3), "ic_pos": round(float(np.mean(np.array(ics) > 0)), 2),
             "top_bot": round(float(tb.get("top", 0) - tb.get("bot", 0)), 1), "n": int(len(oof))}

    matured = panel[panel["date"] <= panel["date"].max() - pd.DateOffset(months=HORIZON)]
    model = _gbm(); model.fit(matured[FEATS], matured["target"])
    latest = panel.sort_values("date").groupby("metro").tail(1).copy()
    latest["pred"] = model.predict(latest[FEATS])
    latest = latest.sort_values("pred", ascending=False)
    ranking = [{"metro": r["metro"], "fwd12": round(float(r["pred"]), 1), "mom12": round(float(r["mom12"]), 1),
                "vs36": round(float(r["px_vs_ma36"]), 1), "vol": round(float(r["vol12"]), 1)}
               for _, r in latest.iterrows()]
    return {"asof": str(panel["date"].max().date()), "skill": skill, "ranking": ranking}


def main():
    d = compute_metro_ranking()
    s = d["skill"]
    print("=== WALK-FORWARD OOS validation ===")
    print(f"monthly cross-sectional rank IC: {s['rank_ic']:+.3f}  (positive {s['ic_pos']:.0%} of months, n={s['n']:,})")
    print(f"top vs bottom third: +{s['top_bot']:.1f}%/yr actual OOS spread")
    print(f"\n=== CURRENT metro ranking (fwd-12m HPA forecast, as of {d['asof']}) ===")
    print(f"{'#':>2} {'metro':16s} {'fwd12m%':>8s} {'mom12%':>7s} {'vs36mMA%':>9s} {'vol%':>5s}")
    for i, r in enumerate(d["ranking"], 1):
        print(f"{i:>2} {r['metro']:16s} {r['fwd12']:>+7.1f} {r['mom12']:>+7.1f} {r['vs36']:>+9.1f} {r['vol']:>5.1f}")
    print("\n[honest read: trust the RANK, not the precise % - OOS rank IC above is the real skill.]")


if __name__ == "__main__":
    main()
