"""
US residential real estate: is there a bottom to call, and what has marked bottoms before?

Same discipline as housing/dubai/analyze_dubai.py and housing/can_housing_be_timed.py:
walk-forward only, gated on the best trivial baseline, turning points scored separately,
and every claim reported whether or not it flatters the model.

WHY THE QUESTION IS NOT THE DUBAI QUESTION. Dubai is ~10% below a peak. US national
nominal prices (Case-Shiller, SA) are AT an all-time high. So "when will it bottom" splits:
  1. nominal national  - nothing to bottom; the question is whether a peak is forming
  2. real national     - deflated by CPI, prices are still below their 2022 peak
  3. metros            - several are genuinely down from their own peaks
  4. affordability     - the thing that has historically ended price declines

All data keyless FRED via pandas_datareader.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web

warnings.filterwarnings("ignore")
DIR = os.path.dirname(os.path.abspath(__file__))
START = "1987-01-01"

METROS = {
    "ATXRSA": "Atlanta", "BOXRSA": "Boston", "CHXRSA": "Chicago", "SDXRSA": "San Diego",
    "DAXRSA": "Dallas", "DNXRSA": "Denver", "DEXRSA": "Detroit", "LXXRSA": "Los Angeles",
    "MIXRSA": "Miami", "MNXRSA": "Minneapolis", "NYXRSA": "New York", "PHXRSA": "Phoenix",
    "POXRSA": "Portland", "LVXRSA": "Las Vegas", "SFXRSA": "San Francisco",
    "SEXRSA": "Seattle", "TPXRSA": "Tampa", "WDXRSA": "Washington DC", "CEXRSA": "Cleveland",
    "CRXRSA": "Charlotte",
}
HORIZONS = (6, 12, 24)
MIN_TRAIN = 120


def fred(sid, start=START):
    return web.DataReader(sid, "fred", start).iloc[:, 0].dropna()


def mstart(s):
    s = s.copy(); s.index = s.index.to_period("M").to_timestamp(); return s


def load():
    cs = mstart(fred("CSUSHPISA"))
    cpi = mstart(fred("CPIAUCSL", "1985-01-01")).reindex(cs.index).ffill()
    real = cs / cpi; real = real / real.iloc[0] * 100
    mort = fred("MORTGAGE30US", "1985-01-01").resample("MS").mean()
    price = mstart(fred("MSPUS", "1985-01-01")).resample("MS").ffill()
    inc = fred("MEHOINUSA646N", "1985-01-01")
    return cs, real, cpi, mort, price, inc


# --------------------------------------------------------------------- cycle anatomy
def episodes(px, min_depth):
    dd = px / px.cummax() - 1
    out, in_dd, pk = [], False, 0
    for i, d in enumerate(dd.values):
        if not in_dd and d < -0.001:
            in_dd, pk = True, i - 1
        elif in_dd and d >= -0.001:
            out.append(_ep(px.iloc[pk:i + 1], True)); in_dd = False
    if in_dd:
        out.append(_ep(px.iloc[pk:], False))
    return [e for e in out if e["depth"] <= -min_depth]


def _ep(seg, rec):
    tr = seg.idxmin()
    return {"peak": str(seg.index[0])[:7], "trough": str(tr)[:7], "recovered": str(seg.index[-1])[:7] if rec else None,
            "depth": float(seg.min() / seg.iloc[0] - 1),
            "months_down": int((tr.year - seg.index[0].year) * 12 + tr.month - seg.index[0].month),
            "months_to_recover": int((seg.index[-1].year - tr.year) * 12 + seg.index[-1].month - tr.month) if rec else None}


# ----------------------------------------------------------------------- affordability
def affordability(price, mort, inc):
    """Monthly P&I on the median sale price (20% down, 30y) as a share of median family income.
    Annual income is stepped forward to the latest year (no look-ahead: held at last published)."""
    inc_m = inc.copy(); inc_m.index = inc_m.index.to_period("M").to_timestamp()
    inc_m = inc_m.reindex(pd.date_range(inc_m.index[0], price.index[-1], freq="MS")).ffill()
    idx = price.index.intersection(mort.index).intersection(inc_m.index)
    p, r, y = price.loc[idx], mort.loc[idx] / 1200, inc_m.loc[idx] / 12
    pay = p * 0.8 * r / (1 - (1 + r) ** -360)
    return (pay / y).rename("pti")


# ------------------------------------------------------------------ walk-forward skill
def _fit(lp, h):
    tr, fw = lp.diff(h), lp.diff(h).shift(-h)
    m = tr.notna() & fw.notna()
    if m.sum() < 24:
        return None
    b, a = np.polyfit(tr[m], fw[m], 1)
    return float(b), float(a)


def walk_forward(px, h):
    lp = np.log(px); rows = []
    for t in range(MIN_TRAIN, len(px) - h):
        lh = lp.iloc[:t + 1]; f = _fit(lh, h)
        if f is None:
            continue
        b, a = f
        rows.append({"date": px.index[t], "pred": a + b * (lh.iloc[-1] - lh.iloc[-1 - h]),
                     "actual": float(lp.iloc[t + h] - lp.iloc[t]), "drift": float(lh.diff().mean() * h),
                     "last": float(lh.iloc[-1] - lh.iloc[-1 - h])})   # "next h looks like last h"
    return pd.DataFrame(rows)


def validate(px):
    out = {}
    for h in HORIZONS:
        wf = walk_forward(px, h)
        rm = lambda c: float(np.sqrt(((wf[c] - wf.actual) ** 2).mean()))   # noqa: E731
        m, rw, dr, ls = rm("pred"), float(np.sqrt((wf.actual ** 2).mean())), rm("drift"), rm("last")
        # turning points: trailing sign differs from forward sign
        turn = np.sign(wf["last"]) != np.sign(wf.actual)
        sub = wf[turn]
        out[h] = {"n": int(len(wf)), "skill_vs_rw": 1 - m / rw, "skill_vs_drift": 1 - m / dr,
                  "skill_vs_last": 1 - m / ls, "hit": float((np.sign(wf.pred) == np.sign(wf.actual)).mean()),
                  "resid_sd": float((wf.pred - wf.actual).std()),
                  "turn_n": int(turn.sum()),
                  "turn_hit": float((np.sign(sub.pred) == np.sign(sub.actual)).mean()) if len(sub) else None}
    return out


def forecast(px, val, real=False):
    lp = np.log(px); out = {}
    for h in HORIZONS:
        v = val[h]
        if v["skill_vs_drift"] <= 0 and h > 12:
            out[h] = {"withheld": True, "skill": v["skill_vs_rw"]}; continue
        b, a = _fit(lp, h)
        pt = a + b * (lp.iloc[-1] - lp.iloc[-1 - h]); sd = v["resid_sd"]
        out[h] = {"withheld": False, "point": float(np.expm1(pt)), "lo": float(np.expm1(pt - 1.28 * sd)),
                  "hi": float(np.expm1(pt + 1.28 * sd)), "trail": float(np.expm1(lp.iloc[-1] - lp.iloc[-1 - h])),
                  "skill": v["skill_vs_rw"]}
    return out


# ------------------------------------------------------------------- metro bottom test
def load_metros():
    d = {}
    for sid, nm in METROS.items():
        try:
            s = mstart(fred(sid, "1987-01-01"))
            if len(s) > 200:
                d[nm] = s
        except Exception:
            pass
    return pd.DataFrame(d)


def metro_state(m):
    rows = []
    for c in m.columns:
        s = m[c].dropna()
        rows.append({"metro": c, "last": str(s.index[-1])[:7],
                     "dd": float(s.iloc[-1] / s.max() - 1), "peak": str(s.idxmax())[:7],
                     "yoy": float(s.iloc[-1] / s.iloc[-13] - 1), "m3": float((s.iloc[-1] / s.iloc[-4]) ** 4 - 1),
                     "m6": float((s.iloc[-1] / s.iloc[-7]) ** 2 - 1)})
    return pd.DataFrame(rows).sort_values("dd")


def metro_bottom_test(m, dd_floor=-0.03, fwd=12):
    """Inside a metro drawdown deeper than 3%, flag months where the 3m annualised change has turned
    positive after being negative 3 months earlier (a 'turn'). Compare the next-12m move and the
    further downside with all other drawdown months. Pooled across metros, but episodes cluster in
    2007-12 and 2022-23, so report by era."""
    rows = []
    for c in m.columns:
        s = m[c].dropna(); lp = np.log(s)
        dd = s / s.cummax() - 1
        m3 = (lp - lp.shift(3)) * 4
        for t in range(12, len(s) - fwd):
            if dd.iloc[t] > dd_floor:
                continue
            turned = bool(m3.iloc[t] > 0 and m3.iloc[t - 3] < 0)
            flat3 = bool(abs(m3.iloc[t]) < 0.01 and abs(m3.iloc[t - 3]) < 0.02)
            rows.append({"metro": c, "date": s.index[t], "turned": turned, "flat": flat3,
                         "fwd12": float(lp.iloc[t + fwd] - lp.iloc[t]),
                         "further": float(s.iloc[t:t + fwd + 1].min() / s.iloc[t] - 1), "dd": float(dd.iloc[t])})
    return pd.DataFrame(rows)


def summarize_turns(d):
    out = {}
    for lab, mask in [("all", d.date.notna()), ("2007-12", (d.date.dt.year >= 2007) & (d.date.dt.year <= 2012)),
                      ("other", ~((d.date.dt.year >= 2007) & (d.date.dt.year <= 2012)))]:
        sub = d[mask]
        for k in ("turned", "flat"):
            f, o = sub[sub[k]], sub[~sub[k]]
            out[f"{lab}_{k}"] = {"n": int(len(f)), "metros": int(f.metro.nunique()),
                                 "fwd12": float(f.fwd12.mean()) if len(f) else None,
                                 "pos": float((f.fwd12 > 0).mean()) if len(f) else None,
                                 "further_mean": float(f.further.mean()) if len(f) else None,
                                 "further_lt5": float((f.further < -0.05).mean()) if len(f) else None,
                                 "other_fwd12": float(o.fwd12.mean()) if len(o) else None,
                                 "other_pos": float((o.fwd12 > 0).mean()) if len(o) else None,
                                 "other_n": int(len(o))}
    return out


# -------------------------------------------------------------------------- orchestrate
def run():
    cs, real, cpi, mort, price, inc = load()
    pti = affordability(price, mort, inc)
    m = load_metros()
    eps_nom = episodes(cs, 0.02)
    eps_real = episodes(real, 0.05)
    val_n, val_r = validate(cs), validate(real)
    d = metro_bottom_test(m)
    pti_pct = float((pti < pti.iloc[-1]).mean())
    res = {
        "asof": str(cs.index[-1])[:7],
        "cs_last": float(cs.iloc[-1]), "cs_peak": str(cs.idxmax())[:7], "cs_dd": float(cs.iloc[-1] / cs.max() - 1),
        "cs_yoy": float(cs.iloc[-1] / cs.iloc[-13] - 1), "cs_m3": float((cs.iloc[-1] / cs.iloc[-4]) ** 4 - 1),
        "cpi_yoy": float(cpi.iloc[-1] / cpi.iloc[-13] - 1),
        "real_last": float(real.iloc[-1]), "real_dd": float(real.iloc[-1] / real.max() - 1),
        "real_peak": str(real.idxmax())[:7], "real_trough_since": str(real.loc[real.idxmax():].idxmin())[:7],
        "real_yoy": float(real.iloc[-1] / real.iloc[-13] - 1), "real_m6": float((real.iloc[-1] / real.iloc[-7]) ** 2 - 1),
        "eps_nom": eps_nom, "eps_real": eps_real,
        "val_nom": val_n, "val_real": val_r,
        "fc_nom": forecast(cs, val_n), "fc_real": forecast(real, val_r),
        "mort_last": float(mort.iloc[-1]), "mort_peak": float(mort.loc["2022":].max()),
        "mort_peak_date": str(mort.loc["2022":].idxmax())[:7],
        "pti_last": float(pti.iloc[-1]), "pti_pct": pti_pct, "pti_max": float(pti.max()), "pti_max_date": str(pti.idxmax())[:7],
        "pti_min": float(pti.min()), "pti_min_date": str(pti.idxmin())[:7],
        "metros": metro_state(m).to_dict("records"),
        "turns": summarize_turns(d),
        "n_obs": int(len(cs)),
    }
    # affordability at each real/nominal bottom
    res["pti_at"] = {k: float(pti.asof(pd.Timestamp(k + "-01"))) for k in ("2006-04", "2012-02", "2023-01", "1990-01", "1997-01")
                     if pd.Timestamp(k + "-01") >= pti.index[0]}
    series = pd.DataFrame({"nom": cs, "real": real, "pti": pti.reindex(cs.index), "mort": mort.reindex(cs.index)})
    series.to_csv(os.path.join(DIR, "us_monthly.csv"))
    json.dump(res, open(os.path.join(DIR, "us_results.json"), "w"), indent=1, default=str)
    return res, series, m, d


if __name__ == "__main__":
    r, s, m, d = run()
    print(json.dumps({k: v for k, v in r.items() if k not in ("metros",)}, indent=1, default=str))
    print(pd.DataFrame(r["metros"]).round(3).to_string())


# ------------------------------------------------------------------------------ extras
def extras():
    """Base rate for a future national decline, and an affordability scenario table. The scenario
    inputs (income growth) are measured from the data; rate and price paths are labelled scenarios,
    not forecasts."""
    s = pd.read_csv(os.path.join(DIR, "us_monthly.csv"), index_col=0, parse_dates=True)
    cs, pti, mort = s["nom"], s["pti"], s["mort"]
    near = cs / cs.cummax() - 1 > -0.01                      # at/near an all-time high
    fwd_min = pd.concat([cs.shift(-k) for k in range(1, 25)], axis=1).min(axis=1) / cs - 1
    ok = near & fwd_min.notna()
    base = {"n_months": int(ok.sum()), "p_lt5": float((fwd_min[ok] < -0.05).mean()),
            "p_lt10": float((fwd_min[ok] < -0.10).mean()),
            "years_lt5": sorted({d.year for d in fwd_min[ok][fwd_min[ok] < -0.05].index})}
    inc = fred("MEHOINUSA646N", "1985-01-01")
    inc_cagr = float((inc.iloc[-1] / inc.iloc[-11]) ** (1 / 10) - 1)
    # scenarios: price growth g (nominal, per yr), mortgage rate path, income growth = trailing 10y CAGR
    price = mstart(fred("MSPUS", "1985-01-01")).iloc[-1]
    inc_now = float(inc.iloc[-1])

    def pti_at(p, r, y):
        rm = r / 1200
        return p * 0.8 * rm / (1 - (1 + rm) ** -360) / (y / 12)

    cur = pti_at(price, float(mort.iloc[-1]), inc_now)
    scen = []
    for g in (0.0, 0.02, 0.04):
        for r in (7.2, 6.0, 5.0):
            hit = {}
            for tgt, lab in ((0.25, "to25"), (0.21, "to21")):
                yrs = None
                for k in range(1, 121):
                    t = k / 12
                    if pti_at(price * (1 + g) ** t, r, inc_now * (1 + inc_cagr) ** t) <= tgt:
                        yrs = round(t, 1); break
                hit[lab] = yrs
            scen.append({"price_g": g, "rate": r, **hit, "pti_in_2y": float(pti_at(price * (1 + g) ** 2, r, inc_now * (1 + inc_cagr) ** 2))})
    return {"base_rate": base, "inc_cagr": inc_cagr, "pti_now_calc": float(cur), "scen": scen, "inc_last_year": str(inc.index[-1])[:4]}


if __name__ == "__main__":
    ex = extras()
    r0 = json.load(open(os.path.join(DIR, "us_results.json"))); r0["extras"] = ex
    json.dump(r0, open(os.path.join(DIR, "us_results.json"), "w"), indent=1, default=str)
    print(json.dumps(ex, indent=1))
