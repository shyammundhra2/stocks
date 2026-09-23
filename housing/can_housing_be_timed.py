"""
Can US house prices be TIMED from the standard leading-indicator checklist?

Tests the six indicators people are told to watch - existing home sales, new
building permits, mortgage defaults, foreclosures, mortgage rates, months' supply
- against Case-Shiller national, the same way techjobs/ tested tech employment.

The discipline that matters, learned the hard way in this repo:

  1. Gate on the BEST trivial baseline, not a convenient one. Home prices are
     enormously persistent, so "next year looks like this year" is very hard to
     beat and beating a random walk with no drift proves nothing.
  2. Score TURNING POINTS separately. Averaged over all months, any persistence
     model looks good because most months are not turns - and turns are the only
     months a timing signal is for.
  3. Check lag-selection bias. Scanning 24 lags and reporting the best one
     manufactures a lead that will not repeat; compare against a fixed lag.

Series (all keyless FRED). Existing home sales and existing-home months' supply
return only ~13 observations through pandas_datareader, so they are reported as
unavailable rather than silently dropped - the checklist's item (1) and (6b)
cannot be tested here. Foreclosure sales has no clean free national monthly
series either; mortgage DELINQUENCY (DRSFRMACBS) is the closest proxy and is
used for items (3)/(4) together.
"""
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web

warnings.filterwarnings("ignore")

TARGET = "CSUSHPINSA"          # Case-Shiller national, monthly, 1987+
INDICATORS = {
    "permits": "PERMIT",            # (2) new building permits
    "delinq": "DRSFRMACBS",         # (3)+(4) mortgage defaults / foreclosure proxy
    "mortrate": "MORTGAGE30US",     # (5) 30yr mortgage rate
    "supply": "MSACSR",             # (6) months' supply, new homes
    "starts": "HOUST",              # bonus: housing starts
}
UNAVAILABLE = {
    "existing_home_sales": "EXHOSLUSM495S (~13 obs via pandas_datareader)",
    "months_supply_existing": "HOSSUPUSM673N (~13 obs)",
    "foreclosure_sales": "no clean free national monthly series; proxied by delinq",
}
HORIZON = 12                   # months ahead
MIN_TRAIN = 120                # 10y before the first walk-forward forecast


def load():
    px = web.DataReader(TARGET, "fred", "1970-01-01").iloc[:, 0].dropna()
    px = px.resample("MS").last()
    raw = {}
    for name, sid in INDICATORS.items():
        s = web.DataReader(sid, "fred", "1970-01-01").iloc[:, 0].dropna()
        # ffill BEFORE differencing. DRSFRMACBS is quarterly, so resampling to
        # month-start leaves NaN in 2 of every 3 months; diffing that first made
        # the whole delinquency column NaN and silently dropped the one indicator
        # that speaks to items (3) and (4) of the checklist.
        raw[name] = s.resample("MS").mean().ffill()

    f = pd.DataFrame(index=px.index)
    f["permits_yoy"] = raw["permits"].pct_change(12) * 100
    f["starts_yoy"] = raw["starts"].pct_change(12) * 100
    f["delinq_chg"] = raw["delinq"].diff(4)            # quarterly series
    f["mortrate_chg"] = raw["mortrate"].diff(12)
    f["supply_chg"] = raw["supply"].diff(12)
    f["supply_lvl"] = raw["supply"]
    return px, f.reindex(px.index).ffill()


def lead_lag(px, f, max_lead=36):
    """Does each indicator actually LEAD prices, and does the lead survive?"""
    lp = np.log(px)
    rows = []
    for c in f.columns:
        s = f[c].dropna()
        best = (0, 0.0)
        for k in range(1, max_lead + 1):
            fwd = lp.shift(-k) - lp
            j = s.index.intersection(fwd.dropna().index)
            if len(j) < 120:
                continue
            r = float(np.corrcoef(s.loc[j], fwd.loc[j])[0, 1])
            if abs(r) > abs(best[1]):
                best = (k, r)
        fwd12 = lp.shift(-12) - lp
        j = s.index.intersection(fwd12.dropna().index)
        r12 = float(np.corrcoef(s.loc[j], fwd12.loc[j])[0, 1]) if len(j) > 120 else np.nan
        rows.append({"indicator": c, "best_lead_m": best[0], "best_corr": best[1],
                     "fixed_12m_corr": r12, "shrinkage": abs(best[1]) - abs(r12)})
    return pd.DataFrame(rows).sort_values("best_corr", key=abs, ascending=False)


def walk_forward(px, f, horizon=HORIZON):
    """Expanding-window OLS on the indicators, vs three trivial baselines."""
    lp = np.log(px)
    y = (lp.shift(-horizon) - lp) * 100
    X = f.shift(1)                                    # publication lag
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    cols = list(X.columns)

    rows = []
    for i in range(MIN_TRAIN, len(d) - horizon):
        tr = d.iloc[:i - horizon]                     # matured targets only
        if len(tr) < 60:
            continue
        A = np.c_[np.ones(len(tr)), tr[cols].values]
        beta = np.linalg.lstsq(A, tr.y.values, rcond=None)[0]
        pred = float(np.r_[1, d[cols].iloc[i].values] @ beta)
        t = d.index[i]
        trail = float((lp.loc[t] - lp.shift(horizon).loc[t]) * 100) if t in lp.index else np.nan
        rows.append({"date": t, "pred": pred, "actual": float(d.y.iloc[i]),
                     "trail": trail, "drift": float(tr.y.mean())})
    return pd.DataFrame(rows).dropna()


def score(wf):
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))   # noqa: E731
    y = wf.actual
    base = {
        "no-change": rmse(pd.Series(0.0, index=y.index), y),
        "persistence": rmse(wf.trail, y),      # next 12m = last 12m
        "avg drift": rmse(wf.drift, y),
    }
    m = rmse(wf.pred, y)
    best = min(base, key=base.get)
    turn = np.sign(wf.actual) != np.sign(wf.trail)
    out = {"n": len(wf), "rmse": m, "baselines": base, "best_baseline": best,
           "skill_vs_best": 1 - m / base[best],
           "dir_acc": float((np.sign(wf.pred) == np.sign(y)).mean())}
    for lbl, msk in [("trend_continues", ~turn), ("TURNS", turn)]:
        if msk.sum() >= 12:
            out[lbl] = {"n": int(msk.sum()), "share": float(msk.mean()),
                        "skill_vs_nochange": float(
                            1 - rmse(wf.pred[msk], y[msk]) / rmse(0 * y[msk], y[msk])),
                        "dir_acc": float((np.sign(wf.pred[msk]) == np.sign(y[msk])).mean())}
    return out


def main():
    px, f = load()
    print(f"Case-Shiller national: {len(px)} months, "
          f"{px.index[0].date()} .. {px.index[-1].date()}\n")
    print("NOT TESTABLE from free monthly data:")
    for k, v in UNAVAILABLE.items():
        print(f"   {k:24s} {v}")

    print("\n=== 1. DO THEY ACTUALLY LEAD? ===")
    ll = lead_lag(px, f)
    print(f"{'indicator':>14} {'best lead':>10} {'corr':>7} {'fixed 12m':>10} {'shrinkage':>10}")
    for _, r in ll.iterrows():
        print(f"{r['indicator']:>14} {int(r['best_lead_m']):>9}m {r['best_corr']:>+7.2f} "
              f"{r['fixed_12m_corr']:>+10.2f} {r['shrinkage']:>+10.2f}")
    print("  (large shrinkage = the 'lead' is an artifact of scanning 36 lags)")

    print(f"\n=== 2. WALK-FORWARD, {HORIZON}-MONTH HORIZON ===")
    wf = walk_forward(px, f)
    sc = score(wf)
    print(f"n={sc['n']} OOS forecasts, {wf.date.min().date()} .. {wf.date.max().date()}")
    print(f"model RMSE {sc['rmse']:.2f}pp  vs baselines: " +
          "  ".join(f"{k} {v:.2f}" for k, v in sc["baselines"].items()))
    print(f"  best trivial rule is '{sc['best_baseline']}'; "
          f"model beats it by {sc['skill_vs_best']:+.1%}")
    print(f"  direction accuracy {sc['dir_acc']:.0%}")
    print("\n  -- split by whether the trend actually turned --")
    for lbl in ("trend_continues", "TURNS"):
        if lbl in sc:
            s = sc[lbl]
            print(f"     {lbl:>16} n={s['n']:>3} ({s['share']:.0%})  "
                  f"skill vs no-change {s['skill_vs_nochange']:>+7.1%}  "
                  f"direction {s['dir_acc']:.0%}")

    print("\n=== 3. VERDICT ===")
    usable = sc["skill_vs_best"] > 0
    turns_ok = sc.get("TURNS", {}).get("dir_acc", 0) > 0.55
    print(f"  beats the best trivial rule?     {'YES' if usable else 'NO'}")
    print(f"  calls turning points better than a coin flip?  "
          f"{'YES' if turns_ok else 'NO'} "
          f"({sc.get('TURNS', {}).get('dir_acc', float('nan')):.0%})")


if __name__ == "__main__":
    main()
