"""
Tech labour market: can it be forecast? Lead-lag diagnostics + walk-forward validation.

Same discipline as housing/predict_housing.py and housing/dubai: nothing is claimed that
wasn't measured out-of-sample, and horizons where the model has no skill return no number.

THE BASELINE THAT MATTERS. The model is scored against three trivial rules - no change,
average drift, and naive momentum (forward growth = trailing growth) - and must beat the
BEST of them, not a convenient one. That distinction turned out to decide the result here:
tech employment growth mean-reverts, so naive momentum is a genuinely bad rule and beating
it is easy and meaningless. No-change is the rule to beat, and the model does not beat it.

WHAT THIS FOUND (see main() output):
  * The LEVEL is not forecastable. At both 6m and 12m the model loses to a no-change
    baseline, so ``forecast`` withholds every point estimate of employment growth.
  * The RANKING is. Cross-sectional rank IC is +0.52 at 6m and +0.40 at 12m, positive in
    90%/82% of months out-of-sample. Which tech sub-industries do better than others is
    predictable even though how much any of them grows is not. Same shape as the finding
    in housing/predict_housing.py - trust the rank, not the number.
  * Turning points are not forecastable at all. On months where the trend actually turned,
    direction accuracy is 38% (6m) and 43% (12m) - worse than a coin flip.

THE TEST THAT MATTERS. Averaged over all months, any persistence model looks good because
most months are not turning points. ``turning_point_skill`` scores the model only on months
where the forward direction differed from the trailing direction - the months you would
actually want a forecast for.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

from .build_panel import FEATS, INDUSTRIES, fetch_diagnostics, fetch_industries, load

WF_START, WF_END = 1999, 2026     # walk-forward test years
MIN_SKILL = 0.0                   # must beat naive momentum to publish a forecast


def _gbm():
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                     subsample=0.8, random_state=0)


# ------------------------------------------------------------------ lead-lag diagnostic
def lead_lag(max_lead: int = 18) -> pd.DataFrame:
    """Does each indicator actually LEAD tech employment, and by how long?

    Correlates indicator at time t with tech employment growth over t..t+k, scanning k.
    The peak-correlation lag is the honest answer to "how much warning does this give".
    """
    ind = fetch_industries()
    tech = np.log(ind.sum(axis=1))            # total tech employment, log
    diag = fetch_diagnostics()

    from .build_panel import fetch_macro
    mac = fetch_macro()
    cand = pd.concat([mac[["temphelp_yoy", "claims_yoy", "curve", "nasdaq_12m", "hours_chg"]],
                      diag.apply(lambda s: s.pct_change(12) * 100)], axis=1)

    rows = []
    for c in cand.columns:
        s = cand[c].dropna()
        best = (0, 0.0, 0)
        for k in range(1, max_lead + 1):
            fwd = tech.shift(-k) - tech           # employment growth over next k months
            j = s.index.intersection(fwd.dropna().index)
            if len(j) < 60:
                continue
            r = np.corrcoef(s.loc[j], fwd.loc[j])[0, 1]
            if abs(r) > abs(best[1]):
                best = (k, float(r), len(j))
        rows.append({"indicator": c, "best_lead_m": best[0], "corr": best[1], "n": best[2]})
    return pd.DataFrame(rows).sort_values("corr", key=abs, ascending=False)


# ------------------------------------------------------------- walk-forward validation
def walk_forward(horizon: int = 12) -> pd.DataFrame:
    """Expanding-window OOS predictions, retrained each year on matured targets only."""
    p = load(horizon)
    tr_all = p.dropna(subset=["target", "naive_mom"])
    out = []
    for yr in range(WF_START, WF_END + 1):
        # only targets that had fully matured before the test year begins
        cut = pd.Timestamp(f"{yr}-01-01") - pd.DateOffset(months=horizon + 1)
        tr = tr_all[tr_all.date <= cut]
        te = tr_all[(tr_all.date >= f"{yr}-01-01") & (tr_all.date < f"{yr+1}-01-01")]
        if len(tr) < 300 or len(te) < 6:
            continue
        m = _gbm().fit(tr[FEATS], tr["target"])
        te = te.copy()
        te["pred"] = m.predict(te[FEATS])
        te["drift"] = tr["target"].mean()
        out.append(te)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score(wf: pd.DataFrame) -> dict:
    """RMSE against the three baselines, plus rank IC and direction accuracy."""
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    y = wf["target"]
    r_model = rmse(wf["pred"], y)
    base = {"zero": rmse(pd.Series(0.0, index=y.index), y),
            "drift": rmse(wf["drift"], y),
            "naive_mom": rmse(wf["naive_mom"], y)}

    ics = [spearmanr(g["pred"], g["target"]).correlation
           for _, g in wf.groupby("date") if len(g) >= 4]
    ics = [x for x in ics if x == x]
    # the bar is the BEST trivial rule, not a convenient one
    best_name = min(base, key=base.get)
    return {
        "n": int(len(wf)), "rmse": r_model, "baselines": base,
        "skill": {k: float(1 - r_model / v) for k, v in base.items()},
        "best_baseline": best_name,
        "skill_vs_best": float(1 - r_model / base[best_name]),
        "rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "ic_pos": float(np.mean(np.array(ics) > 0)) if ics else float("nan"),
        "dir_acc": float((np.sign(wf["pred"]) == np.sign(y)).mean()),
        "dir_acc_naive": float((np.sign(wf["naive_mom"]) == np.sign(y)).mean()),
    }


def turning_point_skill(wf: pd.DataFrame) -> dict:
    """Score only on months where the trend actually turned - the months that matter.

    A 'turn' is a row whose forward direction differs from its trailing direction, i.e.
    exactly the case naive momentum gets wrong by construction.
    """
    turn = np.sign(wf["target"]) != np.sign(wf["naive_mom"])
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    out = {}
    for lbl, mask in [("trend_continues", ~turn), ("trend_turns", turn)]:
        s = wf[mask]
        if len(s) < 20:
            continue
        out[lbl] = {
            "n": int(len(s)), "share": float(mask.mean()),
            "rmse": rmse(s["pred"], s["target"]),
            "rmse_naive": rmse(s["naive_mom"], s["target"]),
            "skill_vs_naive": float(1 - rmse(s["pred"], s["target"]) / rmse(s["naive_mom"], s["target"])),
            "dir_acc": float((np.sign(s["pred"]) == np.sign(s["target"])).mean()),
        }
    return out


# ------------------------------------------------------------------------- forecast
def forecast(horizon: int, sc: dict) -> dict:
    """Ranking if the rank IC earned it; a level forecast only if it beat every trivial rule.

    These two gates come apart in this data, which is the whole result: the ordering of
    tech sub-industries is predictable, the growth rates are not. The ``pred`` values are
    therefore returned as a RANKING KEY, not as forecasts of employment growth, and the
    caller is told so explicitly.
    """
    p = load(horizon)
    tr = p.dropna(subset=["target"])
    m = _gbm().fit(tr[FEATS], tr["target"])
    latest = p.sort_values("date").groupby("industry").tail(1).copy()
    latest["pred"] = m.predict(latest[FEATS])

    levels_ok = sc["skill_vs_best"] > MIN_SKILL
    rank_ok = sc["rank_ic"] > 0.05 and sc["ic_pos"] > 0.6
    return {
        "levels_published": bool(levels_ok),
        "levels_reason": None if levels_ok else (
            f"OOS skill {sc['skill_vs_best']:+.1%} vs the best trivial rule "
            f"('{sc['best_baseline']}') at {horizon}m - the model does not beat it, so no "
            f"point estimate of employment growth is published"),
        "ranking_published": bool(rank_ok),
        "ranking_basis": f"rank IC {sc['rank_ic']:+.3f}, positive in {sc['ic_pos']:.0%} of OOS months",
        "asof": str(latest.date.max().date()),
        "rows": latest.sort_values("pred", ascending=False)[
            ["industry", "emp", "mom12", "naive_mom", "pred"]].to_dict("records"),
    }


def compute_techjobs_outlook() -> dict:
    out = {"lead_lag": lead_lag().to_dict("records"), "horizons": {}}
    for h in (6, 12):
        wf = walk_forward(h)
        if wf.empty:
            continue
        sc = score(wf)
        out["horizons"][h] = {"score": sc, "turning_points": turning_point_skill(wf),
                              "forecast": forecast(h, sc)}
    return out


def main():
    d = compute_techjobs_outlook()

    print("=== DO THE LEADING INDICATORS ACTUALLY LEAD? ===")
    print("(indicator at t vs total tech employment growth over t..t+k, best k)")
    print(f"{'indicator':>16} {'lead':>6} {'corr':>7} {'n':>6}")
    for r in d["lead_lag"]:
        print(f"{r['indicator']:>16} {r['best_lead_m']:>5}m {r['corr']:>+7.2f} {r['n']:>6}")

    for h, blk in d["horizons"].items():
        sc, tp = blk["score"], blk["turning_points"]
        print(f"\n=== WALK-FORWARD, {h}-MONTH HORIZON (n={sc['n']:,} OOS predictions) ===")
        print(f"model RMSE {sc['rmse']:.2f}pp   vs baselines: " +
              "  ".join(f"{k} {v:.2f}" for k, v in sc["baselines"].items()))
        print("skill:      " + "   ".join(f"vs {k} {v:+.1%}" for k, v in sc["skill"].items()))
        print(f"  -> best trivial rule is '{sc['best_baseline']}'; model beats it by "
              f"{sc['skill_vs_best']:+.1%}")
        print(f"direction accuracy: model {sc['dir_acc']:.1%} | naive momentum {sc['dir_acc_naive']:.1%}")
        print(f"cross-sectional rank IC: {sc['rank_ic']:+.3f} (positive {sc['ic_pos']:.0%} of months)")

        print("  -- split by whether the trend actually turned --")
        for lbl, s in tp.items():
            print(f"     {lbl:>16} n={s['n']:>4} ({s['share']:.0%} of months)  "
                  f"skill vs naive {s['skill_vs_naive']:+6.1%}  dir {s['dir_acc']:.0%}")

        f = blk["forecast"]
        print("  -- output --")
        if not f["levels_published"]:
            print(f"     LEVEL FORECAST WITHHELD: {f['levels_reason']}")
        if f["ranking_published"]:
            print(f"     RANKING published ({f['ranking_basis']}), as of {f['asof']}.")
            print(f"     Ordering is the signal; the numbers are a ranking key, NOT forecast growth.")
            for i, r in enumerate(f["rows"], 1):
                print(f"       {i}. {r['industry']:28s} {r['emp']:>8,.0f}k  "
                      f"trailing {h}m {r['naive_mom']:>+6.1f}%")
        else:
            print(f"     RANKING WITHHELD: rank IC {sc['rank_ic']:+.3f} too weak")

    print("\n[honest read: check 'trend_turns' before believing any of it - that is the only "
          "column measuring whether the model sees a change coming, rather than extrapolating.]")


if __name__ == "__main__":
    main()
