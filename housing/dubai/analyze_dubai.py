"""
Dubai residential property: cycle anatomy, momentum properties, walk-forward forecast.

Same discipline as housing/predict_housing.py: no in-sample-only claims. Every forecast
here comes from a rule that was walk-forward tested against a random-walk baseline on
this same series, and the test result is reported whether or not it flatters the model.

WHAT THE VALIDATION SAYS (as of 2026-07, 283 monthly obs):
  6m  : +30.5% RMSE reduction vs random walk, 74.0% direction hit-rate  -> usable
  12m : +15.2% vs random walk but only +7.7% vs RW-with-drift, 56.6% hit -> weak
  24m : -38.1%, i.e. WORSE THAN A RANDOM WALK                           -> refused

So ``forecast()`` returns no point estimate at 24m. It is not an oversight; publishing
a number from a rule with negative measured skill is worse than saying nothing.

Two further honesty flags the caller should carry forward:
  * The skill is regime-contaminated. Split by regime, the 12m rule scores +20.6% when
    forecasting from inside a drawdown and -1.7% from an uptrend, because 146 of 175
    forecast dates sit inside the 2014-2020 downtrend. It largely learned "declines
    persist" and had no ability to anticipate the 2026 turn.
  * Every cycle base rate rests on n=2 prior cycles. Monthly granularity does not fix that.
"""
import numpy as np
import pandas as pd

from .build_series import load

HORIZONS = (6, 12, 24)
MIN_TRAIN = 96          # 8y of history before the first walk-forward forecast
MAX_SKILL_HORIZON = 12  # refuse to publish a point forecast beyond this


# ------------------------------------------------------------------ cycle anatomy
def drawdown_episodes(px: pd.Series, min_depth: float = 0.05) -> list[dict]:
    """Peak-to-recovery episodes deeper than ``min_depth``. Last one may be ongoing."""
    dd = px / px.cummax() - 1
    out, in_dd, peak_i = [], False, 0
    for i, d in enumerate(dd.values):
        if not in_dd and d < -0.001:
            in_dd, peak_i = True, i - 1
        elif in_dd and d >= -0.001:
            out.append(_episode(px.iloc[peak_i:i + 1], recovered=True))
            in_dd = False
    if in_dd:
        out.append(_episode(px.iloc[peak_i:], recovered=False))
    return [e for e in out if e["depth"] <= -min_depth]


def _episode(seg: pd.Series, recovered: bool) -> dict:
    trough = seg.idxmin()
    return {
        "peak": str(seg.index[0]), "trough": str(trough),
        "recovered": str(seg.index[-1]) if recovered else None,
        "depth": float(seg.min() / seg.iloc[0] - 1),
        "months_down": int((trough - seg.index[0]).n),
        "months_to_recover": int((seg.index[-1] - trough).n) if recovered else None,
    }


# --------------------------------------------------------------- series properties
def properties(px: pd.Series) -> dict:
    """Persistence of the series - the thing that makes momentum work, and blind at turns."""
    lp = np.log(px)
    r = lp.diff().dropna()
    ac = {f"lag{k}": float(r.autocorr(k)) for k in range(1, 13)}

    trail, fwd = lp.diff(12), lp.diff(12).shift(-12)
    m = trail.notna() & fwd.notna()
    slope, _ = np.polyfit(trail[m], fwd[m], 1)
    return {
        "acf": ac,
        "ac1": ac["lag1"],
        "mom_corr": float(np.corrcoef(trail[m], fwd[m])[0, 1]),
        "mom_slope": float(slope),   # <1 means momentum decays rather than compounds
        "n_pairs": int(m.sum()),
    }


def deceleration_test(px: pd.Series, dd_floor: float = -0.03, fwd: int = 6) -> dict:
    """Does a run of shrinking monthly declines mark a bottom?

    The market read of the 2026 drawdown is that decelerating falls signal stabilization.
    This tests that claim on Dubai's own record: inside a drawdown, flag every month whose
    last three changes were negative and shrinking, then compare what came next.
    """
    lp = np.log(px)
    r = lp.diff()
    dd = px / px.cummax() - 1

    rows = []
    for t in range(6, len(px) - fwd):
        if dd.iloc[t] > dd_floor:
            continue
        a, b, c = r.iloc[t - 2], r.iloc[t - 1], r.iloc[t]
        rows.append((bool(c > b > a and c < 0), float(lp.iloc[t + fwd] - lp.iloc[t])))
    d = pd.DataFrame(rows, columns=["decel", "fwd"])

    def _agg(sub):
        return {"n": int(len(sub)), "mean_fwd": float(sub.fwd.mean()),
                "pct_positive": float((sub.fwd > 0).mean())}

    return {"horizon": fwd, "n_tested": int(len(d)),
            "decelerating": _agg(d[d.decel]), "other": _agg(d[~d.decel])}


# ------------------------------------------------------------- walk-forward validation
def _fit_decay(lp: pd.Series, h: int):
    """Momentum-decay rule: forward h-month log return ~ a + b * trailing h-month return."""
    trail, fwd = lp.diff(h), lp.diff(h).shift(-h)
    m = trail.notna() & fwd.notna()
    if m.sum() < 24:
        return None
    b, a = np.polyfit(trail[m], fwd[m], 1)
    return float(b), float(a)


def walk_forward(px: pd.Series, h: int) -> pd.DataFrame:
    """Expanding-window out-of-sample forecasts. Coefficients refit on past data only."""
    lp = np.log(px)
    rows = []
    for t in range(MIN_TRAIN, len(px) - h):
        lh = lp.iloc[:t + 1]
        fit = _fit_decay(lh, h)
        if fit is None:
            continue
        b, a = fit
        rows.append({
            "date": px.index[t],
            "pred": a + b * (lh.iloc[-1] - lh.iloc[-1 - h]),
            "actual": float(lp.iloc[t + h] - lp.iloc[t]),
            "rw": 0.0,                              # random walk, no drift
            "drift": float(lh.diff().mean() * h),   # random walk with historical drift
            "dd": float(px.iloc[t] / px.iloc[:t + 1].max() - 1),
        })
    return pd.DataFrame(rows)


def validate(px: pd.Series) -> dict:
    """Walk-forward skill per horizon, plus the regime split that qualifies it."""
    out = {}
    for h in HORIZONS:
        wf = walk_forward(px, h)
        if wf.empty:
            continue
        rmse = lambda c: float(np.sqrt(((wf[c] - wf.actual) ** 2).mean()))  # noqa: E731
        m, rw, dr = rmse("pred"), rmse("rw"), rmse("drift")

        regime = {}
        for lbl, mask in [("uptrend", wf.dd > -0.02), ("drawdown", wf.dd <= -0.02)]:
            if mask.sum() > 10:
                sub = wf[mask]
                sm = np.sqrt(((sub.pred - sub.actual) ** 2).mean())
                sr = np.sqrt(((sub.rw - sub.actual) ** 2).mean())
                regime[lbl] = {"n": int(mask.sum()), "skill": float(1 - sm / sr)}

        out[h] = {"n": int(len(wf)), "rmse": m, "rmse_rw": rw, "rmse_drift": dr,
                  "skill_vs_rw": float(1 - m / rw), "skill_vs_drift": float(1 - m / dr),
                  "hit_rate": float((np.sign(wf.pred) == np.sign(wf.actual)).mean()),
                  "resid_sd": float((wf.pred - wf.actual).std()),
                  "regime": regime,
                  "usable": bool(1 - m / rw > 0 and h <= MAX_SKILL_HORIZON)}
    return out


# ------------------------------------------------------------------------- forecast
def forecast(px: pd.Series, val: dict) -> dict:
    """Point forecasts, but only at horizons where measured skill justifies one."""
    lp = np.log(px)
    level = float(px.iloc[-1])
    out = {}
    for h in HORIZONS:
        v = val.get(h)
        if v is None:
            continue
        if not v["usable"]:
            out[h] = {"withheld": True, "reason":
                      f"walk-forward skill {v['skill_vs_rw']:+.1%} vs random walk"
                      f"{' (negative)' if v['skill_vs_rw'] < 0 else ''}"
                      f" at {h}m - no point estimate published",
                      "skill_vs_rw": v["skill_vs_rw"]}
            continue
        b, a = _fit_decay(lp, h)
        pt = a + b * (lp.iloc[-1] - lp.iloc[-1 - h])
        sd = v["resid_sd"]                       # interval from realised OOS errors
        out[h] = {"withheld": False, "to": str(px.index[-1] + h),
                  "point": float(np.expm1(pt)),
                  "lo80": float(np.expm1(pt - 1.28 * sd)),
                  "hi80": float(np.expm1(pt + 1.28 * sd)),
                  "level": level * float(np.exp(pt)),
                  "level_lo80": level * float(np.exp(pt - 1.28 * sd)),
                  "level_hi80": level * float(np.exp(pt + 1.28 * sd)),
                  "trailing": float(np.expm1(lp.iloc[-1] - lp.iloc[-1 - h])),
                  "skill_vs_rw": v["skill_vs_rw"], "hit_rate": v["hit_rate"]}
    return out


def compute_dubai_outlook(refresh: bool = False) -> dict:
    """Everything the report needs, as one JSON-safe dict."""
    df = load(refresh=refresh)
    px = df["aed_sqm"]
    val = validate(px)
    eps = drawdown_episodes(px)
    closed = [e for e in eps if e["recovered"]and e["depth"] < -0.10]

    return {
        "asof": str(px.index[-1]),
        "level": float(px.iloc[-1]),
        "peak": str(px.idxmax()), "peak_level": float(px.max()),
        "drawdown": float(px.iloc[-1] / px.max() - 1),
        "months_since_peak": int((px.index[-1] - px.idxmax()).n),
        "pct_months_below_peak": float((px / px.cummax() - 1 < -0.001).mean()),
        "episodes": eps,
        "base_rate": {
            "n_major_cycles": len(closed),
            "mean_depth": float(np.mean([e["depth"] for e in closed])) if closed else None,
            "mean_months_down": float(np.mean([e["months_down"] for e in closed])) if closed else None,
            "mean_months_to_recover": float(np.mean([e["months_to_recover"] for e in closed])) if closed else None,
        },
        "properties": properties(px),
        "deceleration_test": deceleration_test(px),
        "validation": val,
        "forecast": forecast(px, val),
        "n_obs": int(len(px)),
        "n_spliced": int((df.source != "BIS").sum()),
    }


def main():
    d = compute_dubai_outlook()
    px = load()["aed_sqm"]

    print(f"=== DUBAI RESIDENTIAL, as of {d['asof']} ({d['n_obs']} monthly obs, "
          f"{d['n_spliced']} spliced) ===")
    print(f"peak {d['peak']} {d['peak_level']:,.0f} -> now {d['level']:,.0f} AED/sqm "
          f"({d['drawdown']:+.1%} over {d['months_since_peak']}m)")
    print(f"the series has spent {d['pct_months_below_peak']:.0%} of its life below a prior peak")

    print("\n--- drawdown episodes >5% ---")
    print(f"{'peak':>9} {'trough':>9} {'recovered':>10} {'depth':>8} {'down':>6} {'recov':>7}")
    for e in d["episodes"]:
        print(f"{e['peak']:>9} {e['trough']:>9} {str(e['recovered'] or 'ONGOING'):>10} "
              f"{e['depth']:>7.1%} {e['months_down']:>5}m "
              f"{(str(e['months_to_recover'])+'m') if e['months_to_recover'] else '-':>7}")
    b = d["base_rate"]
    print(f"\nbase rate over {b['n_major_cycles']} completed major cycles: "
          f"depth {b['mean_depth']:.1%}, {b['mean_months_down']:.0f}m down, "
          f"{b['mean_months_to_recover']:.0f}m back to the old peak")

    p = d["properties"]
    print(f"\n--- persistence ---\nlag-1 autocorr {p['ac1']:+.2f} | trailing-12m vs forward-12m "
          f"corr {p['mom_corr']:+.2f}, slope {p['mom_slope']:+.2f} (<1 = momentum decays)")

    t = d["deceleration_test"]
    print(f"\n--- does decelerating decline mark a bottom? (n={t['n_tested']} months "
          f"in drawdown, fwd {t['horizon']}m) ---")
    for k in ("decelerating", "other"):
        s = t[k]
        print(f"  {k:>13}: n={s['n']:>3}  mean fwd {s['mean_fwd']:+.2%}  "
              f"positive {s['pct_positive']:.0%}")
    print("  -> decelerating declines have been a WORSE sign than non-decelerating ones.")

    print("\n--- walk-forward validation (expanding window, no lookahead) ---")
    for h, v in d["validation"].items():
        print(f"{h:>3}m  n={v['n']:>3}  skill vs RW {v['skill_vs_rw']:+6.1%}  "
              f"vs RW+drift {v['skill_vs_drift']:+6.1%}  hit {v['hit_rate']:.1%}"
              f"   {'USABLE' if v['usable'] else 'NOT USABLE'}")
        for lbl, r in v["regime"].items():
            print(f"       {lbl:>8} n={r['n']:>3}  skill {r['skill']:+.1%}")

    print("\n--- forecast ---")
    for h, f in d["forecast"].items():
        if f["withheld"]:
            print(f"{h:>3}m  WITHHELD - {f['reason']}")
        else:
            print(f"{h:>3}m  to {f['to']}: {f['point']:+.1%} "
                  f"[80% {f['lo80']:+.1%}, {f['hi80']:+.1%}] -> {f['level']:,.0f} AED/sqm")
    print("\n[honest read: trust the 6m number; beyond a year use the cycle base rate, not "
          "this model. n=2 cycles, and the model's skill comes mostly from one long downtrend.]")


if __name__ == "__main__":
    main()
