"""
Deterministic change-log digest for the Macro Cockpit.

NOT a newsletter and NOT an LLM. It snapshots the key DETERMINISTIC signals we
already compute (regime, RoRo, credit, curve, housing, financial conditions),
keeps a rolling ~daily history, and diffs the current reading against the
snapshot ~1 WEEK ago AND ~1 MONTH ago. Writes a TEMPLATED prose digest: what
moved (short- and medium-term), what crossed a threshold, and the 2-3 things
that matter.

Pure fill-in-the-blanks from numbers - zero hallucination (that's why this is
safe where an LLM news tab wasn't). Lazy: only runs when the digest tab opens.
"""
import os
import re
import json
import time
import datetime

from macrocockpit.macro_board import _STORE_DIR

_HIST = os.path.join(_STORE_DIR, "digest_history.json")   # rolling ~daily snapshots
_DAY = 24 * 3600


def _num(s):
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(s))
    return float(m.group()) if m else None


def _gather():
    """Flatten the deterministic signals into {name: value} + {name: read-tag}."""
    sig, read = {}, {}
    try:
        from macrocockpit.macro_board import get_macro_dashboard
        md = get_macro_dashboard()
        sig["RoRo"] = md.get("roro_score")
        read["Regime"] = md.get("regime")
        for _cat, rows in md.get("categories", []):
            for name, val, tag in rows:
                n = _num(val)
                if n is not None:
                    sig[name] = n
                if tag:
                    read[name] = tag
    except Exception as e:
        print(f"digest macro gather error: {e}")
    try:
        from macro.indicators import get_risk_regime
        reg = get_risk_regime()
        sig["Regime confidence"] = reg.get("confidence")
        read["Regime"] = reg.get("status")
    except Exception as e:
        print(f"digest regime gather error: {e}")
    try:
        from macrocockpit.cycles import get_cycles
        re_ = get_cycles().get("realestate", {})
        for k, dst in [("supply", "Months supply"), ("supply_implied_hpa", "Implied fwd-12m HPA"),
                       ("momentum", "Case-Shiller momentum")]:
            if re_.get(k) is not None:
                sig[dst] = re_[k]
        if re_.get("supply_regime"):
            read["Housing"] = re_["supply_regime"]
        if re_.get("lean"):
            read["Housing lean"] = re_["lean"]
    except Exception as e:
        print(f"digest cycles gather error: {e}")
    return {"ts": time.time(), "date": str(datetime.date.today()),
            "signals": sig, "reads": read}


def _load_hist():
    try:
        with open(_HIST) as f:
            return json.load(f)
    except Exception:
        return []


def _closest(hist, target_ts, tol_days=4):
    """Snapshot nearest target_ts, within tol_days; else None (not enough history)."""
    best, bestdiff = None, tol_days * _DAY
    for s in hist:
        diff = abs(s.get("ts", 0) - target_ts)
        if diff <= bestdiff:
            best, bestdiff = s, diff
    return best


def _dir(d):
    return "↑" if d > 0 else "↓"


# Direction of goodness: does a HIGHER value = bullish (+1) or bearish (-1)?
# Used to color a delta green/red. Signals not listed -> neutral (yellow).
_BULL_DIR = {
    "RoRo": 1, "Regime confidence": 1, "Case-Shiller momentum": 1,
    "Implied fwd-12m HPA": 1, "Retail Sales (YoY)": 1, "Consumer Confidence": 1,
    "Mfg survey (PMI proxy)": 1, "Real GDP (YoY)": 1, "Job Openings (JOLTS)": 1,
    "10Y-2Y Spread": 1,
    "VIX": -1, "MOVE (bond vol)": -1, "HY OAS (norm 4.5)": -1,
    "Fin Conditions (NFCI)": -1, "Mortgage Spread (norm 1.80)": -1,
    "5y Inflation Expectations": -1, "Fed Funds": -1, "USD Index (broad)": -1,
    "Unemployment Rate": -1, "Months supply": -1, "CPI (YoY)": -1,
    "Core PCE (YoY, Fed target)": -1, "Initial Claims (4wk avg)": -1,
}


def _changes(cur, base, n=6):
    """Material moves between two snapshots, each tagged bull/bear/neutral."""
    if not base:
        return []
    b = base["signals"]
    out = []
    for k, v in cur["signals"].items():
        if k in b and b[k] is not None and v is not None:
            d = v - b[k]
            rel = abs(d) / (abs(b[k]) + 0.01)
            if rel > 0.02 and round(v, 2) != round(b[k], 2):
                signed = d * _BULL_DIR.get(k, 0)
                sent = "bull" if signed > 0 else "bear" if signed < 0 else "neutral"
                out.append((rel, {"text": f"{k}: {b[k]:g} {_dir(d)} {v:g}", "sentiment": sent}))
    out.sort(key=lambda x: -x[0])
    return [t for _r, t in out[:n]]


def _build(cur, wk, mo):
    sig, read = cur["signals"], cur["reads"]
    conf, roro = sig.get("Regime confidence"), sig.get("RoRo")
    head = f"Regime: {read.get('Regime','?')}"
    if conf is not None:
        head += f" (confidence {conf:.0f}%" + (f", RoRo {roro:.0f})" if roro is not None else ")")

    def flag(text, sent):
        return {"text": text, "sentiment": sent}

    flags, matters = [], []
    # regime flip vs a week ago = headline change (to RISK-ON = bull, else bear)
    if wk and wk["reads"].get("Regime") and wk["reads"]["Regime"] != read.get("Regime"):
        matters.append(flag(f"Regime FLIPPED {wk['reads']['Regime']} → {read.get('Regime')} this week.",
                            "bull" if read.get("Regime") == "RISK-ON" else "bear"))
    # read/tag changes vs a week ago (informational -> neutral)
    if wk:
        for k, tag in read.items():
            bt = wk["reads"].get(k)
            if bt and bt != tag:
                flags.append(flag(f"{k}: “{bt}” → “{tag}”", "neutral"))
    # current extreme flags (from the read tags we already compute)
    # Credit extremes are MEAN-REVERTING watch-flags, not directional calls: tight
    # = bullish-now / bearish-forward, wide = bearish-now / bullish-forward. Both
    # are genuinely mixed -> yellow (caution), so they don't contradict the delta
    # coloring (a tightening MOVE stays green = risk-on).
    hy = read.get("HY OAS (norm 4.5)", "")
    if "Tight" in hy:
        flags.append(flag("Credit TIGHT vs norm — complacent/stretched, mean-reverts wider (watch; cheap tail-hedge window).", "neutral"))
    elif "Wide" in hy or "Blown" in hy:
        flags.append(flag("Credit WIDE — stress priced; mean-reverts tighter, equity historically cheap ahead (watch).", "neutral"))
    imp = sig.get("Implied fwd-12m HPA")
    if imp is not None and imp < 0:
        flags.append(flag(f"Housing: {read.get('Housing','elevated supply')} → implied {imp:+.1f}% forward HPA (softening).", "bear"))
    if "Inverted" in read.get("10Y-2Y Spread", ""):
        flags.append(flag("Yield curve INVERTED — recession risk elevated (slow signal).", "bear"))

    matters += flags[:3]
    if not matters:
        matters = [flag("No material changes or threshold crossings this period.", "neutral")]
    return {"headline": head,
            "changed_week": _changes(cur, wk), "changed_month": _changes(cur, mo),
            "flags": flags, "matters": matters[:3]}


def get_digest():
    hist = _load_hist()
    cur = _gather()
    # append at most ~daily, so week/month lookups are clean
    if not hist or (cur["ts"] - hist[-1].get("ts", 0) > 0.9 * _DAY):
        hist.append(cur)
    else:
        hist[-1] = cur
    hist = [s for s in hist if cur["ts"] - s.get("ts", 0) <= 40 * _DAY]  # keep ~40d
    try:
        with open(_HIST + ".tmp", "w") as f:
            json.dump(hist, f)
        os.replace(_HIST + ".tmp", _HIST)
    except Exception as e:
        print(f"digest persist error: {e}")

    wk = _closest(hist, cur["ts"] - 7 * _DAY)
    mo = _closest(hist, cur["ts"] - 30 * _DAY, tol_days=7)
    out = _build(cur, wk, mo)
    out.update({"date": cur["date"],
                "week_date": wk["date"] if wk else None,
                "month_date": mo["date"] if mo else None})
    return out


if __name__ == "__main__":
    d = get_digest()
    print(f"MACRO DIGEST — {d['date']}\n{d['headline']}\n")
    print(f"Changes vs 1 week ({d['week_date']}):");  [print(f"  • [{c['sentiment']:7s}] {c['text']}") for c in d["changed_week"]] or print("  (building history)")
    print(f"\nChanges vs 1 month ({d['month_date']}):"); [print(f"  • [{c['sentiment']:7s}] {c['text']}") for c in d["changed_month"]] or print("  (building history)")
    print("\nFlags:"); [print(f"  • [{f['sentiment']:7s}] {f['text']}") for f in d["flags"]] or print("  (none)")
    print("\nWhat matters:"); [print(f"  {i}. [{m['sentiment']:7s}] {m['text']}") for i, m in enumerate(d["matters"], 1)]
