"""Book overlays added 2026-11-26 on branch gss-edge-research (research/gss_edge/REPORT.md).

PURE functions (no I/O) so the live book and the historical replay share one implementation.

1. Bond haven while risk-off: when the SPY-200DMA gate is OFF, idle cash (up to HAVEN_FRAC of the
   book) is parked in IEF or SHY - whichever has the better trailing 126d return - but only if
   that return beats BIL's. Otherwise stays in cash. Never displaces positions the optimizer holds
   (DEFENSIVE_ASSETS stay in the book through the gate), it only fills idle cash.
2. Capitulation reserve folded into the book: RESERVE_FRAC (20%) of the book is carved out of the
   optimizer's allocation (core weights are scaled by 1-RESERVE_FRAC) and held as an SPY position
   sized by get_vix_signal's ladder (target $ / book value). Previously the reserve was a separate
   signal that sat beside a full-size optimizer book.
"""
import numpy as np
import pandas as pd

RESERVE_FRAC = 0.20
HAVEN_FRAC = 0.50
HAVEN_LOOKBACK = 126
PORTFOLIO_VALUE = 500_000
OVERLAYS_ENABLED = False      # OFF 2026-11-26: failed the DD rule at full size on the live-optimizer replay
                              # (research/gss_edge/REPORT.md). Re-enable only after choosing RESERVE_FRAC/HAVEN_FRAC.
                              # tests/replays flip this to get the raw (pre-overlay) optimizer book


def _trail_ret(close, n=HAVEN_LOOKBACK):
    c = pd.Series(close).dropna()
    if len(c) <= n:
        return np.nan
    return float(c.iloc[-1] / c.iloc[-1 - n] - 1.0)


def pick_haven(ief_close, shy_close, bil_close):
    """'IEF' / 'SHY' / None. Best trailing-126d return of the two, only if it beats BIL's."""
    bil = _trail_ret(bil_close)
    cands = {"IEF": _trail_ret(ief_close), "SHY": _trail_ret(shy_close)}
    ok = {k: v for k, v in cands.items() if np.isfinite(v) and np.isfinite(bil) and v > bil}
    return max(ok, key=ok.get) if ok else None


def book_overlay(core_weights, gate_riskoff, reserve_target_dollars, haven, reserve_on=True):
    """core_weights: {sym: fraction of book} from the optimizer (pre-scale).
    reserve_on=False turns off the carve-out + SPY reserve (research ablations only).
    Returns dict(core_scale, core, reserve_spy, haven_sym, haven_w, weights) - all fractions of book."""
    scale = (1.0 - RESERVE_FRAC) if reserve_on else 1.0
    core = {s: w * scale for s, w in core_weights.items() if w > 0}
    res = float(np.clip(reserve_target_dollars / PORTFOLIO_VALUE, 0.0, RESERVE_FRAC)) if reserve_on else 0.0
    haven_w = 0.0
    if gate_riskoff and haven:
        haven_w = min(HAVEN_FRAC, max(1.0 - sum(core.values()) - res, 0.0))
    weights = dict(core)
    if res > 0:
        weights["SPY"] = weights.get("SPY", 0.0) + res
    if haven_w > 0:
        weights[haven] = weights.get(haven, 0.0) + haven_w
    return {"core_scale": scale, "core": core, "reserve_spy": res,
            "haven_sym": haven if haven_w > 0 else None, "haven_w": haven_w, "weights": weights}

__all__ = ["RESERVE_FRAC", "HAVEN_FRAC", "HAVEN_LOOKBACK", "PORTFOLIO_VALUE", "OVERLAYS_ENABLED",
           "pick_haven", "book_overlay"]
