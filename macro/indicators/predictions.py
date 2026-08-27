import yfinance as yf
import numpy as np
import pandas as pd
import threading
import time
import math
import os
import json
import hashlib
import tempfile
from functools import wraps, lru_cache
from scipy.optimize import minimize

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # numpy < 1.20
    sliding_window_view = None

from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS
)
from predict import predict_assets, predict_commodities
from macro.paths import model_path

from macro.indicators.cache import *


# =========================
# II. ML Asset Predictions
# =========================
@ttl_cache(30)
def get_ml_sector_prediction():
    # 2026-08-25: replaced the RandomForest sector model (no validated OOS edge,
    # and cross-sectional sector ranking has NEGATIVE IC - sectors share one
    # equity beta) with the SAME adaptive router used in get_trends, scoped to
    # sectors: each sector routed by its own ER to momentum (TREND) or RSI2
    # reversion (CHOP), gated >200DMA, sized inverse-vol. Long-only + cash -
    # there is no "short"; the underweight is simply not held (FLAT -> cash).
    # backtest_gss_sectors_only 2007-26: Sharpe 0.78 vs SPY 0.63, this is the
    # validated per-sector trend/mean-revert tension.
    try:
        from macro.indicators.data import _get_shared_market_data
        from macro.indicators.mathstats import _efficiency_ratio, _trend_stats
        data = _get_shared_market_data()["Close"]
        held = []; flat = []
        for t, name in SECTOR_NAMES.items():
            if t not in data.columns:
                continue
            c = data[t].dropna()
            if len(c) < 210:
                continue
            last = float(c.iloc[-1])
            s50 = float(c.rolling(50).mean().iloc[-1])
            s200 = float(c.rolling(200).mean().iloc[-1])
            slope, r2 = _trend_stats(c, 20, 20)
            rsi2 = float(compute_RSI(c, 2).iloc[-1])
            eff = _efficiency_ratio(c, 20)
            vol63 = float(c.pct_change().rolling(63).std().iloc[-1])
            above200 = last > s200
            sig, state = "FLAT", "MID"
            if eff >= 0.40:
                state = "TREND"
                if above200 and last > s50 and slope > 0:
                    sig = "MOM"
            elif eff <= 0.35:
                state = "CHOP"
                if above200 and rsi2 < 15:
                    sig = "REV"

            # Why is this sector in cash? Surface the reason so a FLAT sector
            # reads as deliberate per-sector defense, not a silent gap. Sectors
            # exit INDIVIDUALLY on their own weakness (which controls drawdown
            # better than one market-wide bear gate - backtested).
            reason = ""
            if sig == "FLAT":
                if not above200:
                    reason = "< 200DMA"                 # below long-term trend: defensive exit
                elif state == "TREND":
                    reason = "trend fading"             # >200DMA but lost 50DMA / slope turned down
                elif state == "CHOP":
                    reason = "choppy · not oversold"    # waiting for an RSI2 dip to buy
                else:
                    reason = "choppy / no trend"        # MID efficiency: no clean signal

            rec = {"ticker": t, "name": name, "state": state, "signal": sig,
                   "reason": reason, "eff_ratio": round(eff, 2),
                   "invvol": (1.0 / vol63) if (vol63 and vol63 > 0) else 0.0}
            (held if sig in ("MOM", "REV") else flat).append(rec)

        # inverse-vol allocation among qualifying sectors, 15%/5% caps,
        # normalized to 100% of the sector sleeve (FLAT sectors = 0%, held in cash)
        tot = sum(h["invvol"] for h in held)
        for h in held:
            raw = (h["invvol"] / tot) if tot > 0 else 0.0
            h["confidence"] = round(min(raw, 0.05 if h["signal"] == "REV" else 0.15) * 100, 1)
        s2 = sum(h["confidence"] for h in held)
        if s2 > 0:
            for h in held:
                h["confidence"] = round(h["confidence"] / s2 * 100, 1)
        for f in flat:
            f["confidence"] = 0.0
        held.sort(key=lambda x: x["confidence"], reverse=True)
        ranked = held + flat
        if not ranked:
            _na = {"name": "N/A", "ticker": "N/A", "confidence": 0, "state": "-", "signal": "FLAT"}
            return {"top": _na, "bottom": _na, "spread": 0, "all": [], "n_held": 0, "n_flat": 0}
        top = ranked[0]
        bottom = ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1),
                "all": ranked, "n_held": len(held), "n_flat": len(flat)}
    except Exception as e:
        print(f"Sector Prediction Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0, "state": "-", "signal": "FLAT"}
        return {"top": _na, "bottom": _na, "spread": 0, "all": [], "n_held": 0, "n_flat": 0}


def _ranker_to_prediction(rotation_list):
    """Convert a validated rotation ranker (sorted best-first, with rank_score
    + ret_126) into the tile's prediction shape. 'confidence' is a display
    ALLOCATION weight: rank_scores shifted so the weakest = 0, normalized to
    100 - a long-only tilt toward the strongest signals (NOT a calibrated
    probability). 2026-08-25: replaces the XGB country/commodity models,
    which had no honest OOS edge (single-split, momentum-signed - wrong for
    commodities). See backtest_rank_predictors / backtest_commodity_reversion.
    """
    valid = [r for r in rotation_list
             if r.get("rank_score") is not None and r["rank_score"] == r["rank_score"]]
    if not valid:
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}
    scores = np.array([r["rank_score"] for r in valid], dtype=float)
    shifted = scores - scores.min()
    total = shifted.sum()
    weights = (shifted / total * 100.0) if total > 0 else np.full(len(valid), 100.0 / len(valid))
    ranked = [
        {"ticker": r["sym"], "name": r["name"], "confidence": round(float(w), 1),
         "ret_126": r.get("ret_126")}
        for r, w in zip(valid, weights)
    ]
    top, bottom = ranked[0], ranked[-1]
    return {"top": top, "bottom": bottom,
            "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}


@ttl_cache(30)
def get_ml_country_prediction():
    # Rerouted to the validated 6-month MOMENTUM ranker (get_country_rotation).
    try:
        from macro.indicators.rotation import get_country_rotation
        return _ranker_to_prediction(get_country_rotation())
    except Exception as e:
        print(f"Country Prediction Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}


@ttl_cache(30)
def get_ml_commodity_prediction():
    # Rerouted to the validated 6-month REVERSION ranker (get_commodity_rotation).
    try:
        from macro.indicators.rotation import get_commodity_rotation
        return _ranker_to_prediction(get_commodity_rotation())
    except Exception as e:
        print(f"Commodity Prediction Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}

__all__ = [
    "get_ml_sector_prediction",
    "get_ml_country_prediction",
    "get_ml_commodity_prediction",
]
