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
    try:
        tickers = list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS
        res = predict_assets(model_path("sector_model.joblib"), tickers, SECTOR_NAMES, "sector")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": SECTOR_NAMES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Sector Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}


@ttl_cache(30)
def get_ml_country_prediction():
    try:
        tickers = list(COUNTRIES.keys()) + ML_MACRO_TICKERS + ['SPY']
        res = predict_assets(model_path("country_model.joblib"), tickers, COUNTRIES, "country")
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COUNTRIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Country Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}


@ttl_cache(30)
def get_ml_commodity_prediction():
    try:
        res = predict_commodities(
            sector_model_path=model_path("commodity_sector_model.joblib"),
            commodity_model_path=model_path("commodity_model.joblib"),
            friendly_names=COMMODITIES,
            use_cache=True,
            top_n_sectors=5,
            top_n_per_sector=5
        )
        probs = res["probabilities"]
        ranked = [
            {"ticker": k, "name": COMMODITIES.get(k, k), "confidence": round(v * 100, 1)}
            for k, v in probs.items()
        ]
        top, bottom = ranked[0], ranked[-1]
        return {"top": top, "bottom": bottom,
                "spread": round(top["confidence"] - bottom["confidence"], 1), "all": ranked}
    except Exception as e:
        print(f"ML Commodity Error: {e}")
        _na = {"name": "N/A", "ticker": "N/A", "confidence": 0}
        return {"top": _na, "bottom": _na, "spread": 0, "all": []}

__all__ = [
    "get_ml_sector_prediction",
    "get_ml_country_prediction",
    "get_ml_commodity_prediction",
]
