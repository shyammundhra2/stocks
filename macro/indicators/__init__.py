"""macro.indicators - facade re-exporting the split submodules.

Refactored 2026-08 from a single 2120-line module into cohesive submodules
(cache, mathstats, data, regime, predictions, signals, rotation, portfolio)
with NO logic change. Every prior `from macro.indicators import X` and
`macro.indicators.X` access still resolves through the star-imports below.
Dependency layering: cache -> mathstats -> data -> regime -> {predictions,
signals, rotation} -> portfolio.
"""
from macro.indicators.cache import *
from macro.indicators.mathstats import *
from macro.indicators.data import *
from macro.indicators.regime import *
from macro.indicators.predictions import *
from macro.indicators.signals import *
from macro.indicators.rotation import *
from macro.indicators.portfolio import *

# Re-export the header names too, so callers that reach into the module
# namespace (e.g. profile_dashboard uses engine.SECTORS / engine._get_*) work.
from macro.helpers import compute_RSI, compute_ATR
from macro.constants import (
    SECTOR_NAMES, SECTORS, COUNTRIES, COMMODITIES,
    CURRENCIES, TREND_ASSETS, ML_MACRO_TICKERS,
)
from macro.paths import model_path
from predict import predict_assets, predict_commodities
