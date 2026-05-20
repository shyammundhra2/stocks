import yfinance as yf
import pandas as pd
import joblib
import argparse
import warnings
from datetime import timedelta, datetime
from functools import lru_cache
from macro.constants import SECTOR_NAMES, COMMODITIES, COUNTRIES, CURRENCIES, ML_MACRO_TICKERS, TREND_ASSETS
from macro.helpers import compute_RSI, compute_ATR
import numpy as np

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=ResourceWarning)

# ----------------- Shared Data Cache -----------------
_PREDICT_CACHE = {}
_CACHE_LOCK = None


def _get_cache_lock():
    """Lazy init for thread lock to avoid import issues"""
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        import threading
        _CACHE_LOCK = threading.Lock()
    return _CACHE_LOCK


def _get_shared_predict_data(tickers, start_date, end_date, cache_key=None):
    """
    Shared data fetcher for predict_assets calls.
    Uses a simple cache with date-based key to avoid redundant downloads.
    """
    import time

    # Create cache key based on tickers and date range
    if cache_key is None:
        cache_key = (tuple(sorted(tickers)), start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))

    lock = _get_cache_lock()
    now = time.time()

    with lock:
        if cache_key in _PREDICT_CACHE:
            data, timestamp = _PREDICT_CACHE[cache_key]
            # Cache for 30 seconds
            if now - timestamp < 30:
                return data
            # Cleanup old entries
            expired = [k for k, (_, ts) in _PREDICT_CACHE.items() if now - ts >= 30]
            for k in expired:
                del _PREDICT_CACHE[k]

    # Download outside lock - use auto_adjust=False and group_by='column'
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False,
                           auto_adjust=False, group_by='column')

    # Return raw data with MultiIndex intact
    with lock:
        _PREDICT_CACHE[cache_key] = (raw_data, now)

    return raw_data


# ----------------- Helper Functions -----------------
@lru_cache(maxsize=32)
def _get_model_bundle(model_path):
    """Cache model loading to avoid repeated disk reads"""
    return joblib.load(model_path)


# ----------------- Feature Computation -----------------

def compute_risk_features(data):
    """Compute features specific to risk models"""
    horizon_q = 63
    horizon_1m = 21
    X = pd.DataFrame(index=data.index)
    cols = data.columns

    data = data.ffill().bfill()

    if 'DX-Y.NYB' in cols:
        X['DXY_mom'] = data['DX-Y.NYB'].pct_change(horizon_q)
        X['DXY_Mom'] = data['DX-Y.NYB'].pct_change(horizon_1m)
    if '^VIX' in cols:
        vix_rolling = data['^VIX'].rolling(21)
        X['VIX_level'] = vix_rolling.mean()
        X['VIX_Level'] = data['^VIX']
    if '^MOVE' in cols:
        X['MOVE_level'] = data['^MOVE']
        X['MOVE_Level'] = data['^MOVE']
    if '^TYX' in cols and '^TNX' in cols:
        X['Yield_Curve'] = data['^TYX'] - data['^TNX']
        X['TNX_vol'] = data['^TNX'].rolling(horizon_q).std()
    if 'LQD' in cols and 'HYG' in cols:
        X['Credit_Spread'] = data['LQD'] / data['HYG']
        X['Credit_Spread_Proxy'] = X['Credit_Spread']

    # Risk-specific calculations
    if 'RSP' in cols and 'SPY' in cols:
        breadth_ratio = data['RSP'] / data['SPY']
        X['Breadth_Ratio'] = breadth_ratio
        X['Breadth_MA_Diff'] = breadth_ratio - breadth_ratio.rolling(50).mean()
        X['SPY_Trend'] = data['SPY'] / data['SPY'].rolling(200).mean()

    required_rot = ['XLU', 'XLP', 'XLK', 'XLY']
    if all(s in cols for s in required_rot):
        X['Defensive_Rotation'] = (data['XLU'] + data['XLP']) / (data['XLK'] + data['XLY'])

    if 'XLF' in cols and 'SPY' in cols:
        X['XLF_Relative_Strength'] = data['XLF'] / data['SPY']

    # Vectorized momentum calculations
    sectors = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    available_sectors = [s for s in sectors if s in cols]
    if available_sectors:
        sector_data = data[available_sectors]
        sector_mom = sector_data.pct_change(horizon_q)
        for s in available_sectors:
            X[f'{s}_3M_Mom'] = sector_mom[s]

    if 'Yield_Curve' in X.columns and 'Defensive_Rotation' in X.columns:
        def_rot_ma = X['Defensive_Rotation'].rolling(126).mean()
        X['Labor_Stress_Proxy'] = ((X['Yield_Curve'] < 0.1) &
                                   (X['Defensive_Rotation'] > def_rot_ma)).astype(int)

    return X


def compute_sector_features(df):
    """
    Compute sector features matching the 63.86% production model.
    Optimized to avoid DataFrame fragmentation warnings.

    Args:
        df (pd.DataFrame): Historical Close prices for sectors + SPY + QQQ + macro tickers

    Returns:
        X (pd.DataFrame): Feature matrix for sector model
    """
    SECTORS = ['XLK', 'XLF', 'XLI', 'XLY', 'XLE', 'XLV', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']

    # Collect all features in a dict, then create DataFrame once
    features = {}

    # ========== SECTOR POSITIONING ==========
    for s in SECTORS:
        if s not in df.columns:
            continue

        # Multi-timeframe returns and ranks
        for period in [10, 21, 42, 63, 126]:
            ret = df[s].pct_change(period, fill_method=None)
            features[f'{s}_Ret_{period}d'] = ret

            # Rank vs other sectors
            available_sectors = [sec for sec in SECTORS if sec in df.columns]
            if len(available_sectors) > 1:
                sector_rets = df[available_sectors].pct_change(period, fill_method=None)
                features[f'{s}_Rank_{period}d'] = sector_rets.rank(axis=1, pct=True)[s]

        # Moving average positioning
        ma_20 = df[s].rolling(20).mean()
        ma_50 = df[s].rolling(50).mean()
        ma_200 = df[s].rolling(200).mean()

        features[f'{s}_MA20'] = (df[s] - ma_20) / ma_20
        features[f'{s}_MA50'] = (df[s] - ma_50) / ma_50
        features[f'{s}_MA200'] = (df[s] - ma_200) / ma_200
        features[f'{s}_MA_Slope'] = (ma_20 - ma_200) / ma_200

        # Volatility metrics
        vol_21 = df[s].pct_change(fill_method=None).rolling(21).std()
        vol_63 = df[s].pct_change(fill_method=None).rolling(63).std()
        features[f'{s}_Vol_21d'] = vol_21
        features[f'{s}_Vol_Ratio'] = vol_21 / (vol_63 + 1e-6)

        # Risk-adjusted returns (Sharpe ratio)
        features[f'{s}_Sharpe_63d'] = (df[s].pct_change(63, fill_method=None) / (vol_63 * np.sqrt(63) + 1e-6))

    # ========== ROTATION SIGNALS ==========
    available_sectors = [s for s in SECTORS if s in df.columns]
    if len(available_sectors) > 1:
        sector_mom_21 = df[available_sectors].pct_change(21, fill_method=None)
        sector_mom_63 = df[available_sectors].pct_change(63, fill_method=None)

        for s in available_sectors:
            # Rank improvement
            rank_21 = sector_mom_21.rank(axis=1, pct=True)[s]
            rank_63 = sector_mom_63.rank(axis=1, pct=True)[s]
            features[f'{s}_Rank_Improvement'] = rank_21 - rank_63

            # Distance from sector median
            median_ret = sector_mom_63.median(axis=1)
            features[f'{s}_vs_Median_63d'] = sector_mom_63[s] - median_ret

    # ========== MACRO CONDITIONS ==========
    if '^VIX' in df.columns:
        vix_ma63 = df['^VIX'].rolling(63).mean()
        features['VIX'] = df['^VIX']
        features['VIX_MA63'] = vix_ma63
        features['VIX_Delta'] = df['^VIX'] - vix_ma63

    if '^MOVE' in df.columns:
        features['MOVE'] = df['^MOVE']
        features['MOVE_MA63'] = df['^MOVE'].rolling(63).mean()

    if '^TNX' in df.columns and '^TYX' in df.columns:
        yc = df['^TYX'] - df['^TNX']
        yc_ma63 = yc.rolling(63).mean()
        features['TNX'] = df['^TNX']
        features['TYX'] = df['^TYX']
        features['YieldCurve'] = yc
        features['YC_MA63'] = yc_ma63
        features['YC_Slope'] = yc - yc_ma63

    if 'HYG' in df.columns and 'LQD' in df.columns:
        hyg_ret = df['HYG'].pct_change(63, fill_method=None)
        lqd_ret = df['LQD'].pct_change(63, fill_method=None)
        features['HYG_Ret_63d'] = hyg_ret
        features['LQD_Ret_63d'] = lqd_ret
        features['Credit_Spread'] = lqd_ret - hyg_ret

    if 'DX-Y.NYB' in df.columns:
        features['DXY_Ret_63d'] = df['DX-Y.NYB'].pct_change(63, fill_method=None)

    if 'GLD' in df.columns:
        features['Gold_Ret_63d'] = df['GLD'].pct_change(63, fill_method=None)

    if 'USO' in df.columns:
        features['Oil_Ret_63d'] = df['USO'].pct_change(63, fill_method=None)

    # Market regime
    if 'SPY' in df.columns:
        spy_ma_50 = df['SPY'].rolling(50).mean()
        spy_ma_200 = df['SPY'].rolling(200).mean()
        features['SPY_Trend'] = (spy_ma_50 - spy_ma_200) / spy_ma_200
        features['SPY_vs_MA200'] = (df['SPY'] - spy_ma_200) / spy_ma_200
        features['SPY_Ret_21d'] = df['SPY'].pct_change(21, fill_method=None)
        features['SPY_Ret_63d'] = df['SPY'].pct_change(63, fill_method=None)

    if 'QQQ' in df.columns and 'SPY' in df.columns:
        qqq_ret = df['QQQ'].pct_change(63, fill_method=None)
        spy_ret = features.get('SPY_Ret_63d', df['SPY'].pct_change(63, fill_method=None))
        features['QQQ_Ret_63d'] = qqq_ret
        features['Tech_Premium'] = qqq_ret - spy_ret

    # Sector style factors
    defensive = ['XLP', 'XLU', 'XLV']
    cyclical = ['XLY', 'XLI', 'XLK']
    value = ['XLE', 'XLF', 'XLI']

    defensive_avail = [s for s in defensive if s in df.columns]
    cyclical_avail = [s for s in cyclical if s in df.columns]
    value_avail = [s for s in value if s in df.columns]

    if len(defensive_avail) > 0 and len(cyclical_avail) > 0:
        defensive_ret = df[defensive_avail].mean(axis=1).pct_change(63, fill_method=None)
        cyclical_ret = df[cyclical_avail].mean(axis=1).pct_change(63, fill_method=None)
        features['Cyclical_vs_Def'] = cyclical_ret - defensive_ret

    if len(value_avail) > 0 and 'XLK' in df.columns:
        value_ret = df[value_avail].mean(axis=1).pct_change(63, fill_method=None)
        growth_ret = df['XLK'].pct_change(63, fill_method=None)
        features['Growth_vs_Value'] = growth_ret - value_ret

    # Create DataFrame once from dict
    X = pd.DataFrame(features, index=df.index)
    return X.fillna(0)


def compute_commodity_features(data):
    """
    Compute enhanced commodity features with cross-sectional analysis.
    Compatible with hierarchical sector→commodity prediction while maintaining
    the same output format for UI.
    """
    cols = data.columns
    data = data.ffill().bfill()

    # Define commodities and sectors (matching training script)
    COMMODITY_TICKERS = list(COMMODITIES.keys())

    SECTOR_MAP = {
        'GC=F': 'Precious_Metals',
        'SI=F': 'Precious_Metals',
        'PL=F': 'Precious_Metals',
        'PA=F': 'Precious_Metals',
        'HG=F': 'Industrial_Metals',
        'CL=F': 'Energy',
        'NG=F': 'Energy',
        'RB=F': 'Energy',
        'ZC=F': 'Grains',
        'ZS=F': 'Grains',
        'ZW=F': 'Grains',
        'KC=F': 'Softs',
        'SB=F': 'Softs',
        'CT=F': 'Softs',
        'CC=F': 'Softs',
    }

    commodity_tickers = [c for c in COMMODITY_TICKERS if c in cols]

    # Collect all features in a dictionary (avoids fragmentation)
    features = {}

    # ========== MACRO FEATURES ==========

    if 'DX-Y.NYB' in cols:
        dxy = data['DX-Y.NYB']
        dxy_ma_63 = dxy.rolling(63, min_periods=1).mean()
        dxy_ma_252 = dxy.rolling(252, min_periods=1).mean()

        features['DXY_mom_3m'] = dxy.pct_change(63)
        features['DXY_mom_1m'] = dxy.pct_change(21)
        features['DXY_trend'] = dxy_ma_63 / (dxy_ma_252 + 1e-10)

    if '^VIX' in cols:
        vix = data['^VIX']
        vix_ma_252 = vix.rolling(252, min_periods=1).mean()

        features['VIX_level'] = vix.rolling(21, min_periods=1).mean()
        features['VIX_change'] = vix.pct_change(21)
        features['VIX_regime'] = (vix > vix_ma_252).astype(int)

    if '^TYX' in cols and '^TNX' in cols:
        yield_curve = data['^TYX'] - data['^TNX']
        features['Yield_Curve'] = yield_curve
        features['Yield_Curve_change'] = yield_curve.diff(21)

    if 'LQD' in cols and 'HYG' in cols:
        credit_spread = data['LQD'] / (data['HYG'] + 1e-10)
        features['Credit_Spread'] = credit_spread
        features['Credit_Spread_change'] = credit_spread.pct_change(21)

    # ========== COMMODITY FEATURES WITH CROSS-SECTIONAL ANALYSIS ==========

    # First pass: Calculate all commodity returns and volatilities
    commodity_returns_3m = {}
    commodity_returns_1m = {}
    commodity_vols_3m = {}
    commodity_vols_1m = {}

    for ticker in commodity_tickers:
        price = data[ticker]

        # Calculate returns and volatility
        ret_3m = price.pct_change(63)
        ret_1m = price.pct_change(21)
        vol_3m = price.pct_change(1).rolling(63, min_periods=1).std()
        vol_1m = price.pct_change(1).rolling(21, min_periods=1).std()

        # Store for cross-sectional analysis
        commodity_returns_3m[ticker] = ret_3m
        commodity_returns_1m[ticker] = ret_1m
        commodity_vols_3m[ticker] = vol_3m
        commodity_vols_1m[ticker] = vol_1m

        # Absolute features
        features[f'{ticker}_mom_3m'] = ret_3m
        features[f'{ticker}_mom_1m'] = ret_1m
        features[f'{ticker}_mom_1w'] = price.pct_change(5)
        features[f'{ticker}_vol_3m'] = vol_3m
        features[f'{ticker}_vol_1m'] = vol_1m

        # Trend features
        ma_63 = price.rolling(63, min_periods=1).mean()
        ema_21 = price.ewm(span=21, min_periods=1).mean()

        features[f'{ticker}_sma_ratio'] = price / (ma_63 + 1e-10)
        features[f'{ticker}_ema_ratio'] = price / (ema_21 + 1e-10)

        # Range features
        rolling_high = price.rolling(63, min_periods=1).max()
        rolling_low = price.rolling(63, min_periods=1).min()
        range_size = rolling_high - rolling_low
        features[f'{ticker}_pct_rank'] = (price - rolling_low) / (range_size + 1e-10)

        # RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        features[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))

    # ========== NEW: CROSS-SECTIONAL FEATURES ==========

    if len(commodity_returns_3m) > 0:
        # Calculate means across all commodities
        mean_ret_3m = pd.DataFrame(commodity_returns_3m).mean(axis=1)
        mean_ret_1m = pd.DataFrame(commodity_returns_1m).mean(axis=1)
        mean_vol_3m = pd.DataFrame(commodity_vols_3m).mean(axis=1)

        for ticker in commodity_tickers:
            if ticker not in commodity_returns_3m:
                continue

            # Relative momentum (vs all commodities)
            features[f'{ticker}_mom_3m_vs_all'] = commodity_returns_3m[ticker] - mean_ret_3m
            features[f'{ticker}_mom_1m_vs_all'] = commodity_returns_1m[ticker] - mean_ret_1m

            # Relative volatility
            features[f'{ticker}_vol_3m_vs_all'] = commodity_vols_3m[ticker] - mean_vol_3m

            # Risk-adjusted momentum
            features[f'{ticker}_risk_adj_mom'] = (
                    commodity_returns_3m[ticker] / (commodity_vols_3m[ticker] + 1e-10)
            )

    # ========== NEW: SECTOR-LEVEL FEATURES ==========

    # Group returns by sector
    sector_returns_3m = {}
    for ticker in commodity_tickers:
        sector = SECTOR_MAP.get(ticker, 'Other')
        if sector not in sector_returns_3m:
            sector_returns_3m[sector] = []
        if ticker in commodity_returns_3m:
            sector_returns_3m[sector].append(commodity_returns_3m[ticker])

    # Calculate sector averages
    sector_avg_3m = {}
    for sector, rets_list in sector_returns_3m.items():
        if rets_list:
            sector_avg_3m[sector] = pd.DataFrame(rets_list).mean(axis=0)

    # Add sector momentum features
    for sector, avg_ret in sector_avg_3m.items():
        features[f'Sector_{sector}_mom_3m'] = avg_ret

    # Add relative to sector features
    for ticker in commodity_tickers:
        sector = SECTOR_MAP.get(ticker, 'Other')
        if sector in sector_avg_3m and ticker in commodity_returns_3m:
            features[f'{ticker}_mom_3m_vs_sector'] = (
                    commodity_returns_3m[ticker] - sector_avg_3m[sector]
            )

    # ========== NEW: CORRELATION FEATURES ==========

    if len(commodity_tickers) >= 3:
        # Calculate rolling correlation matrix
        commodity_prices = data[commodity_tickers]

        # Use 63-day window for correlation
        corr_values = []
        for i in range(len(commodity_prices)):
            if i >= 63:
                window = commodity_prices.iloc[i - 63:i]
                corr_matrix = window.corr()
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(corr_matrix.values, k=1)
                avg_corr = corr_matrix.values[triu_indices].mean()
                std_corr = corr_matrix.values[triu_indices].std()
                corr_values.append({'avg': avg_corr, 'std': std_corr})
            else:
                corr_values.append({'avg': 0, 'std': 0})

        features['commodity_corr_avg'] = pd.Series([c['avg'] for c in corr_values], index=data.index)
        features['commodity_corr_std'] = pd.Series([c['std'] for c in corr_values], index=data.index)

    # ========== NEW: SEASONALITY FEATURES ==========

    for ticker in commodity_tickers:
        if ticker not in data.columns:
            continue

        price = data[ticker]
        rets = price.pct_change(21)

        # Calculate average return by month
        seasonal_avg = rets.groupby(rets.index.month).transform('mean')
        seasonal_std = rets.groupby(rets.index.month).transform('std')

        features[f'{ticker}_seasonal_avg'] = seasonal_avg
        features[f'{ticker}_seasonal_std'] = seasonal_std

    # ========== LEGACY: CROSS-COMMODITY FEATURES (keep for compatibility) ==========

    if commodity_tickers:
        commodity_prices = data[commodity_tickers]
        commodity_returns = commodity_prices.pct_change(21)
        commodity_avg_return = commodity_returns.mean(axis=1)

        for ticker in commodity_tickers:
            if ticker in commodity_returns.columns:
                features[f'{ticker}_rel_strength'] = commodity_returns[ticker] - commodity_avg_return

    # Create DataFrame once from dictionary
    X = pd.DataFrame(features, index=data.index)
    return X.fillna(0)


def compute_country_features(data):
    """
    Compute enhanced features for country rotation model (60%+ Top-3 accuracy version)
    Optimized to avoid DataFrame fragmentation warnings.
    """
    # Forward fill missing data
    data = data.ffill().bfill()

    # Get country tickers
    cols = data.columns
    country_tickers = [c for c in cols if c in COUNTRIES.keys()]

    if not country_tickers:
        return pd.DataFrame(index=data.index)

    # Collect all features in a dict, then create DataFrame once
    features = {}

    # ========== MACRO FEATURES ==========

    # USD Strength (3 features)
    if 'DX-Y.NYB' in cols:
        usd_rets = data['DX-Y.NYB'].pct_change()
        features['DXY_mom_1m'] = usd_rets.rolling(21).sum()
        features['DXY_mom_3m'] = usd_rets.rolling(63).sum()
        features['DXY_trend'] = (
                data['DX-Y.NYB'].rolling(20).mean() /
                data['DX-Y.NYB'].rolling(63).mean() - 1
        )

    # Volatility Regime (3 features)
    if '^VIX' in cols:
        vix = data['^VIX']
        features['VIX_level'] = vix.rolling(21).mean()
        features['VIX_percentile'] = vix.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
        )
        features['VIX_spike'] = (vix > vix.rolling(63).quantile(0.75)).astype(int)

    # Yield Curve (4 features)
    if '^TNX' in cols:
        if '^IRX' in cols:
            yield_curve = data['^TNX'] - data['^IRX']
            features['Yield_Curve'] = yield_curve
            features['Yield_Curve_mom'] = yield_curve.diff(21)
        elif '^FVX' in cols:
            yield_curve = data['^TNX'] - data['^FVX']
            features['Yield_Curve'] = yield_curve
            features['Yield_Curve_mom'] = yield_curve.diff(21)

        features['TNX_level'] = data['^TNX'].rolling(21).mean()
        features['TNX_mom'] = data['^TNX'].diff(63)

    # Credit Spreads (3 features)
    if 'LQD' in cols and 'HYG' in cols:
        lqd_rets = data['LQD'].pct_change()
        hyg_rets = data['HYG'].pct_change()

        lqd_mom = lqd_rets.rolling(63).sum()
        hyg_mom = hyg_rets.rolling(63).sum()

        features['LQD_mom_3m'] = lqd_mom
        features['HYG_mom_3m'] = hyg_mom
        features['Credit_Spread'] = lqd_mom - hyg_mom

    # Risk Appetite (1 feature)
    if '^VIX' in cols and 'HYG' in cols:
        vix_normalized = -data['^VIX'] / data['^VIX'].rolling(252).mean()
        hyg_normalized = data['HYG'].pct_change().rolling(63).sum()
        features['Risk_Appetite'] = (vix_normalized + hyg_normalized) / 2

    # Commodities (2 features)
    if 'GLD' in cols:
        features['Gold_mom_3m'] = data['GLD'].pct_change().rolling(63).sum()

    if 'USO' in cols:
        features['Oil_mom_3m'] = data['USO'].pct_change().rolling(63).sum()

    # ========== COUNTRY-SPECIFIC FEATURES ==========

    for ticker in country_tickers:
        if ticker not in data.columns:
            continue

        rets = data[ticker].pct_change()
        country_name = COUNTRIES.get(ticker, '')

        # Multi-timeframe Momentum (4 features)
        features[f'{ticker}_mom_1m'] = rets.rolling(21).sum()
        features[f'{ticker}_mom_3m'] = rets.rolling(63).sum()
        features[f'{ticker}_mom_6m'] = rets.rolling(126).sum()
        features[f'{ticker}_mom_12m'] = rets.rolling(252).sum()

        # Risk-adjusted Momentum (1 feature)
        mean_ret = rets.rolling(63).mean()
        std_ret = rets.rolling(63).std()
        features[f'{ticker}_sharpe_3m'] = (mean_ret / std_ret) * np.sqrt(252)

        # Trend Strength (2 features)
        features[f'{ticker}_sma_ratio'] = (
                data[ticker].rolling(20).mean() / data[ticker].rolling(63).mean() - 1
        )
        features[f'{ticker}_vs_ma200'] = (
                data[ticker] / data[ticker].rolling(200).mean() - 1
        )

        # Volatility (1 feature) - CAPPED to prevent dominance
        vol_annual = rets.rolling(63).std() * np.sqrt(252)
        vol_long = vol_annual.rolling(252).mean()
        vol_z_raw = vol_annual / vol_long
        features[f'{ticker}_vol_z'] = np.clip(vol_z_raw, 0.5, 2.5)

        # Relative Strength vs SPY (1 feature)
        if 'SPY' in cols:
            spy_rets = data['SPY'].pct_change()
            features[f'{ticker}_rel_spy_3m'] = (
                    rets.rolling(63).sum() - spy_rets.rolling(63).sum()
            )

        # Relative Strength vs Region (1 feature)
        if ticker not in ['SPY']:
            benchmark = 'EFA' if ticker in ['EWJ', 'EWG', 'EWU', 'EFA'] else 'EEM'

            if benchmark in cols:
                bench_rets = data[benchmark].pct_change()
                features[f'{ticker}_rel_region_3m'] = (
                        rets.rolling(63).sum() - bench_rets.rolling(63).sum()
                )

        # Momentum Acceleration (1 feature)
        mom_3m = rets.rolling(63).sum()
        features[f'{ticker}_mom_accel'] = mom_3m - mom_3m.shift(21)

        # Drawdown (1 feature)
        cummax = data[ticker].expanding().max()
        features[f'{ticker}_drawdown'] = (data[ticker] / cummax - 1)

        # FX Features (up to 2 features)
        fx_pair = next((k for k, v in CURRENCIES.items() if v == country_name), None)
        if fx_pair and fx_pair in cols:
            fx_rets = data[fx_pair].pct_change()

            if 'DX-Y.NYB' in cols:
                usd_rets = data['DX-Y.NYB'].pct_change()
                features[f'{ticker}_FX_rel_3m'] = (
                        fx_rets.rolling(63).sum() - usd_rets.rolling(63).sum()
                )

            features[f'{ticker}_FX_trend'] = (
                    data[fx_pair].rolling(20).mean() / data[fx_pair].rolling(63).mean() - 1
            )

    # Create DataFrame once from dict (no fragmentation!)
    X = pd.DataFrame(features, index=data.index)
    return X.fillna(0)


def compute_features(data, model_type):
    """Dispatcher for feature computation by model type"""
    if model_type == 'risk':
        return compute_risk_features(data)
    elif model_type == 'sector':
        return compute_sector_features(data)
    elif model_type == 'commodity':
        return compute_commodity_features(data)
    elif model_type == 'country':
        return compute_country_features(data)
    else:
        return pd.DataFrame()


def compute_trend_features(df):
    """
    Compute features for trend prediction models.
    Must match exactly the features used in train_trends.py.
    """
    if len(df) < 200:
        return pd.DataFrame()

    close = df['Close'].squeeze()

    # --- Trend / Technical ---
    rsi = compute_RSI(close, 14)
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()
    sma50_dist = (close - sma50) / sma50
    sma200_dist = (close - sma200) / sma200

    # --- Linear Regression (14-day) ---
    period = 14
    x = np.arange(period)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    vals = close.values
    slopes = np.full(len(vals), np.nan)
    r2s = np.full(len(vals), np.nan)
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1:i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_yy = ((y - y_mean) ** 2).sum()
        slope = ss_xy / ss_xx
        slopes[i] = slope / y_mean
        r2s[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-9)
    lr_slope = pd.Series(slopes, index=close.index)
    lr_r2 = pd.Series(r2s, index=close.index)

    # --- Volatility Regime ---
    log_returns = np.log(close / close.shift(1))
    real_vol = log_returns.rolling(21).std() * np.sqrt(252)
    high_low = close.rolling(2).max() - close.rolling(2).min()
    atr21_pct = high_low.rolling(21).mean() / close

    X = pd.DataFrame({
        'RSI14':       rsi,
        'SMA50_dist':  sma50_dist,
        'SMA200_dist': sma200_dist,
        'LR14_slope':  lr_slope,
        'LR14_r2':     lr_r2,
        'RealVol21':   real_vol,
        'ATR21_pct':   atr21_pct,
    }, index=close.index)

    return X


def explain_tree_prediction(model, X_scaled, feature_names, class_idx, top_n=5):
    """
    Extract feature contributions for tree-based models using class-specific importance.

    Args:
        model: Trained model (RandomForest, GradientBoosting, etc.)
        X_scaled: Scaled feature array for prediction
        feature_names: List of feature names
        class_idx: Index of the class to explain
        top_n: Number of top features to return

    Returns:
        List of tuples (feature_name, contribution_score)
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier

        # For GradientBoosting, extract class-specific tree importances
        if isinstance(model, GradientBoostingClassifier):
            # Get estimators for this specific class
            n_classes = len(model.classes_)

            if n_classes == 2:
                # Binary classification: use all estimators
                class_estimators = model.estimators_[:, 0]
            else:
                # Multi-class: get estimators for this class
                class_estimators = model.estimators_[:, class_idx]

            # Compute average importance across this class's trees
            importances = np.zeros(len(feature_names))
            for tree in class_estimators:
                importances += tree.feature_importances_
            importances /= len(class_estimators)

        elif hasattr(model, 'feature_importances_'):
            # Fallback: use global feature importances
            importances = model.feature_importances_
        else:
            return []

        # Weight by feature values (larger absolute values matter more)
        feature_values = X_scaled[0]
        weighted_importance = importances * np.abs(feature_values)

        # Create feature contribution pairs
        contributions = list(zip(feature_names, weighted_importance))
        contributions.sort(key=lambda x: x[1], reverse=True)

        return contributions[:top_n]
    except Exception as e:
        print(f"  Warning: Could not extract feature importance: {e}")
        return []


def get_feature_descriptions():
    """Return human-readable descriptions for common features"""
    return {
        'DXY_mom': 'Dollar momentum (3-month)',
        'DXY_Mom': 'Dollar momentum (1-month)',
        'VIX_level': 'Market volatility (VIX avg)',
        'VIX_Level': 'Current VIX level',
        'MOVE_level': 'Bond volatility',
        'MOVE_Level': 'Current bond volatility',
        'Yield_Curve': '30Y-10Y yield spread',
        'TNX_vol': '10Y Treasury volatility',
        'Credit_Spread': 'Investment grade vs High yield',
        'Credit_Spread_Proxy': 'Credit risk indicator',
        'Breadth_Ratio': 'Equal-weight vs Cap-weight',
        'Breadth_MA_Diff': 'Breadth trend deviation',
        'SPY_Trend': 'S&P 500 vs 200-day MA',
        'Defensive_Rotation': 'Defensive vs Growth sectors',
        'XLF_Relative_Strength': 'Financials vs Market',
        'Labor_Stress_Proxy': 'Labor market stress signal',
        'RSI14':       'Relative Strength Index',
        'SMA50_dist':  'Distance from 50-day MA',
        'SMA200_dist': 'Distance from 200-day MA',
        'LR14_slope':  '14-day linear regression slope',
        'LR14_r2':     '14-day linear regression R²',
        'RealVol21':   '21-day realised volatility',
        'ATR21_pct':   '21-day ATR as % of price',
    }


def format_feature_contribution(feature_name, value, descriptions):
    """Format a feature contribution with description and value"""
    # Extract base feature name (remove ticker prefix for momentum features)
    if '_3M_Mom' in feature_name:
        ticker = feature_name.replace('_3M_Mom', '')
        desc = f'{ticker} 3-month momentum'
    elif '_mom' in feature_name and not feature_name.startswith('DXY'):
        ticker = feature_name.split('_')[0]
        desc = f'{ticker} momentum'
    elif '_vol' in feature_name:
        ticker = feature_name.split('_')[0]
        desc = f'{ticker} volatility'
    else:
        desc = descriptions.get(feature_name, feature_name)

    return f"{desc:<35} (importance: {value:.3f})"


def predict_trends(model_path, tickers, friendly_names, as_of_date=None, use_cache=True, explain=True):
    """
    Predict trend confidence for multiple assets using dual model approach.

    Args:
        model_path: Path to the trend model file (trend_model.joblib)
        tickers: List of ticker symbols to analyze
        friendly_names: Dict mapping tickers to friendly names
        as_of_date: Date to predict for (None = today)
        use_cache: Whether to use shared data cache
        explain: Whether to include feature explanations

    Returns:
        Dict with predictions for each asset
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    # Load trend model bundle (cached)
    try:
        bundle = _get_model_bundle(model_path)
        m_fast = bundle.get('model_fast', None)
        m_slow = bundle.get('model_slow', None)
        scaler = bundle.get('scaler', None)
        features = bundle.get('features', None)

        if not all([m_fast, m_slow, scaler, features]):
            return {
                'date': as_of_date.strftime('%Y-%m-%d'),
                'predictions': {
                    friendly_names.get(t, t): {'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'No Model'} for t
                    in tickers}
            }
    except FileNotFoundError:
        return {
            'date': as_of_date.strftime('%Y-%m-%d'),
            'predictions': {
                friendly_names.get(t, t): {'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'No Model'} for t in
                tickers}
        }

    # Determine date range
    max_rolling = 200
    buffer_days = 10
    start_date = as_of_date - pd.Timedelta(days=int(max_rolling * 1.5 + buffer_days))
    fetch_end = as_of_date + timedelta(days=1)

    # Download data - use cache if enabled
    if use_cache:
        raw_data = _get_shared_predict_data(tickers, start_date, fetch_end)
    else:
        raw_data = yf.download(tickers, start=start_date, end=fetch_end, progress=False, auto_adjust=False,
                               group_by='column')

    # Handle MultiIndex properly
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data
    else:
        data = raw_data

    # Process each ticker
    predictions = {}
    descriptions = get_feature_descriptions()

    for ticker in tickers:
        try:
            # Extract ticker data based on structure
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    try:
                        ticker_data = pd.DataFrame({
                            'Close': data['Close'][ticker],
                            'High': data['High'][ticker] if 'High' in data.columns.get_level_values(0) else
                            data['Close'][ticker],
                            'Low': data['Low'][ticker] if 'Low' in data.columns.get_level_values(0) else data['Close'][
                                ticker],
                            'Volume': data['Volume'][ticker] if 'Volume' in data.columns.get_level_values(
                                0) else pd.Series(0, index=data.index)
                        }).dropna()
                    except KeyError:
                        predictions[friendly_names.get(ticker, ticker)] = {
                            'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'No Data'
                        }
                        continue
                else:
                    try:
                        ticker_data = data.xs(ticker, level=1, axis=1).dropna()
                    except KeyError:
                        predictions[friendly_names.get(ticker, ticker)] = {
                            'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'No Data'
                        }
                        continue
            else:
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    predictions[friendly_names.get(ticker, ticker)] = {
                        'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'Data Error'
                    }
                    continue

            if len(ticker_data) < 200:
                predictions[friendly_names.get(ticker, ticker)] = {
                    'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'Insufficient Data'
                }
                continue

            # Compute features for the full series, take last row
            X = compute_trend_features(ticker_data)

            if X.empty:
                predictions[friendly_names.get(ticker, ticker)] = {
                    'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'Error'
                }
                continue

            X_last = X.iloc[[-1]]

            # Fill missing features
            X_live_dict = {f: X_last[f].iloc[0] if f in X_last.columns else 0.0 for f in features}
            X_live = pd.DataFrame([X_live_dict], columns=features)

            # Scale and predict
            X_scaled = scaler.transform(X_live)

            fast_idx = list(m_fast.classes_).index(1)
            slow_idx = list(m_slow.classes_).index(1)

            p_fast = m_fast.predict_proba(X_scaled)[0][fast_idx] * 100
            p_slow = m_slow.predict_proba(X_scaled)[0][slow_idx] * 100

            # Volatility adjustment
            close = ticker_data['Close'].squeeze()
            atr = compute_ATR(ticker_data, 14).iloc[-1]
            last_close = close.iloc[-1]
            vol_adj = 1 - min(atr / last_close, 0.1)

            f_conf = round(p_fast * vol_adj, 1)
            s_conf = round(p_slow * vol_adj, 1)

            # Blended score
            blended = round(s_conf * 0.7 + f_conf * 0.3, 1)

            # Determine regime
            if f_conf > 60 and s_conf > 60:
                regime = "Aggressive Bull" if f_conf > s_conf else "Structural Bull"
            elif f_conf < 40 and s_conf < 40:
                regime = "Capitulation/Bear"
            elif f_conf > 60 and s_conf < 45:
                regime = "Dead Cat Bounce"
            else:
                regime = "Neutral/Chop"

            pred_data = {
                'fast': f_conf,
                'slow': s_conf,
                'blended': blended,
                'regime': regime
            }

            # Add explanations if requested
            if explain:
                slow_contrib = explain_tree_prediction(m_slow, X_scaled, features, slow_idx, top_n=3)
                pred_data['drivers'] = [format_feature_contribution(f, v, descriptions)
                                        for f, v in slow_contrib]

            predictions[friendly_names.get(ticker, ticker)] = pred_data

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            predictions[friendly_names.get(ticker, ticker)] = {
                'fast': 50.0, 'slow': 50.0, 'blended': 50.0, 'regime': 'Error'
            }

    return {
        'date': as_of_date.strftime('%Y-%m-%d'),
        'predictions': dict(sorted(predictions.items(), key=lambda x: x[1]['blended'], reverse=True))
    }


def predict_assets(model_path, tickers, friendly_names, model_type, as_of_date=None, use_cache=True, explain=True):
    """
    Fetch data, compute features, and run prediction bundle with explanations.

    Args:
        model_path: Path to the model file
        tickers: List of tickers to fetch
        friendly_names: Dict mapping tickers to friendly names
        model_type: Type of model ('risk', 'sector', 'commodity', 'country')
        as_of_date: Date to predict for (None = today)
        use_cache: Whether to use shared data cache (default True)
        explain: Whether to include feature explanations (default True)
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    # Load model bundle (cached)
    bundle = _get_model_bundle(model_path)
    model = bundle['model']
    scaler = bundle.get('scaler', None)  # FIX: use .get() — some models (e.g. RandomForest) have no scaler
    feature_names = bundle['features']

    # Check if model has label_encoder (new sector model format)
    label_encoder = bundle.get('label_encoder', None)

    # Determine minimal start date for rolling windows
    max_rolling = 200
    buffer_days = 10
    start_date = as_of_date - pd.Timedelta(days=int(max_rolling * 1.5 + buffer_days))
    fetch_end = as_of_date + timedelta(days=1)

    # Download data
    if use_cache:
        raw_data = _get_shared_predict_data(tickers, start_date, fetch_end)
    else:
        raw_data = yf.download(tickers, start=start_date, end=fetch_end, progress=False,
                               auto_adjust=False, group_by='column')

    # Extract Close prices
    data = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data

    X = compute_features(data, model_type)

    # Ensure all features exist - use pd.concat to avoid fragmentation
    missing_features = {feat: 0 for feat in feature_names if feat not in X.columns}
    if missing_features:
        missing_df = pd.DataFrame(missing_features, index=X.index)
        X = pd.concat([X, missing_df], axis=1)

    # Scale and predict (FIX: handle missing scaler)
    X_tail = X.tail(1)[feature_names]
    X_scaled = scaler.transform(X_tail) if scaler is not None else X_tail.values

    proba = model.predict_proba(X_scaled)[0]

    # Handle class labels (could be integers or strings)
    classes = model.classes_

    # Decode if label encoder exists (new sector model)
    if label_encoder is not None:
        classes = label_encoder.inverse_transform(classes)

    # Build probability dictionary
    if model_type == 'risk':
        proba_dict = {f"Class {c}": p for c, p in zip(classes, proba)}
    else:
        proba_dict = {friendly_names.get(c, c): p for c, p in zip(classes, proba)}

    result = {
        'date': as_of_date.strftime('%Y-%m-%d'),
        'probabilities': dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))
    }

    # Add explanations if requested
    if explain:
        explanations = {}
        descriptions = get_feature_descriptions()

        for i, class_name in enumerate(classes):
            display_name = friendly_names.get(class_name, class_name)
            if model_type == 'risk':
                display_name = f"Class {class_name}"

            contrib = explain_tree_prediction(model, X_scaled, feature_names, i, top_n=5)
            explanations[display_name] = [format_feature_contribution(f, v, descriptions)
                                          for f, v in contrib]

        result['explanations'] = explanations

    return result


def predict_commodities(
        sector_model_path,
        commodity_model_path,
        friendly_names,
        as_of_date=None,
        use_cache=True,
        explain=True,
        top_n_sectors=3,
        top_n_per_sector=1
):
    """
    Two-stage hierarchical commodity prediction.

    Stage 1: Use sector model to identify promising sectors
    Stage 2: Within those sectors, pick best commodities

    Returns same format as predict_assets() for UI compatibility.

    Args:
        sector_model_path: Path to sector model (commodity_sector_model.joblib)
        commodity_model_path: Path to commodity model (fallback)
        friendly_names: Dict mapping tickers to names
        as_of_date: Date to predict for
        use_cache: Whether to use data cache
        explain: Whether to include explanations
        top_n_sectors: Number of sectors to select (default 3)
        top_n_per_sector: Commodities per sector (default 1)

    Returns:
        Dict with same structure as predict_assets() output
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.today()
    else:
        as_of_date = pd.Timestamp(as_of_date)

    tickers = list(COMMODITIES.keys()) + ML_MACRO_TICKERS

    # Define sector mapping
    SECTOR_MAP = {
        'GC=F': 'Precious_Metals',
        'SI=F': 'Precious_Metals',
        'PL=F': 'Precious_Metals',
        'PA=F': 'Precious_Metals',
        'HG=F': 'Industrial_Metals',
        'CL=F': 'Energy',
        'NG=F': 'Energy',
        'RB=F': 'Energy',
        'ZC=F': 'Grains',
        'ZS=F': 'Grains',
        'ZW=F': 'Grains',
        'KC=F': 'Softs',
        'SB=F': 'Softs',
        'CT=F': 'Softs',
        'CC=F': 'Softs',
    }

    # Inverse mapping: sector -> commodities
    sector_to_commodities = {}
    for commodity, sector in SECTOR_MAP.items():
        if sector not in sector_to_commodities:
            sector_to_commodities[sector] = []
        sector_to_commodities[sector].append(commodity)

    try:
        # Load sector model
        bundle = _get_model_bundle(sector_model_path)
        sector_model = bundle['model']
        sector_features = bundle['features']

        # Scaler is optional for RandomForest models
        sector_scaler = bundle.get('scaler', None)

        # Download data
        max_rolling = 252
        buffer_days = 10
        start_date = as_of_date - pd.Timedelta(days=int(max_rolling * 1.5 + buffer_days))
        fetch_end = as_of_date + timedelta(days=1)

        if use_cache:
            raw_data = _get_shared_predict_data(tickers, start_date, fetch_end)
        else:
            raw_data = yf.download(tickers, start=start_date, end=fetch_end,
                                   progress=False, auto_adjust=False, group_by='column')

        # Extract Close prices
        data = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data

        # Compute features
        X = compute_commodity_features(data)

        # Ensure all sector features exist
        missing_features = {feat: 0 for feat in sector_features if feat not in X.columns}
        if missing_features:
            missing_df = pd.DataFrame(missing_features, index=X.index)
            X = pd.concat([X, missing_df], axis=1)

        # ========== STAGE 1: SECTOR PREDICTION ==========

        # Prepare features - scale only if scaler exists
        X_sector = X.tail(1)[sector_features]

        if sector_scaler is not None:
            X_sector_scaled = sector_scaler.transform(X_sector)
        else:
            # No scaling needed for RandomForest
            X_sector_scaled = X_sector.values

        sector_probs = sector_model.predict_proba(X_sector_scaled)[0]

        # Get sector classes
        sector_classes = sector_model.classes_

        # Decode if label encoder exists
        label_encoder = bundle.get('label_encoder', None)
        if label_encoder is not None:
            sector_classes = label_encoder.inverse_transform(sector_classes)

        # Create sector probability dict
        sector_prob_dict = {sector: prob for sector, prob in zip(sector_classes, sector_probs)}

        # Get top N sectors
        top_sectors = sorted(sector_prob_dict.items(), key=lambda x: x[1], reverse=True)[:top_n_sectors]

        # ========== STAGE 2: COMMODITY SELECTION WITHIN SECTORS ==========

        # Calculate momentum scores for commodities.
        # FIX: skip macro tickers; dropna() so .iloc[-1] lands on real data
        # even when the most recent row is NaN (e.g. market closed today).
        commodity_scores = {}

        for ticker in tickers:
            if ticker not in data.columns:
                continue
            if ticker not in COMMODITIES:
                continue

            price = data[ticker].dropna()

            if len(price) < 63:
                continue

            ret_3m = price.pct_change(63).iloc[-1]
            ret_1m = price.pct_change(21).iloc[-1]
            vol_3m = price.pct_change(1).tail(63).std()

            if not np.isnan(ret_3m) and not np.isnan(ret_1m) and not np.isnan(vol_3m) and vol_3m > 1e-10:
                risk_adj_mom = ret_3m / vol_3m
                momentum_trend = ret_1m / ret_3m if abs(ret_3m) > 1e-6 else 0
                commodity_scores[ticker] = 0.7 * risk_adj_mom + 0.3 * momentum_trend

        # Select commodities from top sectors
        final_probabilities = {}
        explanations = {}

        for sector, sector_prob in top_sectors:
            # Get commodities in this sector
            sector_commodities = [c for c in sector_to_commodities.get(sector, [])
                                  if c in commodity_scores]

            if not sector_commodities:
                continue

            # Sort by momentum score
            sorted_commodities = sorted(
                [(c, commodity_scores[c]) for c in sector_commodities],
                key=lambda x: x[1],
                reverse=True
            )

            # Pick top N per sector
            for i, (commodity, score) in enumerate(sorted_commodities[:top_n_per_sector]):
                # Weight by sector probability
                commodity_prob = sector_prob * (1.0 - i * 0.2)  # Slight penalty for rank

                display_name = friendly_names.get(commodity, commodity)
                final_probabilities[display_name] = commodity_prob

                # Add explanation if requested
                if explain:
                    explanations[display_name] = [
                        f"Sector: {sector:<20} (probability: {sector_prob:.1%})",
                        f"Rank in sector: #{i + 1:<10} (score: {score:.3f})",
                        f"Selection strategy: Hierarchical (sector-based)"
                    ]

        # Normalize probabilities to sum to 1
        total_prob = sum(final_probabilities.values())
        if total_prob > 0:
            final_probabilities = {k: v / total_prob for k, v in final_probabilities.items()}

        # Build result in same format as predict_assets
        result = {
            'date': as_of_date.strftime('%Y-%m-%d'),
            'probabilities': dict(sorted(final_probabilities.items(),
                                         key=lambda x: x[1], reverse=True))
        }

        if explain:
            result['explanations'] = explanations

        return result

    except FileNotFoundError as e:
        print(f"⚠️  Sector model not found: {e}")
        print("   Falling back to standard commodity prediction...")

        # Fallback to regular prediction
        return predict_assets(
            commodity_model_path, tickers, friendly_names, 'commodity',
            as_of_date, use_cache, explain
        )

    except Exception as e:
        print(f"❌ Error in hierarchical prediction: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to regular prediction
        return predict_assets(
            commodity_model_path, tickers, friendly_names, 'commodity',
            as_of_date, use_cache, explain
        )


def print_prediction(title, result, show_explanations=True):
    """Pretty-print predictions with feature explanations"""
    print(f"\n📊 {title.upper()} PREDICTIONS ({result['date']})")
    print("=" * 80)

    # Check if this is a trend prediction result
    if 'predictions' in result:
        for name, data in result['predictions'].items():
            blended = data.get('blended', 0)
            slow = data.get('slow', 0)
            fast = data.get('fast', 0)
            regime = data.get('regime', 'N/A')

            print(f"\n  {name:<25} {blended:>3.1f}%  [slow:{slow:>3.1f}% fast:{fast:>3.1f}%] [{regime}]")

            # Show drivers if available
            if show_explanations and 'drivers' in data:
                print("    Key drivers:")
                for driver in data['drivers']:
                    print(f"      • {driver}")
    else:
        # Standard probability output
        top_n = 5  # Show top 5 predictions with explanations
        count = 0

        for name, p in result['probabilities'].items():
            prob_pct = round(float(p) * 100, 2)

            if "Class" in name:
                label = "Risk-On (Positive)" if "1" in name else "Risk-Off (Negative)"
                print(f"\n  {label:<35} {prob_pct:>6.1f}%")
            else:
                print(f"\n  {name:<35} {prob_pct:>6.1f}%")

            # Show explanations for top predictions
            if show_explanations and count < top_n and 'explanations' in result:
                if name in result['explanations']:
                    print("    Key drivers:")
                    for driver in result['explanations'][name]:
                        print(f"      • {driver}")
                    count += 1


# ----------------- Main Execution -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Prediction Dashboard with Explanations")
    parser.add_argument("--date", type=str, default=None, help="Date to predict (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")
    parser.add_argument("--no-explain", action="store_true", help="Disable feature explanations")
    args = parser.parse_args()

    use_cache = not args.no_cache
    show_explain = not args.no_explain

    model_configs = [
        ("Risk Regime", "risk_model.joblib", list(set(ML_MACRO_TICKERS + ['RSP', 'SPY'] + list(SECTOR_NAMES.keys()))),
         {}, 'risk'),
        ("Sector Rotation", "sector_model.joblib", list(SECTOR_NAMES.keys()) + ML_MACRO_TICKERS + ['QQQ', 'GLD', 'USO'],
         SECTOR_NAMES, 'sector'),
        # Commodities handled separately below with hierarchical prediction
        ("Countries", "country_model.joblib", list(COUNTRIES.keys()) + ML_MACRO_TICKERS + list(CURRENCIES.keys()),
         COUNTRIES, 'country')
    ]

    for title, path, tickers, names, m_type in model_configs:
        try:
            res = predict_assets(path, tickers, names, m_type, args.date,
                                 use_cache=use_cache, explain=show_explain)
            print_prediction(title, res, show_explanations=show_explain)
        except FileNotFoundError:
            print(f"\n⚠️ File '{path}' not found. Skipping {title}.")
        except Exception as e:
            print(f"\n❌ Error processing {title}: {e}")

    # Special handling for commodities with hierarchical prediction
    print("\n" + "=" * 80)
    try:
        res = predict_commodities(
            sector_model_path="commodity_sector_model.joblib",
            commodity_model_path="commodity_model.joblib",  # fallback
            friendly_names=COMMODITIES,
            as_of_date=args.date,
            use_cache=use_cache,
            explain=show_explain,
            top_n_sectors=3,
            top_n_per_sector=1
        )
        print_prediction("Commodities (Hierarchical)", res, show_explanations=show_explain)

    except FileNotFoundError:
        print(f"\n⚠️ Commodity models not found. Skipping commodities.")
    except Exception as e:
        print(f"\n❌ Error processing Commodities: {e}")

    # Add trend predictions
    print("\n" + "=" * 80)
    print("TREND ANALYSIS")
    print("=" * 80)
    try:
        trend_res = predict_trends(
            "trend_model.joblib",
            list(TREND_ASSETS.keys()),
            TREND_ASSETS,
            args.date,
            use_cache=use_cache,
            explain=show_explain
        )
        print_prediction("Trend Predictions", trend_res, show_explanations=show_explain)
    except FileNotFoundError:
        print(f"\n⚠️ File 'trend_model.joblib' not found. Skipping trend predictions.")
    except Exception as e:
        print(f"\n❌ Error processing trend predictions: {e}")
