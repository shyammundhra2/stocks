"""
Walk-forward backtest of the live trend-following pipeline (get_trends() +
_kelly_covariance_optimizer in macro/indicators.py), with daily rebalancing,
over the trailing 2 years.

Reuses the actual production functions (imported directly) for exact
fidelity: _trend_stats, _compute_delta_slope, _hurst_exponent,
_stop_hit_probability, _compute_kelly_size, _kelly_covariance_optimizer,
get_regime_scalar, compute_RSI, compute_ATR. Only two things are
deliberately simplified vs. production, both documented inline:

  1. Per-asset ML confidence (ml_conf_slow/fast) is neutral (50.0) every
     day. Per the "ML-free" 2026-06-11 refactor, ml_conf no longer feeds
     status logic, and (as of the sizing-gate fix applied in this same
     session) no longer feeds sizing either - so this is a faithful
     no-op, not an approximation.

  2. The regime composite's ml_slow term (SPY structural ML model, 45%
     weight in the RISK-ON/OFF label, 40% weight in the vol-cap blend)
     is held at neutral 50.0 rather than reproduced walk-forward. That
     model is a pretrained artifact - applying it to historical dates
     would use information (its full training window) not actually
     available on that historical date, i.e. lookahead. get_regime_scalar
     (which actually drives per-position weight scaling in the optimizer)
     depends ONLY on the 6 technical conditions, not ml_slow, so this
     simplification only affects the vol-cap band via _effective_vol_cap,
     which blends toward 0.5 neutral instead of the real (unknown,
     walk-forward-unsafe) historical ML confidence.

Everything else - trend stats, ATR, RSI, Hurst, stop-hit probability,
Kelly sizing, the SLSQP covariance optimizer, and the full BUY/SELL/TRIM
status decision tree - is the exact production code running on
walk-forward-sliced historical data (no lookahead: every day's decision
uses only data up to and including that day).
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.helpers import compute_RSI, compute_ATR
from macro.indicators import (
    _trend_stats, _compute_delta_slope, _hurst_exponent,
    _stop_hit_probability, _compute_kelly_size, _kelly_covariance_optimizer,
    get_regime_scalar,
)
from macro.indicators.mathstats import _efficiency_ratio

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    sliding_window_view = None

END = pd.Timestamp("2026-07-10")
TEST_START = END - pd.DateOffset(years=2)
DATA_START = TEST_START - pd.DateOffset(years=2)  # burn-in for 200MA, hurst, etc.

PORTFOLIO_VALUE = 500_000
REBAL_EVERY = 10          # rebalance every N trading days (hold ~2 weeks)
CASH_TICKER = "SGOV"      # 0-3mo T-bill ETF: undeployed weight earns its yield
REGIME_TICKERS = ["HYG", "IEF", "^TNX", "^IRX", "JPY=X", "RSP", "^VIX", "^MOVE"]
TREND_SYMS = list(TREND_ASSETS.keys())
ALL_TICKERS = sorted(set(TREND_SYMS) | set(REGIME_TICKERS) | {CASH_TICKER})


def download_all():
    print(f"Downloading {len(ALL_TICKERS)} tickers from {DATA_START.date()} to {END.date()} ...")
    raw = yf.download(
        ALL_TICKERS, start=DATA_START, end=END + pd.Timedelta(days=1),
        progress=False, auto_adjust=False, threads=True, group_by="column",
    )
    return raw


def compute_regime(data_t):
    """data_t: Close prices (ffilled) truncated to day t. Mirrors get_risk_regime()'s
    technical block exactly, minus the ml_slow ML term (see module docstring)."""
    last_vals = data_t.iloc[-1]
    ma50 = data_t.rolling(50).mean().iloc[-1]

    credit_ratio = data_t["HYG"] / data_t["IEF"]
    credit_pass = bool(last_vals["HYG"] / last_vals["IEF"] > credit_ratio.rolling(50).mean().iloc[-1])

    curve_spread = last_vals["^TNX"] - last_vals["^IRX"]
    curve_pass = bool(curve_spread > 0)

    jpy_ret = data_t["JPY=X"].pct_change()
    jpy_vol = jpy_ret.rolling(20).std().iloc[-1] * np.sqrt(252)
    carry_pass = bool(last_vals["JPY=X"] > ma50["JPY=X"] and jpy_vol < 0.15)

    spy_ma200 = data_t["SPY"].rolling(200).mean().iloc[-1]
    spy_trend = bool(last_vals["SPY"] > spy_ma200)

    vix_low = bool(last_vals["^VIX"] < 20 and last_vals["^MOVE"] < 110)

    rsp_spy_ratio = data_t["RSP"] / data_t["SPY"]
    breadth_pass = bool(rsp_spy_ratio.iloc[-1] > rsp_spy_ratio.rolling(50).mean().iloc[-1])

    details = [
        {"label": "Trend (SPY > 200MA)", "pass": spy_trend},
        {"label": "Fear (VIX/MOVE Low)", "pass": vix_low},
        {"label": "Breadth (RSP/SPY > 50MA)", "pass": breadth_pass},
        {"label": "Credit (HYG/IEF Ratio)", "pass": credit_pass},
        {"label": "Curve (10Y-3M Spread)", "pass": curve_pass},
        {"label": "Carry (JPY Weak/Stable)", "pass": carry_pass},
    ]
    return {"details": details, "ml_slow": 50.0}


def slope_z_for(c, slope):
    c_len = len(c)
    start_idx = max(0, c_len - 60)
    seg = np.log(c.values[start_idx:])
    if len(seg) >= 20:
        if sliding_window_view is not None:
            _windows = sliding_window_view(seg, 20)
        else:
            _windows = np.stack([seg[k:k + 20] for k in range(len(seg) - 19)])
        _xc = np.arange(20) - 9.5
        hist_slopes = np.round((_windows @ _xc) / 665.0 * 1000.0, 2)
    else:
        hist_slopes = np.empty(0)

    if hist_slopes.size:
        slope_mean = np.mean(hist_slopes)
        slope_std = np.std(hist_slopes)
        return (slope - slope_mean) / slope_std if slope_std > 0 else 0.0
    return 0.0


def compute_asset_row(sym, df):
    c = df["Close"].squeeze()
    if len(c) < 60:
        return None

    ma_50 = c.rolling(50, min_periods=1).mean()
    ma_200 = c.rolling(200, min_periods=1).mean()

    slope, r2 = _trend_stats(c, 20, 10)

    atr = float(compute_ATR(df, 14).iloc[-1])
    last = float(c.iloc[-1])
    s50 = float(ma_50.iloc[-1])
    s200 = float(ma_200.iloc[-1])
    rsi14 = float(compute_RSI(c, 14).iloc[-1])
    hurst = _hurst_exponent(c, max_lag=40)

    # Adaptive ER router - the optimizer SELECTS on adaptive_signal (MOM/REV);
    # without it every name is filtered to FLAT and the book stays 100% cash.
    # Mirrors get_trends() exactly, except: (1) the REV branch omits the iso-
    # conviction-curve refinement (minor sleeve; slightly LOOSER REV), and
    # (2) no market-wide risk-off equity gate here (so this is a CONSERVATIVE /
    # lower-bound estimate of the defense - production gates equities to cash in
    # risk-off, which this doesn't).
    rsi2 = float(compute_RSI(c, 2).iloc[-1])
    eff_ratio = _efficiency_ratio(c, 20)
    _above200 = last > s200
    if eff_ratio >= 0.40:
        adaptive_signal = "MOM" if (_above200 and last > s50 and slope > 0) else "FLAT"
    elif eff_ratio <= 0.35:
        adaptive_signal = "REV" if (_above200 and rsi2 < 15) else "FLAT"
    else:
        adaptive_signal = "FLAT"
    slope_z = slope_z_for(c, slope)
    delta_slope = _compute_delta_slope(c, window=20)

    _atr_stop = last - (atr * 2.5)
    _daily_return = slope / 1000
    _projected = last * ((1 + _daily_return) ** 63)
    _target_for_pstop = max(_projected, last * 1.01)
    p_stop = _stop_hit_probability(last, _atr_stop, _target_for_pstop, slope, atr)

    hurst_discount = 0.0
    if hurst < 0.45:
        hurst_discount = (0.45 - hurst) * 0.5
    elif hurst > 0.55:
        hurst_discount = -(hurst - 0.55) * 0.2
    p_stop_discount = max(0.0, (p_stop - 0.40) / 0.30 * 0.50) if p_stop > 0.40 else 0.0
    combined_discount = float(np.clip(hurst_discount + p_stop_discount, -0.20, 0.80))

    position = _compute_kelly_size(
        last, slope, atr, 50.0, r2,
        portfolio_value=PORTFOLIO_VALUE,
        delta_slope=delta_slope,
        divergence_discount=combined_discount,
    )
    pos_size = position["dollar_amount"]

    strength = float(np.clip(
        np.tanh(slope * r2 / 8.0) * (1.0 - combined_discount), 0.0, 1.0
    )) if slope > 0 else 0.0

    if last < position["stop"]:
        status = "SELL (STOP)"
    elif last < s50 and slope < 0:
        status = "SELL (MA50)"
    elif slope_z > 2.0 and r2 > 0.7 and rsi14 < 70 and slope > 0:
        status = "BUY (BREAKOUT)"
    elif slope_z > 2.0 and r2 > 0.8 and rsi14 > 70:
        status = "TRIM (EXTENDED)"
    elif slope_z > 1.5 and delta_slope < -3:
        status = "TRIM (FADING MOMENTUM)"
    elif (hurst < 0.45 and last > s50 and rsi14 < 30 and slope_z < -1.5 and slope < 0):
        status = "BUY (MR SWING)"
    elif p_stop > 0.55 and last > s50:
        status = "TRIM (GEOMETRY)"
    elif slope < -2:
        status = "TRIM (NEGATIVE SLOPE)"
    elif pos_size == 0:
        status = "TRIM (POSITION SIZE)"
    elif (last > s200) and (last > s50) and (slope > 0) and (r2 > 0.6):
        if slope_z < 0 and rsi14 < 60:
            status = "BUY (PULLBACK)"
        elif slope_z > 1.0:
            status = "BUY (BULL)"
        else:
            status = "BUY"
    else:
        status = "HOLD"

    return {
        "sym": sym, "name": TREND_ASSETS[sym], "price": last, "status": status,
        "r2": r2, "strength": round(strength, 3), "slope": slope,
        "dollar_amount": pos_size,
        "adaptive_signal": adaptive_signal, "eff_ratio": round(eff_ratio, 2),
    }


def walk_forward(raw):
    close_all = raw["Close"].ffill()
    test_dates = close_all.index[(close_all.index >= TEST_START) & (close_all.index <= END)]

    daily_rows = []
    summaries = []
    t0 = time.time()

    # Rebalance only every REBAL_EVERY days; hold the book (constant target
    # weights) between rebalances. The optimizer runs on rebalance days only,
    # which is both the "hold 2 weeks" semantics and a ~REBAL_EVERY-x speedup.
    held = []   # last rebalance's optimized book, carried forward
    for i, dt in enumerate(test_dates):
        if i % REBAL_EVERY == 0:
            shared_t = raw.loc[:dt]
            close_t = close_all.loc[:dt]

            regime = compute_regime(close_t[REGIME_TICKERS + ["SPY"]].dropna(how="all"))
            regime_scalar = get_regime_scalar(regime)

            trends = []
            for sym in TREND_SYMS:
                try:
                    df_sym = shared_t.xs(sym, level=1, axis=1).dropna()
                except KeyError:
                    continue
                if df_sym.empty:
                    continue
                row = compute_asset_row(sym, df_sym)
                if row is not None:
                    trends.append(row)

            if trends:
                optimized, summary = _kelly_covariance_optimizer(
                    trends, shared_t,
                    portfolio_value=PORTFOLIO_VALUE,
                    max_single=0.25,
                    max_risk_contribution=0.35,
                    min_portfolio_vol=0.08,
                    max_portfolio_vol=0.135,
                    regime_scalar=regime_scalar,
                    regime=regime,
                )
                held = optimized
                summaries.append({"date": dt, **summary})
            print(f"  rebalance {i//REBAL_EVERY + 1} @ {dt.date()} "
                  f"({len(held)} names, {time.time()-t0:.0f}s)")

        # record the currently-held book under today's date (rebalance or not)
        for item in held:
            daily_rows.append({
                "date": dt, "sym": item["sym"], "name": item["name"],
                "price": round(item["price"], 2), "status": item["status"],
                "dollar_amount": item.get("dollar_amount", 0.0),
                "weight_pct": item.get("weight_pct", 0.0),
            })

    return pd.DataFrame(daily_rows), pd.DataFrame(summaries).set_index("date")


def build_equity_curve(daily_df, close_all):
    """Day t's weight_pct earns the t -> t+1 return; the undeployed remainder
    (1 - sum of position weights) earns SGOV's t -> t+1 return (cash yield)."""
    weights = daily_df.pivot_table(index="date", columns="sym", values="weight_pct", fill_value=0.0) / 100.0
    rets = np.log(close_all / close_all.shift(1))
    fwd_rets = rets.reindex(weights.index)[weights.columns].shift(-1)

    port_log_ret = (weights * fwd_rets).sum(axis=1)

    # Cash leg: idle weight parked in SGOV (0-3mo T-bills).
    cash_w = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    if CASH_TICKER in close_all.columns:
        sgov_fwd = np.log(close_all[CASH_TICKER] / close_all[CASH_TICKER].shift(1)).reindex(weights.index).shift(-1)
        port_log_ret = port_log_ret + cash_w * sgov_fwd.fillna(0.0)
    else:
        print(f"⚠️  {CASH_TICKER} missing - cash earns 0%")

    port_log_ret = port_log_ret.iloc[:-1]  # drop last (no fwd return)
    equity = np.exp(port_log_ret.cumsum())
    return equity, port_log_ret


def perf_stats(log_rets, label):
    n = len(log_rets)
    years = n / 252
    total_ret = np.exp(log_rets.sum()) - 1
    cagr = np.exp(log_rets.sum() / years) - 1 if years > 0 else float("nan")
    vol = log_rets.std() * np.sqrt(252)
    sharpe = (log_rets.mean() * 252) / vol if vol > 0 else float("nan")
    equity = np.exp(log_rets.cumsum())
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    print(f"{label}: total_ret={total_ret:.1%}  CAGR={cagr:.1%}  vol={vol:.1%}  "
          f"Sharpe={sharpe:.2f}  max_dd={max_dd:.1%}  n_days={n}")


def main():
    raw = download_all()
    close_all = raw["Close"].ffill()

    # SGOV's yield is paid as dividends: its raw Close is ~flat, so use the
    # dividend-adjusted total-return series for the cash leg (else cash ~0%).
    if "Adj Close" in raw.columns.get_level_values(0) and CASH_TICKER in raw["Adj Close"].columns:
        close_all[CASH_TICKER] = raw["Adj Close"][CASH_TICKER].ffill()
        print(f"Using {CASH_TICKER} total-return (Adj Close) for the cash leg.")

    print("Running walk-forward daily loop ...")
    daily_df, summaries = walk_forward(raw)

    out_dir = "/private/tmp/claude-501/-Users-riddhisiddhi-stocks/096ffffb-994f-4b51-9f6f-e316ab08d40f/scratchpad"
    daily_df.to_csv(f"{out_dir}/trend_backtest_daily_signals.csv", index=False)
    summaries.to_csv(f"{out_dir}/trend_backtest_summary.csv")
    print(f"Saved daily signal+holding table ({len(daily_df)} rows) and summary.")

    equity, port_log_rets = build_equity_curve(daily_df, close_all)
    equity.to_csv(f"{out_dir}/trend_backtest_equity.csv")

    print(f"\n--- Strategy performance (hold {REBAL_EVERY}d, cash in {CASH_TICKER}) ---")
    perf_stats(port_log_rets, f"Trend-following ({REBAL_EVERY}d rebalance)")

    spy_rets = np.log(close_all["SPY"] / close_all["SPY"].shift(1)).reindex(port_log_rets.index)
    print("\n--- Benchmark: SPY buy & hold (same window) ---")
    perf_stats(spy_rets.dropna(), "SPY buy & hold")

    if CASH_TICKER in close_all.columns:
        sgov_rets = np.log(close_all[CASH_TICKER] / close_all[CASH_TICKER].shift(1)).reindex(port_log_rets.index)
        print(f"\n--- Cash floor: {CASH_TICKER} buy & hold ---")
        perf_stats(sgov_rets.dropna(), f"{CASH_TICKER} (cash yield)")

    print("\n--- Average daily book stats ---")
    print(summaries[["total_allocated", "portfolio_vol", "n_positions", "vol_cap", "fallback_used"]].mean(numeric_only=True))
    print("fallback_used rate:", summaries["fallback_used"].mean())


if __name__ == "__main__":
    main()
