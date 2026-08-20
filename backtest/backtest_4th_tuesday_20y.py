"""
20-year backtest of the trend-following pipeline, rebalancing only on the
4th Tuesday of each month. Reuses compute_regime/compute_asset_row from
backtest_trend_following.py for exact fidelity with the production signal
and Kelly-sizing logic (including this session's ml_conf_slow gate fix).

"Ignore if an ETF didn't exist in a particular month": on each rebalance
date, an asset is only included in that month's universe if it has at
least MIN_HISTORY_DAYS of trailing price history. Younger ETFs (e.g. IBIT,
inception Jan 2024) simply aren't eligible until they cross that bar -
same effect as the len(c)<60 gate in compute_asset_row, but applied
before an asset even enters the optimizer's candidate list, so the
portfolio's opportunity set grows over the 20 years as ETFs listed.

Only ~240 rebalance dates get the expensive per-asset signal + SLSQP
optimizer treatment; all other trading days just need forward returns
for the frozen weights, which is cheap.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.constants import TREND_ASSETS
from macro.indicators import get_regime_scalar, _kelly_covariance_optimizer
from backtest_trend_following import compute_regime, compute_asset_row

END = pd.Timestamp("2026-07-10")
START = END - pd.DateOffset(years=20)
DOWNLOAD_START = pd.Timestamp("1998-01-01")  # as far back as useful ETF history goes
MIN_HISTORY_DAYS = 210  # ~200 trading days for the 200MA + margin

PORTFOLIO_VALUE = 500_000
REGIME_TICKERS = ["HYG", "IEF", "^TNX", "^IRX", "JPY=X", "RSP", "^VIX", "^MOVE"]
TREND_SYMS = list(TREND_ASSETS.keys())
ALL_TICKERS = sorted(set(TREND_SYMS) | set(REGIME_TICKERS))


def download_all():
    print(f"Downloading {len(ALL_TICKERS)} tickers from {DOWNLOAD_START.date()} to {END.date()} ...")
    raw = yf.download(
        ALL_TICKERS, start=DOWNLOAD_START, end=END + pd.Timedelta(days=1),
        progress=False, auto_adjust=False, threads=True, group_by="column",
    )
    return raw


def fourth_tuesdays(trading_days, start, end):
    tuesdays = trading_days[(trading_days.dayofweek == 1) & (trading_days >= start) & (trading_days <= end)]
    month_key = tuesdays.to_period("M")
    fourth = tuesdays.to_series().groupby(month_key).nth(3)
    return pd.DatetimeIndex(sorted(fourth.values))


def run_rebalance(raw, rebal_date):
    shared_t = raw.loc[:rebal_date]
    close_t = raw["Close"].ffill().loc[:rebal_date]

    regime_cols = [c for c in REGIME_TICKERS + ["SPY"] if c in close_t.columns]
    regime = compute_regime(close_t[regime_cols].dropna(how="all"))
    regime_scalar = get_regime_scalar(regime)

    trends = []
    for sym in TREND_SYMS:
        try:
            df_sym = shared_t.xs(sym, level=1, axis=1).dropna()
        except KeyError:
            continue
        if len(df_sym) < MIN_HISTORY_DAYS:
            continue  # ETF didn't exist / doesn't have enough history yet
        row = compute_asset_row(sym, df_sym)
        if row is not None:
            trends.append(row)

    if not trends:
        return {}, {}

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
    weights = {item["sym"]: item.get("weight_pct", 0.0) / 100.0 for item in optimized}
    return weights, summary


def main():
    raw = download_all()
    close_all = raw["Close"].ffill()
    trading_days = close_all.index

    rebal_dates = fourth_tuesdays(trading_days, START, END)
    print(f"Found {len(rebal_dates)} 4th-Tuesday rebalance dates from {rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    t0 = time.time()
    weight_rows = {}
    n_assets_used = []
    for i, dt in enumerate(rebal_dates):
        weights, summary = run_rebalance(raw, dt)
        weight_rows[dt] = weights
        n_assets_used.append(len(weights))
        if (i + 1) % 24 == 0:
            print(f"  {i+1}/{len(rebal_dates)} rebalances ({time.time()-t0:.0f}s elapsed)")

    weights_df = pd.DataFrame.from_dict(weight_rows, orient="index").fillna(0.0)
    weights_df = weights_df.reindex(columns=sorted(set().union(*[set(w) for w in weight_rows.values()])))
    weights_df.to_csv("/private/tmp/claude-501/-Users-riddhisiddhi-stocks/096ffffb-994f-4b51-9f6f-e316ab08d40f/scratchpad/tuesday20y_rebal_weights.csv")

    # forward-fill weights across all trading days between rebalances
    test_days = trading_days[(trading_days >= rebal_dates[0]) & (trading_days <= END)]
    weights_daily = weights_df.reindex(test_days, method="ffill").fillna(0.0)

    rets = np.log(close_all / close_all.shift(1)).reindex(test_days)[weights_daily.columns]
    fwd_rets = rets.shift(-1)
    port_log_ret = np.log1p((weights_daily * (np.exp(fwd_rets) - 1)).sum(axis=1)).iloc[:-1]
    port_log_ret.to_csv("/private/tmp/claude-501/-Users-riddhisiddhi-stocks/096ffffb-994f-4b51-9f6f-e316ab08d40f/scratchpad/tuesday20y_logrets.csv")

    irx = close_all["^IRX"].reindex(port_log_ret.index).ffill()
    cash_daily_simple = (irx / 100.0) / 252.0
    cash_weight = (1.0 - weights_daily.sum(axis=1)).clip(lower=0).reindex(port_log_ret.index)
    blended_simple = (np.exp(port_log_ret) - 1) + cash_weight * cash_daily_simple
    blended_log = np.log1p(blended_simple).dropna()

    spy_log = np.log(close_all["SPY"] / close_all["SPY"].shift(1)).reindex(port_log_ret.index).dropna()

    def stats(log_r, label):
        n = len(log_r)
        years = n / 252
        total_ret = np.exp(log_r.sum()) - 1
        cagr = np.exp(log_r.sum() / years) - 1
        vol = log_r.std() * np.sqrt(252)
        sharpe = (log_r.mean() * 252) / vol if vol > 0 else float("nan")
        eq = np.exp(log_r.cumsum())
        dd = (eq / eq.cummax() - 1).min()
        downside = log_r[log_r < 0]
        ddev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        sortino = (log_r.mean() * 252) / ddev if ddev > 0 else float("nan")
        print(f"{label}: n_days={n} years={years:.1f} total={total_ret:.1%} CAGR={cagr:.1%} "
              f"vol={vol:.1%} Sharpe={sharpe:.2f} Sortino={sortino:.2f} max_dd={dd:.1%}")

    print("\n--- Performance (20y, 4th Tuesday rebalance) ---")
    stats(port_log_ret, "Strategy, no cash yield")
    stats(blended_log, "Strategy + T-bill on idle cash")
    stats(spy_log, "SPY buy & hold (same window)")

    print("\n--- Universe growth (# ETFs eligible per rebalance) ---")
    n_series = pd.Series(n_assets_used, index=rebal_dates)
    print(n_series.resample("YE").mean().round(1))

    eqfull = np.exp(port_log_ret.cumsum())
    monthly_ret = eqfull.resample("ME").last().pct_change().dropna()
    print("\nBest months:"); print((monthly_ret.sort_values(ascending=False).head(5) * 100).round(2))
    print("Worst months:"); print((monthly_ret.sort_values().head(5) * 100).round(2))

    print("\nSaved: tuesday20y_rebal_weights.csv, tuesday20y_logrets.csv")


if __name__ == "__main__":
    main()
