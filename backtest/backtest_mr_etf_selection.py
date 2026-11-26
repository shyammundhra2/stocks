"""
Walk-forward test: does picking ETFs for RSI2 mean reversion from trailing
history beat (a) trading the whole ETF universe and (b) random picks?

Per-ETF MR rule = production RSI2 rule: long when RSI2<=10 and price>200SMA,
exit at RSI2>=70, flat below 200SMA. Idle capital earns BIL. Signals are lagged
one day. Every QUARTER, rank ETFs by a trailing statistic computed ONLY on data
up to that date, hold the top K equal-weight for the next quarter.
Selectors: trailing strategy Sharpe over 1y / 3y / 5y, and 5y lag-1 return
autocorrelation (structural: most negative = most reverting).
Baselines: all ETFs equal-weight, random-K (200 draws), SPY buy&hold.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from macro.helpers import compute_RSI

TICKERS = pd.read_csv("/Users/riddhisiddhi/stocks/trendselection/trend_etf_ranking.csv")["Ticker"].tolist()
K = 8
TEST_START = "2012-01-01"
COST_BPS = 5.0


def perf(r, label):
    r = pd.Series(r).dropna()
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    sh = r.mean() * 252 / vol if vol > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    print(f"{label:34s} CAGR={cagr:6.1%} vol={vol:5.1%} Sharpe={sh:5.2f} maxDD={dd:6.1%}")
    return sh


def etf_positions(px):
    rsi = compute_RSI(px, 2)
    sma = px.rolling(200, min_periods=200).mean()
    pos, cur = np.zeros(len(px)), 0.0
    for t in range(len(px)):
        if np.isnan(sma.iloc[t]) or px.iloc[t] < sma.iloc[t]:
            cur = 0.0
        elif rsi.iloc[t] <= 10:
            cur = 1.0
        elif rsi.iloc[t] >= 70:
            cur = 0.0
        pos[t] = cur
    return pd.Series(pos, index=px.index)


def main():
    raw = yf.download(TICKERS + ["SPY", "BIL"], start="2004-01-01", end="2026-11-20",
                      progress=False, auto_adjust=True)["Close"].ffill()
    cash = raw["BIL"].pct_change().fillna(0)
    spy = raw["SPY"]
    px = raw[[t for t in TICKERS if t in raw.columns]]
    px = px.loc[:, px.notna().sum() > 800]
    rets = px.pct_change()
    pos = pd.DataFrame({c: etf_positions(px[c].dropna()).reindex(px.index).fillna(0) for c in px})
    # strategy return per ETF: lagged position * ret + (1-pos)*cash
    strat = pos.shift(1) * rets + (1 - pos.shift(1)) * cash.values[:, None]
    strat = strat.where(px.notna())
    excess = strat.sub(cash, axis=0)
    ac1 = rets.rolling(1260, min_periods=900).apply(lambda x: pd.Series(x).autocorr(1), raw=False) \
        if False else None

    idx = px.index
    reb = pd.Series(idx, index=idx).groupby([idx.year, (idx.month - 1) // 3]).first().values
    reb = [pd.Timestamp(d) for d in reb if pd.Timestamp(d) >= pd.Timestamp(TEST_START)]
    rng = np.random.default_rng(0)

    def score(sel, d, lb):
        i = idx.get_loc(d)
        w = excess.iloc[max(0, i - lb):i]       # strictly before d
        if sel == "ac":
            r = rets.iloc[max(0, i - lb):i]
            return -r.apply(lambda s: s.autocorr(1) if s.count() > 400 else np.nan)
        m, s = w.mean(), w.std()
        return (m / s).where(w.count() > 0.8 * lb)

    methods = {"sharpe_1y": ("s", 252), "sharpe_3y": ("s", 756), "sharpe_5y": ("s", 1260),
               "autocorr_5y(structure)": ("ac", 1260)}
    out = {m: pd.Series(0.0, index=idx) for m in list(methods) + ["all_etfs", "random_K"]}
    rand_runs = np.zeros((200, len(idx)))
    prev = {m: set() for m in out}
    for n, d in enumerate(reb):
        end = reb[n + 1] if n + 1 < len(reb) else idx[-1] + pd.Timedelta(days=1)
        seg = idx[(idx >= d) & (idx < end)]
        avail = [c for c in px if px[c].loc[:d].count() > 300]
        for m, (kind, lb) in methods.items():
            sc = score(kind, d, lb).reindex(avail).dropna().sort_values(ascending=False)
            pick = set(sc.index[:K])
            r = strat.loc[seg, list(pick)].mean(axis=1) if pick else cash.loc[seg]
            turn = len(pick ^ prev[m]) / (2 * K)
            r = r.copy(); r.iloc[0] -= turn * COST_BPS / 1e4 * 2
            out[m].loc[seg] = r.values; prev[m] = pick
        out["all_etfs"].loc[seg] = strat.loc[seg, avail].mean(axis=1).values
        for k in range(200):
            p = rng.choice(avail, size=min(K, len(avail)), replace=False)
            rand_runs[k, idx.get_indexer(seg)] = strat.loc[seg, p].mean(axis=1).values

    mask = idx >= reb[0]
    print(f"\nTest {reb[0].date()} -> {idx[-1].date()}, K={K}, universe={px.shape[1]} ETFs\n")
    shs = {}
    for m in methods:
        shs[m] = perf(out[m][mask], m)
    shs["all"] = perf(out["all_etfs"][mask], "all ETFs equal-weight (no selection)")
    rs = np.array([perf_q(rand_runs[k][mask]) for k in range(200)])
    print(f"{'random-K picks (200 draws)':34s} Sharpe mean={rs.mean():5.2f}  5-95%=[{np.percentile(rs,5):.2f},{np.percentile(rs,95):.2f}]")
    perf(spy[mask].pct_change(), "SPY buy & hold")
    perf(cash[mask], "BIL (cash)")
    print("\nPercentile of each selector vs random-K draws:")
    for m in methods:
        print(f"  {m:26s} beats {(rs < shs[m]).mean():.0%} of random draws")


def perf_q(r):
    r = pd.Series(r)
    v = r.std() * np.sqrt(252)
    return r.mean() * 252 / v if v > 0 else np.nan


if __name__ == "__main__":
    main()
