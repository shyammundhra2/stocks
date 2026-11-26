"""
Robustness of the autocorrelation selector from backtest_mr_etf_selection.py.
Grid over K and lookback, on two universes:
  A) the momentum-ranked trend universe (survivorship-biased)
  B) a fixed, performance-blind list of liquid ETFs that all existed by 2007
Each cell: Sharpe of top-K most-negative lag-1-autocorr ETFs (quarterly reselect,
RSI2 rule per ETF, idle in BIL, 5bps), vs all-ETF equal-weight and 200 random-K.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "/Users/riddhisiddhi/stocks")
from backtest.backtest_mr_etf_selection import etf_positions

A = pd.read_csv("/Users/riddhisiddhi/stocks/trendselection/trend_etf_ranking.csv")["Ticker"].tolist()
B = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU",
     "XLV", "XLY", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO", "VNQ", "EWJ", "EWZ", "EWG",
     "EWY", "EWT", "EWH", "EWA", "EWC", "MDY", "IJR", "IWD", "IWF", "SHY", "AGG", "DBC"]
TEST_START, COST = "2012-01-01", 5.0


def sharpe(r):
    r = pd.Series(r).dropna()
    v = r.std() * np.sqrt(252)
    return r.mean() * 252 / v if v > 0 else np.nan


def run(tickers, name, raw):
    cash = raw["BIL"].pct_change().fillna(0)
    px = raw[[t for t in tickers if t in raw.columns]]
    px = px.loc[:, px.notna().sum() > 800]
    rets = px.pct_change()
    pos = pd.DataFrame({c: etf_positions(px[c].dropna()).reindex(px.index).fillna(0) for c in px})
    strat = (pos.shift(1) * rets + (1 - pos.shift(1)) * cash.values[:, None]).where(px.notna())
    idx = px.index
    reb = pd.Series(idx, index=idx).groupby([idx.year, (idx.month - 1) // 3]).first().values
    reb = [pd.Timestamp(d) for d in reb if pd.Timestamp(d) >= pd.Timestamp(TEST_START)]
    mask = idx >= reb[0]
    ac = {lb: rets.rolling(lb, min_periods=int(lb * .8)).corr(rets.shift(1)).shift(1)
          for lb in (504, 756, 1260)}
    rng = np.random.default_rng(0)
    print(f"\n=== Universe {name}: {px.shape[1]} ETFs, test {reb[0].date()}+ ===")
    segs = [(d, reb[n + 1] if n + 1 < len(reb) else idx[-1] + pd.Timedelta(days=1)) for n, d in enumerate(reb)]
    allr = pd.Series(0.0, index=idx)
    for d, e in segs:
        seg = idx[(idx >= d) & (idx < e)]
        av = [c for c in px if px[c].loc[:d].count() > 300]
        allr.loc[seg] = strat.loc[seg, av].mean(axis=1).values
    print(f"all-ETF equal-weight Sharpe={sharpe(allr[mask]):.2f}")
    print("K  lookback  Sharpe  maxDD   vs-random-pctile")
    for K in (4, 8, 12, 20):
        rs = []
        for k in range(100):
            r = pd.Series(0.0, index=idx)
            for d, e in segs:
                seg = idx[(idx >= d) & (idx < e)]
                av = [c for c in px if px[c].loc[:d].count() > 300]
                p = rng.choice(av, size=min(K, len(av)), replace=False)
                r.loc[seg] = strat.loc[seg, p].mean(axis=1).values
            rs.append(sharpe(r[mask]))
        rs = np.array(rs)
        for lb in (504, 756, 1260):
            r, prev = pd.Series(0.0, index=idx), set()
            for d, e in segs:
                seg = idx[(idx >= d) & (idx < e)]
                av = [c for c in px if px[c].loc[:d].count() > 300]
                sc = ac[lb].loc[d].reindex(av).dropna().sort_values()
                pick = set(sc.index[:K])
                x = strat.loc[seg, list(pick)].mean(axis=1).copy()
                x.iloc[0] -= len(pick ^ prev) / (2 * max(K, 1)) * COST / 1e4 * 2
                r.loc[seg] = x.values; prev = pick
            eq = (1 + r[mask]).cumprod()
            print(f"{K:<3d}{lb/252:>5.0f}y   {sharpe(r[mask]):6.2f} {(eq/eq.cummax()-1).min():7.1%}   {(rs < sharpe(r[mask])).mean():.0%}")


if __name__ == "__main__":
    raw = yf.download(sorted(set(A + B)) + ["BIL"], start="2004-01-01", end="2026-11-20",
                      progress=False, auto_adjust=True)["Close"].ffill()
    run(A, "A momentum-ranked", raw)
    run(B, "B performance-blind", raw)
