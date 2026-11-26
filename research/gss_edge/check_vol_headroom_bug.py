"""Reproduce the suspected _vol_headroom_deploy bug WITHOUT touching live code: import the live function,
monkeypatch get_portfolio_summary (in-process only), feed real QQQ data from prices.csv."""
import sys, warnings; warnings.filterwarnings("ignore")
import pandas as pd
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
import macro.indicators.portfolio as pf
from macro.indicators.signals import _vol_headroom_deploy
P = pd.read_csv("/Users/riddhisiddhi/stocks/research/gss_edge/prices.csv", index_col=0, parse_dates=True).ffill(limit=5)
shared = {"Close": P[["SPY", "QQQ", "GLD"]].tail(300)}
def with_book(w):
    pf.get_portfolio_summary = lambda: {"weights": w}
    return _vol_headroom_deploy(shared)
print("empty book  {}                ->", with_book({}))
print("no summary  None-ish weights    ->", with_book(None))
print("all-cash book (zero weights)    ->", with_book({"GLD": 0.0}))
print("small book {'GLD':0.10}         ->", with_book({"GLD": 0.10}))
print("full book {'SPY':0.5,'GLD':0.5}->", with_book({"SPY": 0.5, "GLD": 0.5}))
