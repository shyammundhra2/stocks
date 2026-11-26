"""Old inline get_vix_signal (from git HEAD) vs new capitulation_state: must be identical on historical slices."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/riddhisiddhi/stocks")
import pandas as pd
from macro.indicators import _signals_old_tmp as old, signals as new
O = pd.read_pickle("/Users/riddhisiddhi/stocks/research/gss_edge/ohlcv.pkl")["Close"]
bad = 0; dates = O.index[400::130][:40]
for d in dates:
    sub = O.loc[:d]; vix, spy = sub["^VIX"].dropna(), sub["SPY"].dropna()
    old._get_shared_market_data = lambda: None; old._get_close = lambda sd, t: vix if t == "^VIX" else spy
    a = old.get_vix_signal.__wrapped__() if hasattr(old.get_vix_signal, "__wrapped__") else old.get_vix_signal()
    b = new.capitulation_state(vix, spy)
    if a != b: bad += 1; print("MISMATCH", d.date(), a, b)
print("checked", len(dates), "dates; mismatches:", bad)
