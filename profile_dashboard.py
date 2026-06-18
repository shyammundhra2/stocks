"""
GSS dashboard load profiler.

Run this in the REAL environment (real yfinance, real models, real network):

    python3 profile_dashboard.py

It answers three questions, in order:
  A. How long is the network download itself?
  B. With data already cached, how long does each dashboard panel take to
     COMPUTE? (ttl cache bypassed via __wrapped__, so this is real recompute,
     not a 30s cache hit.)
  C. Across one full render, which functions dominate cumulative time?

Paste the output back. The three sections together pinpoint whether the
remaining slowness is I/O (download), a specific panel's compute, model
loading, or the HMM fit.
"""
import time
import io
import cProfile
import pstats

import macro.indicators as engine

# functions called to build the dashboard (adjust if your route differs)
PANELS = [
    "get_risk_regime",
    "get_regime_hmm_state",
    "get_trends",
    "get_ml_sector_prediction",
    "get_ml_country_prediction",
    "get_ml_commodity_prediction",
    "get_vix_signal",
    "get_mean_reversion",
    "get_sector_rotation",
    "get_country_rotation",
    "get_commodity_rotation",
    "get_currency_rotation",
]


def raw(name):
    """Underlying function with the ttl_cache removed, so it actually recomputes."""
    f = getattr(engine, name, None)
    if f is None:
        return None
    return getattr(f, "__wrapped__", f)


def timed(fn, *a, **k):
    t = time.perf_counter()
    try:
        fn(*a, **k)
        ok = ""
    except Exception as e:
        ok = f"  ERROR: {e}"
    return (time.perf_counter() - t), ok


print("=" * 64)
print("A. DOWNLOAD COST (warm the store; measure the network leg)")
print("=" * 64)

# First touch pays the cold download once and fills the store.
t = time.perf_counter()
try:
    engine._get_shared_market_data()
    engine._get_extended_data()
    cold = time.perf_counter() - t
    print(f"  cold fill (shared 1y + extended 300d), incl. download: {cold:7.2f}s")
except Exception as e:
    print(f"  cold fill ERROR: {e}")

# Now the store is warm. A direct download of the same universe isolates
# the pure network leg (this is what the store now SKIPS on warm loads).
try:
    import yfinance as yf
    uni = sorted(set(
        list(engine.SECTORS.keys()) + list(engine.COUNTRIES.keys()) +
        list(engine.COMMODITIES.keys()) + list(engine.CURRENCIES.keys()) +
        list(engine.TREND_ASSETS.keys()) + list(engine.SECTOR_NAMES.keys()) +
        ['SPY', 'RSP', '^VIX', '^MOVE', 'HYG', 'IEF', '^TNX', '^IRX',
         'JPY=X', 'HG=F', 'GC=F', 'QQQ']))
    t = time.perf_counter()
    yf.download(uni, period="1y", progress=False, auto_adjust=False,
                threads=True, group_by="column")
    print(f"  raw yf.download of {len(uni)} tickers, 1y:               "
          f"{time.perf_counter() - t:7.2f}s   <-- skipped on warm loads now")
except Exception as e:
    print(f"  raw download probe ERROR: {e}")

print()
print("=" * 64)
print("B. PER-PANEL COMPUTE (store warm, ttl bypassed = real recompute)")
print("=" * 64)
rows = []
for name in PANELS:
    fn = raw(name)
    if fn is None:
        continue
    dt, err = timed(fn)
    rows.append((dt, name, err))
for dt, name, err in sorted(rows, reverse=True):
    print(f"  {dt:7.2f}s  {name}{err}")
print(f"  {'-'*7}")
print(f"  {sum(r[0] for r in rows):7.2f}s  TOTAL (serial sum)")

print()
print("=" * 64)
print("C. cPROFILE OF ONE FULL RENDER (top 25 by cumulative time)")
print("=" * 64)
pr = cProfile.Profile()
pr.enable()
for name in PANELS:
    fn = raw(name)
    if fn:
        try:
            fn()
        except Exception:
            pass
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
# trim pstats header noise
out = s.getvalue().splitlines()
for line in out[:33]:
    print(line)

print()
print("Read it like this:")
print("  - If A's raw download dominates and B is small -> it was I/O; warm")
print("    loads are already fast, only the FIRST load after restart is slow")
print("    (fix: warm the store on startup).")
print("  - If B has one panel >> others -> that panel's compute is the cost.")
print("  - If C shows hmmlearn/_fit_regime_hmm high -> the HMM fit; persist it.")
print("  - If C shows predict_proba / feature builders high -> model inference")
print("    or the commodity/country rolling loops.")
