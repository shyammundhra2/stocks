"""
Refresh the S&P 500 universe AND its GICS sector map together, atomically.

WHY BOTH AT ONCE. sp500.csv and sp500_sectors.csv are not independent files -
they are one invariant. predict_top_stocks.load_sectors() falls back to using a
ticker as its OWN sector when it is missing from the map ("Missing names get
their own ticker as 'sector' so they're never capped away"), which means any
symbol present in the universe but absent from the sector map silently BYPASSES
the <=3-per-sector cap. Refreshing the symbol list alone therefore makes
diversification worse, not better: newly added names would each count as a
sector of one, and the book could hold six semis wearing a diversification
badge. This script refuses to write unless every symbol has a sector.

(As of 2026-09-03 that had already drifted: the universe was last refreshed in
June, 29 names differed from the live index, and the 14 symbols missing from the
sector map were exactly the 14 stale ones. Three current members absent from the
local list - CIEN, COHR, MRVL - out-ranked the book's 10th holding on 12-1
momentum, so the staleness was actively costing selections.)

Sources:
    symbols  - slickcharts.com/sp500  (same source the old fetch_sp500.py used)
    sectors  - Wikipedia's List_of_S&P_500_companies GICS Sector column
Both are normalised to yfinance ticker form (dots -> dashes) before comparison.

Replaces fetch_sp500.py, which wrote only the symbol list, and wrote it relative
to the current working directory.
"""
import io
import os
import sys

import pandas as pd
import requests

DIR = os.path.dirname(os.path.abspath(__file__))
SYMBOLS_CSV = os.path.join(DIR, "sp500.csv")
SECTORS_CSV = os.path.join(DIR, "sp500_sectors.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SLICKCHARTS = "https://www.slickcharts.com/sp500"
WIKIPEDIA = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

MIN_NAMES = 480          # sanity floor; the index is ~503


def _norm(s):
    return s.astype(str).str.strip().str.replace(".", "-", regex=False)


def fetch_symbols() -> pd.DataFrame:
    r = requests.get(SLICKCHARTS, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    df["Symbol"] = _norm(df["Symbol"])
    out = df[["Symbol", "Company"]].drop_duplicates("Symbol").reset_index(drop=True)
    if len(out) < MIN_NAMES:
        raise RuntimeError(f"slickcharts returned only {len(out)} names - refusing to write")
    return out


def fetch_sectors() -> pd.DataFrame:
    r = requests.get(WIKIPEDIA, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    df = pd.read_html(io.StringIO(r.text))[0]
    col = next((c for c in df.columns if "GICS" in str(c) and "Sub" not in str(c)), None)
    if col is None:
        raise RuntimeError(f"no GICS Sector column found; got {list(df.columns)}")
    df = df.rename(columns={col: "Sector"})
    df["Symbol"] = _norm(df["Symbol"])
    out = df[["Symbol", "Sector"]].drop_duplicates("Symbol").reset_index(drop=True)
    if len(out) < MIN_NAMES:
        raise RuntimeError(f"wikipedia returned only {len(out)} names - refusing to write")
    return out


def main():
    print("fetching symbols (slickcharts) ...")
    syms = fetch_symbols()
    print(f"  {len(syms)} symbols")
    print("fetching GICS sectors (wikipedia) ...")
    secs = fetch_sectors()
    print(f"  {len(secs)} sector rows")

    # THE INVARIANT: every symbol in the universe must have a sector.
    smap = dict(zip(secs["Symbol"], secs["Sector"]))
    missing = sorted(set(syms["Symbol"]) - set(smap))
    if missing:
        print(f"\nREFUSING TO WRITE: {len(missing)} symbol(s) have no GICS sector:")
        print("  " + ", ".join(missing))
        print("\nWriting anyway would let these bypass the <=3-per-sector cap entirely,")
        print("since load_sectors() treats an unmapped ticker as its own sector.")
        print("Fix the sector source (or add a manual override) and re-run.")
        return 1

    # keep the sector file aligned to the universe, in the same order
    secs_aligned = pd.DataFrame({"Symbol": syms["Symbol"],
                                 "Sector": [smap[s] for s in syms["Symbol"]]})

    # report the drift before overwriting
    for path, new, key in ((SYMBOLS_CSV, syms, "Symbol"), (SECTORS_CSV, secs_aligned, "Symbol")):
        name = os.path.basename(path)
        if os.path.exists(path):
            old = pd.read_csv(path)
            a, b = set(old[key]), set(new[key])
            print(f"\n{name}: {len(old)} -> {len(new)}")
            if a - b:
                print(f"  removed ({len(a-b)}): {', '.join(sorted(a-b))}")
            if b - a:
                print(f"  added   ({len(b-a)}): {', '.join(sorted(b-a))}")
            if not (a - b) and not (b - a):
                print("  no membership change")
        else:
            print(f"\n{name}: creating with {len(new)} rows")

    syms.to_csv(SYMBOLS_CSV, index=False)
    secs_aligned.to_csv(SECTORS_CSV, index=False)
    print(f"\nwrote {SYMBOLS_CSV}")
    print(f"wrote {SECTORS_CSV}")
    print(f"\nsector mix: {secs_aligned['Sector'].value_counts().to_dict()}")
    print("\nre-run the ranker to pick these up:  python -m stockselection.predict_top_stocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
