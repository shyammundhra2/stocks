"""
Dubai residential price series: one monthly AED/sqm series, 2003-01 to present.

WHY A SPLICE. The only free, long, official Dubai price history is the BIS "detailed
residential property prices" flow, which runs 2003-01 .. 2024-04 and then stops. The
raw Dubai Land Department transaction microdata that would carry it forward lives on
dubaipulse.gov.ae, which is geo-blocked outside the UAE and needs an issued API key
besides. So the recent tail is rebuilt from published ValuStrat Price Index anchor
points and spliced on.

Over Dec-2023 -> Apr-2024, where both sources are observable, they imply +8.17% (BIS)
and +8.45% (VPI anchors) - 28bp apart over four months, so the join is close to
seamless. ``build_series`` asserts that gap stays small, and will fail loudly if a
future anchor revision breaks it.

HONESTY NOTE: everything after 2024-04 is interpolated between published anchors.
The annual and quarterly levels are real; the month-to-month wiggles inside a leg
are not observations. The ``source`` column marks which is which, and every consumer
should respect it.
"""
import os

import numpy as np
import pandas as pd
import requests

DIR = os.path.dirname(os.path.abspath(__file__))
BIS_CACHE = os.path.join(DIR, "bis_dubai_raw.csv")
SERIES_CSV = os.path.join(DIR, "dubai_price_monthly.csv")

# BIS SDMX. Key is FREQ.REF_AREA.COVERED_AREA -> M(onthly).AE.4, where COVERED_AREA
# 4 = Dubai (0 = 5 Emirates, 2 = Abu Dhabi). Returns the Dubai series alone, ~256 rows.
BIS_URL = "https://stats.bis.org/api/v1/data/BIS,WS_DPP,1.0/M.AE.4/all?format=csv"

JUNCTION = pd.Period("2024-04", "M")   # last BIS observation

# ValuStrat Price Index (Dubai residential, base Jan-2021 = 100), from published releases.
# Comments show the arithmetic that recovers each level from the published change.
VPI_ANCHORS = {
    "2023-12": 155.8,   # 238.0 / 1.198 (2025 +19.8%) / 1.275 (2024 +27.5%)
    "2024-12": 198.7,   # 238.0 / 1.198
    "2025-02": 207.3,   # 243.4 / 1.174  (Feb-26 printed +17.4% YoY)
    "2025-06": 219.8,   # 220.0 / 1.001  (Jun-26 printed +0.1% YoY)
    "2025-07": 222.8,   # 219.2 / 0.984  (Jul-26 printed -1.6% YoY)
    "2025-12": 238.0,   # implied by 2025 full-year +19.8% and the Feb-26 print
    "2026-02": 243.4,   # published level, +17.4% YoY  <- cycle peak
}
# Published month-on-month changes after the Feb-2026 peak (the drawdown leg). These
# are actual prints, not interpolation; they land on the published 220.0 and 219.2.
VPI_MOM = {
    "2026-03": -0.059,
    "2026-04": -0.019,
    "2026-05": -0.012,
    "2026-06": -0.010,
    "2026-07": -0.003,
}


def fetch_bis(refresh: bool = False) -> pd.Series:
    """Monthly Dubai price in AED/sqm from the BIS SDMX API, cached to disk."""
    if refresh or not os.path.exists(BIS_CACHE):
        r = requests.get(BIS_URL, timeout=120)
        r.raise_for_status()
        with open(BIS_CACHE, "w") as fh:
            fh.write(r.text)

    d = pd.read_csv(BIS_CACHE, low_memory=False)
    d = d[d["TITLE_GRP"].str.contains("Dubai", na=False)]
    d = d.sort_values("TIME_PERIOD")
    s = pd.Series(d["OBS_VALUE"].values.astype(float),
                  index=pd.PeriodIndex(d["TIME_PERIOD"], freq="M"), name="aed_sqm")
    if s.empty:
        raise RuntimeError("BIS returned no Dubai rows - check the WS_DPP key or TITLE_GRP")
    return s


def build_vpi() -> pd.Series:
    """Monthly VPI from 2023-12: log-linear between anchors, then the actual MoM prints.

    Log-linear (not linear) interpolation means a leg between two anchors compounds at a
    constant monthly rate, which is the right shape for a price index.
    """
    last_anchor = pd.Period(max(VPI_ANCHORS), "M")
    idx = pd.period_range(min(VPI_ANCHORS), last_anchor, freq="M")
    anchors = pd.Series({pd.Period(k, "M"): v for k, v in VPI_ANCHORS.items()})
    vpi = np.log(anchors).reindex(idx).interpolate(method="index").pipe(np.exp)

    for per, mom in sorted(VPI_MOM.items()):
        vpi.loc[pd.Period(per, "M")] = vpi.iloc[-1] * (1 + mom)
    return vpi.sort_index()


def build_series(refresh: bool = False) -> pd.DataFrame:
    """The spliced monthly series. Columns: aed_sqm (float), source ('BIS'|'VPI-spliced')."""
    bis, vpi = fetch_bis(refresh), build_vpi()

    # splice guard: the two sources must agree over the stretch where both are observable
    lead = pd.Period("2023-12", "M")
    bis_leg = bis.loc[JUNCTION] / bis.loc[lead] - 1
    vpi_leg = vpi.loc[JUNCTION] / vpi.loc[lead] - 1
    gap = abs(bis_leg - vpi_leg)
    if gap > 0.02:
        raise RuntimeError(
            f"splice mismatch {lead}->{JUNCTION}: BIS {bis_leg:+.2%} vs VPI {vpi_leg:+.2%} "
            f"({gap*1e4:.0f}bp). An anchor was probably revised - re-derive VPI_ANCHORS.")

    vpi_aed = vpi * (bis.loc[JUNCTION] / vpi.loc[JUNCTION])   # rescale onto AED/sqm
    px = pd.concat([bis, vpi_aed.loc[vpi_aed.index > JUNCTION]]).sort_index()
    px.name = "aed_sqm"

    src = pd.Series("BIS", index=px.index, name="source")
    src.loc[src.index > JUNCTION] = "VPI-spliced"
    return pd.DataFrame({"aed_sqm": px, "source": src})


def load(refresh: bool = False) -> pd.DataFrame:
    """Cached series off disk, rebuilding when missing or when ``refresh`` is set."""
    if refresh or not os.path.exists(SERIES_CSV):
        df = build_series(refresh=refresh)
        df.to_csv(SERIES_CSV)
        return df
    df = pd.read_csv(SERIES_CSV, index_col=0)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def main():
    df = build_series(refresh=True)
    df.to_csv(SERIES_CSV)
    px = df["aed_sqm"]

    print(f"series: {px.index[0]} .. {px.index[-1]}  ({len(px)} months)")
    print(f"  BIS official : {(df.source == 'BIS').sum()} months (through {JUNCTION})")
    print(f"  VPI spliced  : {(df.source != 'BIS').sum()} months (interpolated between anchors)")

    bis_leg = px.loc[JUNCTION] / px.loc[pd.Period('2023-12')] - 1
    vpi_leg = build_vpi().pipe(lambda v: v.loc[JUNCTION] / v.loc[pd.Period('2023-12')] - 1)
    print(f"  splice check : BIS {bis_leg:+.2%} vs VPI {vpi_leg:+.2%} "
          f"({abs(bis_leg-vpi_leg)*1e4:.0f}bp apart)")

    print("\nyear-end level (AED/sqm) and YoY:")
    ye = px.groupby(px.index.year).last()
    for y, v in ye.items():
        yoy = f"{v/ye[y-1]-1:+7.1%}" if y - 1 in ye else "      -"
        tag = "" if y <= JUNCTION.year else "  (spliced)"
        print(f"  {y}  {v:9,.0f}  {yoy}{tag}")
    print(f"\nwrote {SERIES_CSV}")


if __name__ == "__main__":
    main()
