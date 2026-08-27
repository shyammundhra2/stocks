# macro/constants.py

# --- Sector Mapping ---
SECTOR_NAMES = {
    'XLE':'Energy', 'XLB':'Materials', 'XLI':'Industrials',
    'XLY':'Discr', 'XLF':'Financials', 'XLC':'Comm Serv', 
    'XLK':'Tech', 'XLV':'Health', 'XLP':'Staples',
    'XLU':'Utilities', 'XLRE':'Real Estate'
}

SECTORS = {
    "XLC":"Comm Serv","XLY":"Discr","XLP":"Staples","XLE":"Energy",
    "XLF":"Financials","XLV":"Health","XLI":"Industrials","XLB":"Materials",
    "XLRE":"Real Estate","XLK":"Tech","XLU":"Utilities"
}

# --- Country Mapping ---
COUNTRIES = {
    "SPY":"USA","EFA":"Dev ex-US","EEM":"Emerging","EWJ":"Japan",
    "EWZ":"Brazil","INDA":"India","FXI":"China","EWU":"UK","EWG":"Germany", "EWY":  "South Korea"

}

# --- Updated Commodity Mapping ---
COMMODITIES = {
    "DBC": "Broad Commodities", "GC=F": "Gold", "SI=F": "Silver", "HG=F": "Copper",
    "ALI=F": "Aluminium", "PL=F": "Platinum", "PA=F": "Palladium", "CL=F": "Crude Oil (WTI)",
    "BZ=F": "Brent Oil", "NG=F": "Natural Gas", "ZS=F": "Soybeans", "ZC=F": "Corn",
    "ZW=F": "Wheat", "KC=F": "Coffee", "SB=F": "Sugar", "CT=F": "Cotton", "LE=F": "Live Cattle",
    "HE=F": "Lean Hogs", "LBR=F": "Lumber",
    "TIO=F": "Iron Ore (62% FE CFR)",
    "HRC=F": "U.S. Midwest Domestic Hot-Rolled Steel"
    #"U-UN.TO": "Uranium"
}
# --- Currency Mapping ---
CURRENCIES = {
    "EURUSD=X":"EUR","JPY=X":"JPY","GBPUSD=X":"GBP","AUDUSD=X":"AUD", "USDINR=X":"INR"
}

# --- Trend Assets ---
TREND_ASSETS = {
    #Index
    "SPY" : "SP 500",
    "QQQ" : "Nasdaq 100",
    "IWM" : "Russel 2000",
    # VLUE: US value-factor tilt - top scanner trender (+64% 12-1, R2 0.81, at
    # 52wk high), and a factor uncorrelated-ish to the growth/semi book. 2013+,
    # ~$280M/day.
    "VLUE": "US Value Factor",
    # US Equity Sectors
    "XLK": "Tech",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLP": "Staples",
    "XLRE" : "Real Estate",

    # Subsectors
    "SOXX": "Semiconductors",
    # SMH: 2nd semi vehicle (VanEck) alongside SOXX - deliberately thickens the
    # semiconductor tilt for the current commodity+semi cycle thesis. More
    # NVDA/TSMC-concentrated than SOXX. History to 2005, highly liquid.
    # (DRAM / Roundhill Memory ETF is on-thesis but too new - launched Apr 2026,
    #  <200d history - so it can't clear the 200DMA gate yet; add ~early 2027.)
    "SMH": "Semiconductors (VanEck)",
    "IGV": "Software",
    "CIBR": "Cyber Security",
    "ITA": "Defense",
    # XBI: equal-weight biotech - top scanner trender (+68% 12-1, R2 0.81, at
    # 52wk high). A distinct healthcare subsector, diversifying vs commodity+semi.
    # Deep liquidity ($1.3B/day), history to 2010.
    "XBI": "Biotech",
    # 2026-08-25: reverted the 15-name breadth batch, then re-added ONLY the
    # names that improve full-deployment 2020-26 CAGR (backtest_gss_addwinners):
    # FDN +0.19% and SIL +0.86% are ROBUST (help 2011-19 too); XOP +0.79% and
    # REMX +1.21% are regime-specific commodity/energy bets (flat/neg 2011-19).
    # The other 11 (KRE -3.83% worst) HURT CAGR and were dropped. Universe 40.
    # (Return-max posture: at full deployment, fewer/better names concentrate;
    # on the throttled config the wider set was better - see 36_vs_51.)
    "FDN": "Internet",
    "SIL": "Silver Miners",
    "XOP": "Oil & Gas E&P",

    # Commodities & Real Assets
    "GLD": "Gold",
    "XME": "Metals",
    "DBC": "Commodities",
    "MOO": "Agri",
    "FCG": "FCG Natural Gas Stocks",
    # (throttle config for the names below is at THROTTLED_ASSETS, end of file)
    # MLPX: midstream energy / MLPs - fills the pipeline/infrastructure gap
    # (book had upstream XOP/FCG but no midstream). Low corr (~0.4 to semis, ~0.35
    # to broad market) = genuine diversification for the concentrated commodity+
    # semi book. Global X ETF, history to 2013. (CPER/copper skipped - COPX+SLV
    # already proxy metals per user.)
    "MLPX": "Midstream Energy (MLP)",
    "COAL": "Coal Stocks",
    "URNM": "Uranium Stocks",
    "COPX": "Copper Stocks",
    "SLX": "Steel Stocks",
    "WOOD" : "Lumber Stocks",
    # (SIL/REMX kept above with the CAGR-improvers; LIT -0.95% and GNR +0.08%
    #  were dropped 2026-08-25 - LIT reduces full-deploy CAGR, GNR is noise.)

    # International Equities
    "EWY": "Korea",
    "EWJ": "Japan",
    "EWZ": "Brazil",
    "IEV": "Europe",

    # Fixed Income & Crypto
    "IBIT": "Bitcoin",

    # Diversifier / crisis hedge: managed futures (trend-following, uncorrelated
    # -0.04 to the book). Validated overlay 2019-26 (dbmf_overlay.py): adds Sharpe
    # 1.22->1.41 @15%, shallows maxDD -8.1%->-6.9%, helps BOTH 2020 & 2022.
    # Momentum-timed here (held when trending, dropped in chop) - captures the
    # crisis-alpha AND sidesteps managed-futures' flat-period bleed (2011-19).
    # In DEFENSIVE_ASSETS (below) so the risk-off gate/throttle don't shrink it
    # when it's most needed. Caveat: short live history (2019+), best-run sample.
    "DBMF": "Managed Futures",

    # Single Stocks
    "RKT": "Rocket",
}

# Market-structure names: kept in the universe for the trend map's regional /
# rates read (they show full slope/R²/signal), but THROTTLED in live sizing so
# their whipsaw can't damage the book. Backtests flagged countries as whipsaw-
# prone (Sharpe 0.84->0.67) and long bonds as return-dilutive (defense only), so
# rather than delete them we cap their allocation to THROTTLE_FACTOR x the
# optimizer's weight; the freed capital simply falls to cash. EWY is left at full
# weight (Samsung = semiconductor supply chain, on-thesis, not display-only).
THROTTLED_ASSETS = {"EWZ", "EWJ", "IEV"}
THROTTLE_FACTOR = 0.30

# Names NOT zeroed by the risk-off equity gate (backtest_gss_eqgate 2007-26:
# gating these too destroyed the 2008 crisis alpha - TLT/GLD were the book's
# best positions while equities collapsed). They still need their own >200DMA
# trend to be bought; the gate only spares them from the SPY-based shutoff.
DEFENSIVE_ASSETS = {"GLD", "DBC", "DBMF"}

# --- ML Macro Tickers ---
ML_MACRO_TICKERS = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD']

