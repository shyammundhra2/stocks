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
    # US Equity Sectors
    "XLK": "Tech",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLV": "Healthcare",
    "XLF": "Financials",
    "XLY": "Discretionary",
    "XLP": "Staples",
    "XLU" : "Utilities",
    "XLRE" : "Real Estate",

    # Subsectors
    "SOXX": "Semiconductors",
    "IGV": "Software",
    "CIBR": "Cyber Security",
    "ITA": "Defense",

    # Commodities & Real Assets
    "GLD": "Gold",
    "SLV": "Silver",
    "XME": "Metals",
    "DBC": "Commodities",
    "MOO": "Agri",
    "FCG": "FCG Natural Gas Stocks",
    "COAL": "Coal Stocks",
    "CCJ": "Cameco",
    "COPX": "Copper Stocks",
    "WOOD" : "Lumber Stocks",

    # International Equities
    "INDA": "India",
    "EWY": "Korea",
    "EWJ": "Japan",
    "EWZ": "Brazil",
    "IEV": "Europe",

    # Fixed Income & Crypto
    "TLT": "30yr Bond",
    "IBIT": "Bitcoin",

    # Single Stocks
    "RKT": "Rocket",
}

# --- ML Macro Tickers ---
ML_MACRO_TICKERS = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD']

