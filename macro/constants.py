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
    "EWZ":"Brazil","INDA":"India","FXI":"China","EWU":"UK","EWG":"Germany"
}

# --- Commodity Mapping ---
COMMODITIES = {
    "DBC":"Broad Commodities","GC=F":"Gold","SI=F":"Silver","HG=F":"Copper",
    "ALI=F":"Aluminium","PL=F":"Platinum","PA=F":"Palladium","CL=F":"Crude Oil (WTI)",
    "BZ=F":"Brent Oil","NG=F":"Natural Gas","ZS=F":"Soybeans","ZC=F":"Corn",
    "ZW=F":"Wheat","KC=F":"Coffee","SB=F":"Sugar","CT=F":"Cotton","LE=F":"Live Cattle",
    "HE=F":"Lean Hogs","LBR=F":"Lumber"
}

# --- Currency Mapping ---
CURRENCIES = {
    "EURUSD=X":"EUR","JPY=X":"JPY","GBPUSD=X":"GBP","AUDUSD=X":"AUD", "USDINR=X":"INR"
}

# --- Trend Assets ---
TREND_ASSETS = {
    "VGT":"Tech","VDE":"Energy","VIS":"Industrials","XME":"Metals",
    "GLD":"Gold","IBIT":"Bitcoin","TLT":"30yr Bond", "RKT": "Rocket"
}

# --- ML Macro Tickers ---
ML_MACRO_TICKERS = ['DX-Y.NYB', '^VIX', '^TNX', '^MOVE', '^TYX', 'HYG', 'LQD']

