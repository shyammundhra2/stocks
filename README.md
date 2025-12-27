# Consolidated Macro Investing Dashboard (US Focus)

A professional-grade macro analysis tool built with Streamlit to track US economic health, liquidity, and market sentiment. This dashboard aggregates 16 key metrics to calculate a proprietary **RoRo (Risk-On/Risk-Off) Score**, helping investors determine the current market regime.

## 📊 Core Features

### 1. The RoRo Score
A weighted index ranging from **1-100** that signals market sentiment:
* **0-40:** Risk-Off (Defensive)
* **41-60:** Neutral
* **61-100:** Risk-On (Aggressive)

### 2. 16 Key Macro Metrics
The dashboard tracks data across eight critical categories:

* **Economic Growth:** Real GDP (Annualized), ISM Manufacturing PMI, Bank Loan Growth, and New Construction Spending.
* **Labor Market:** Unemployment Rate.
* **Inflation:** Consumer Price Index (CPI) YoY.
* **Monetary Policy & Risk:** Fed Funds Rate, 10Y-2Y Yield Spread, VIX Index, MOVE Index, and High Yield OAS.
* **Liquidity:** M2 Money Supply YoY.
* **Carry Trade & FX:** USD/JPY Interest Rate Differential, Realized Volatility, and CFTC JPY Net Positions.
* **Market Breadth:** % of S&P 500 stocks above their 200-Day Moving Average.
* **Sector Strength:** Real-time tracking of the 3 strongest and 3 weakest US sectors.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Sourcing:** `yfinance` (Market data), `pandas-datareader` (FRED/Economic data)
* **Analysis:** `pandas`, `numpy`
* **Visualization:** `plotly`, `matplotlib`

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/shyammundhra2/stocks.git](https://github.com/shyammundhra2/stocks.git)
   cd stocks
