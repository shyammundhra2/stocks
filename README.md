# Quantitative Risk & Trading Dashboard

A Flask-based financial analytics dashboard that automates technical analysis and risk regime identification. This tool integrates real-time data from Yahoo Finance to provide actionable signals across four distinct quantitative modules.

## 🛠️ System Architecture

The application is built on a modular logic engine that processes price data into higher-level market signals:

### 1. Risk Regime Module
Determines if the market is in a **RISK-ON** or **RISK-OFF** environment by evaluating three conditions:
* **Trend:** Is SPY trading above its 200-day Moving Average?
* **Fear:** Is the VIX Index below 20?
* **Breadth:** Is the Equal-Weight S&P 500 (RSP) outperforming the Cap-Weighted SPY (50-day relative strength)?

### 2. QQQ Tactical Mean Reversion
A high-velocity strategy targeting the Nasdaq-100 (QQQ) using a **2-period RSI** (Wilder's Smoothing):
* **Strong Buy:** RSI(2) ≤ 10 and price > 200MA.
* **Exit/Profit Take:** RSI(2) ≥ 70.
* **Risk Mitigation:** No new buys if price is below the 200-day Moving Average.

### 3. Sector Rotation Ranking
Calculates the 20-day relative momentum of all 11 S&P 500 sectors (XLF, XLK, XLE, etc.) against the SPY benchmark. It ranks them from strongest to weakest to identify where institutional capital is flowing.

### 4. Cross-Asset Trend Following
Monitors a diverse basket of assets (Tech, Energy, Gold, Bitcoin, Treasuries) using a multi-factor trend logic:
* **Buy Logic:** SMA 50 > SMA 200 + Low Volatility (ATR Filter) + RSI < 70.
* **Exit Logic:** Price drops below SMA 50 or hits a **10x ATR Trailing Stop** from recent highs.

## 🔧 Technical Stack
* **Framework:** Flask (Python)
* **Data Sourcing:** `yfinance` (Yahoo Finance API)
* **Data Processing:** `pandas`, `numpy` (Vectorized RSI and ATR calculations)
* **Frontend:** HTML/CSS via Jinja2 templates (`dashboard.html`)

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install flask yfinance pandas numpy
