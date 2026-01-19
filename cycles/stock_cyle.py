import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def analyze_sp500_cycles():
    # 1. Download Full S&P 500 History (Back to 1927)
    ticker = "^GSPC"
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, start="1927-01-01")

    # Fix for potential yfinance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Resample to Monthly for Macro Cycle Analysis
    # We use log prices to calculate percentage deviations
    df_m = df['Close'].resample('ME').last().to_frame()
    df_m['Log_Price'] = np.log(df_m['Close'])

    # 3. Step 1: Identify Cycle Length (Spectral Analysis)
    # Using the HP Filter to isolate the 'cyclical' heartbeat
    # Lambda=129600 for monthly data
    hpi_cycle, hpi_trend = hpfilter(df_m['Log_Price'], lamb=129600)
    df_m['Cycle_Dev'] = hpi_cycle

    # Perform Periodogram to find the dominant frequency
    fs = 12 # Monthly sampling
    frequencies, power_density = periodogram(df_m['Cycle_Dev'].dropna(), fs=fs)
    periods = 1 / frequencies[frequencies > 0]
    psd = power_density[frequencies > 0]

    # Search for the dominant cycle between 2 and 15 years
    mask = (periods >= 2) & (periods <= 15)
    peak_cycle = periods[mask][np.argmax(psd[mask])]

    # 4. Step 2: Identify Regimes (Machine Learning - GMM)
    df_m['Log_Return'] = df_m['Log_Price'].diff()
    df_m['Volatility'] = df_m['Log_Return'].rolling(window=12).std()
    df_ml = df_m.dropna().copy()

    # Scale features: Returns (Momentum) and Volatility (Risk)
    X = StandardScaler().fit_transform(df_ml[['Log_Return', 'Volatility']])
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(X)
    df_ml['Regime_ID'] = gmm.predict(X)

    # Label clusters based on historical performance
    means = df_ml.groupby('Regime_ID')['Log_Return'].mean()
    mapping = {means.idxmax(): 'Bull (Expansion)', 
               means.idxmin(): 'Bear (Contraction)'}
    for i in range(3):
        if i not in mapping: mapping[i] = 'Neutral/Transition'
    df_ml['Regime'] = df_ml['Regime_ID'].map(mapping)

    # 5. Comprehensive Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top Plot: Log Price colored by ML Regime
    colors = {'Bull (Expansion)': '#2ca02c', 'Neutral/Transition': '#ff7f0e', 'Bear (Contraction)': '#d62728'}
    for label, color in colors.items():
        mask = df_ml['Regime'] == label
        ax1.scatter(df_ml.index[mask], df_ml['Close'][mask], color=color, label=label, s=8, alpha=0.6)

    ax1.set_yscale('log')
    ax1.set_title(f'S&P 500 Cycle & Regime Detection (Dominant Pulse: {peak_cycle:.2f} Years)', fontsize=16)
    ax1.set_ylabel('Index Value (Log Scale)')
    ax1.legend(loc='upper left', title="Market Regime")
    ax1.grid(True, alpha=0.2, which='both')

    # Bottom Plot: The Cyclical Wave (HP Filter)
    ax2.plot(df_m.index, df_m['Cycle_Dev'], color='teal', linewidth=1)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev'] >= 0), color='green', alpha=0.15)
    ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev'] < 0), color='red', alpha=0.15)
    ax2.set_title('Cyclical Deviation from Secular Trend (The "Heartbeat")', fontsize=13)
    ax2.set_ylabel('Log Deviation')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"Detected Peak Cycle Length: {peak_cycle:.2f} years")

if __name__ == "__main__":
    analyze_sp500_cycles()
