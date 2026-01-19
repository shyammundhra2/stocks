import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def analyze_sp500_cycles():
    # -----------------------------
    # 1️⃣ Download S&P 500 History
    # -----------------------------
    ticker = "^GSPC"
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, start="1927-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Resample to Monthly End (ME)
    df_m = df['Close'].resample('ME').last().to_frame()
    df_m = df_m.asfreq('M').ffill()  # Fill any missing months
    df_m['Log_Price'] = np.log(df_m['Close'])

    # -----------------------------
    # 2️⃣ HP Filter: Cyclical Component
    # -----------------------------
    hpi_cycle, hpi_trend = hpfilter(df_m['Log_Price'], lamb=129600)
    df_m['Cycle_Dev'] = hpi_cycle

    # -----------------------------
    # 3️⃣ Spectral Analysis for Dominant Cycle
    # -----------------------------
    fs = 12  # monthly frequency
    frequencies, power_density = periodogram(df_m['Cycle_Dev'].dropna(), fs=fs)
    periods = 1 / frequencies[frequencies > 0]
    psd = power_density[frequencies > 0]
    mask = (periods >= 2) & (periods <= 15)
    peak_cycle = periods[mask][np.argmax(psd[mask])]
    print(f"Detected Peak Cycle Length: {peak_cycle:.2f} years")

    # Prepare the theoretical cycle wave
    t = np.arange(len(df_m))
    predicted_cycle = np.sin(2 * np.pi * t / (peak_cycle * 12))  # scale to months

    # -----------------------------
    # 4️⃣ ML Regime Detection (GMM)
    # -----------------------------
    df_m['Log_Return'] = df_m['Log_Price'].diff()
    df_m['Volatility'] = df_m['Log_Return'].rolling(window=12).std()
    df_ml = df_m.dropna().copy()

    X = StandardScaler().fit_transform(df_ml[['Log_Return', 'Volatility']])
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(X)
    df_ml['Regime_ID'] = gmm.predict(X)

    # Label clusters
    means = df_ml.groupby('Regime_ID')['Log_Return'].mean()
    mapping = {means.idxmax(): 'Bull (Expansion)',
               means.idxmin(): 'Bear (Contraction)'}
    for i in range(3):
        if i not in mapping:
            mapping[i] = 'Neutral/Transition'
    df_ml['Regime'] = df_ml['Regime_ID'].map(mapping)

    # -----------------------------
    # 5️⃣ Visualization
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Top: S&P 500 Log Prices with Regime Coloring
    colors = {'Bull (Expansion)': '#2ca02c', 'Neutral/Transition': '#ff7f0e', 'Bear (Contraction)': '#d62728'}
    for label, color in colors.items():
        mask = df_ml['Regime'] == label
        ax1.scatter(df_ml.index[mask], df_ml['Close'][mask], color=color, label=label, s=8, alpha=0.6)

    ax1.set_yscale('log')
    ax1.set_title(f"S&P 500 Cycle & Regime Detection (Dominant Pulse: {peak_cycle:.2f} Years)", fontsize=16)
    ax1.set_ylabel('Index Value (Log Scale)')
    ax1.legend(loc='upper left', title="Market Regime")
    ax1.grid(True, alpha=0.2, which='both')

    # Bottom: Cyclical Deviation + Predicted Cycle Wave
    ax2.plot(df_m.index, df_m['Cycle_Dev'], color='teal', linewidth=1, label='HP Filter Cycle')
    ax2.plot(df_m.index, predicted_cycle * df_m['Cycle_Dev'].std(), color='blue', linestyle='--', alpha=0.7, label='Predicted Cycle')

    # Fill expansions and contractions
    ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev'] >= 0), color='green', alpha=0.15)
    ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev'] < 0), color='red', alpha=0.15)

    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Cyclical Deviation from Secular Trend (Market Heartbeat)', fontsize=14)
    ax2.set_ylabel('Log Deviation')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_sp500_cycles()

