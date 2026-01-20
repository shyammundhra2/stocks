import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram, correlate
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def analyze_asset_cycles(ticker, start_date="1970-01-01", extend_months=48):
    print(f"Downloading historical data for {ticker}...")
    df = yf.download(ticker, start=start_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Monthly resample
    df_m = df['Close'].resample('ME').last().asfreq('M').ffill().to_frame()
    df_m['Log_Price'] = np.log(df_m['Close'])

    # HP filter
    cycle, trend = hpfilter(df_m['Log_Price'], lamb=129600)
    df_m['Cycle_Dev'] = cycle

    # Spectral analysis
    fs = 12  # monthly
    frequencies, power_density = periodogram(df_m['Cycle_Dev'].dropna(), fs=fs)
    periods = 1 / frequencies[frequencies > 0]
    psd = power_density[frequencies > 0]
    mask = (periods >= 2) & (periods <= 15)
    peak_cycle = periods[mask][np.argmax(psd[mask])]
    print(f"{ticker}: Detected Peak Cycle Length: {peak_cycle:.2f} years")

    # Generate historical predicted sine wave
    t = np.arange(len(df_m))
    predicted_cycle = np.sin(2 * np.pi * t / (peak_cycle * 12))

    # --- Align phase using cross-correlation ---
    corr = correlate(df_m['Cycle_Dev'].fillna(0), predicted_cycle, mode='full')
    shift_idx = corr.argmax() - (len(df_m) - 1)
    predicted_cycle_aligned = np.roll(predicted_cycle, shift_idx) * df_m['Cycle_Dev'].std()
    df_m['Predicted_Cycle'] = predicted_cycle_aligned

    # Extend predicted cycle into the future
    t_future = np.arange(len(df_m) + extend_months)
    predicted_cycle_future = np.sin(2 * np.pi * t_future / (peak_cycle * 12))
    predicted_cycle_future = np.roll(predicted_cycle_future, shift_idx) * df_m['Cycle_Dev'].std()
    last_date = df_m.index[-1]
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=extend_months, freq='M')
    future_index = df_m.index.append(future_dates)
    predicted_cycle_full = np.concatenate([predicted_cycle_aligned, predicted_cycle_future[-extend_months:]])

    return df_m, predicted_cycle_full, future_index, peak_cycle

def plot_assets_with_forecast(assets_data):
    n_assets = len(assets_data)
    fig, axes = plt.subplots(n_assets, 2, figsize=(18, 5 * n_assets),
                             gridspec_kw={'width_ratios':[2,1]})

    for i, (name, (df_m, predicted_full, full_index, peak_cycle)) in enumerate(assets_data.items()):
        # Left: Price
        ax1 = axes[i,0] if n_assets > 1 else axes[0]
        ax1.plot(df_m.index, df_m['Close'], color='black', label='Price')
        ax1.set_yscale('log')
        ax1.set_title(f"{name} Price (Log Scale)")
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.2)

        # Right: Cycles + forecast
        ax2 = axes[i,1] if n_assets > 1 else axes[1]
        ax2.plot(df_m.index, df_m['Cycle_Dev'], color='teal', linewidth=1, label='HP Cycle')
        ax2.plot(full_index, predicted_full, color='blue', linestyle='--', alpha=0.8, label='Predicted Cycle')
        ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev']>=0), color='green', alpha=0.15)
        ax2.fill_between(df_m.index, 0, df_m['Cycle_Dev'], where=(df_m['Cycle_Dev']<0), color='red', alpha=0.15)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title(f"{name} Cyclical Deviation + 4-Year Forecast")
        ax2.set_ylabel('Log Deviation')
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc='upper left')

        # --- Cycle duration annotation ---
        ax2.annotate(
            f"Dominant Cycle: {peak_cycle:.2f} yrs",
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.4, edgecolor='black')
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    assets = {
        "S&P 500": "^GSPC",
        "Gold": "GC=F",
        "WTI Oil": "CL=F"
    }

    assets_data = {}
    for name, ticker in assets.items():
        df_m, predicted_full, full_index, peak_cycle = analyze_asset_cycles(ticker, extend_months=48)
        assets_data[name] = (df_m, predicted_full, full_index, peak_cycle)

    plot_assets_with_forecast(assets_data)
