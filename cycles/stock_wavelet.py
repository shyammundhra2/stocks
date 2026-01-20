import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pywt
from scipy.optimize import leastsq
from scipy.stats import norm


def run_seamless_macro_model():
    # 1. DATA PREP
    ticker = "^GSPC"
    df = yf.download(ticker, start="1973-01-01")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    data = np.log(df['Close'].resample('ME').last().ffill())
    # Centering using a 200-month window to capture the Kuznets cycle
    data_centered = (data - data.rolling(window=200).mean()).dropna()

    # 2. WAVELET EXTRACTION
    # Scales: 42mo (Kitchin), 200mo (Kuznets)
    c_kitchin, _ = pywt.cwt(data_centered.values, np.arange(30, 60), 'cmor1.5-1.0')
    c_kuznets, _ = pywt.cwt(data_centered.values, np.arange(180, 240), 'cmor1.5-1.0')
    k_wave_raw = np.real(c_kitchin.mean(axis=0))
    z_wave_raw = np.real(c_kuznets.mean(axis=0))

    # 3. THE "SURGICAL JOIN" LOGIC
    # We trim the last 12 months because they are distorted by edge effects
    trim = 12
    k_hist = k_wave_raw[:-trim]
    z_hist = z_wave_raw[:-trim]
    hist_dates = data_centered.index[:-trim]

    # Projection Fitting (based on the clean data before the trim)
    def project(wave, freq, periods=72):  # 6-year forecast
        x = np.arange(len(wave))
        x_fit, y_fit = x[-180:], wave[-180:]  # Fit to last 15 years
        optimize_func = lambda p: p[0] * np.sin(2 * np.pi * freq * x_fit + p[1]) - y_fit
        p_est, _ = leastsq(optimize_func, [np.std(y_fit), 0])

        # Project starting EXACTLY from the trim point
        x_proj = np.arange(len(wave), len(wave) + periods)
        return p_est[0] * np.sin(2 * np.pi * freq * x_proj + p_est[1])

    k_proj = project(k_hist, 1 / 42)
    z_proj = project(z_hist, 1 / 200)

    # Create future dates starting from the end of k_hist
    proj_dates = pd.date_range(start=hist_dates[-1] + pd.DateOffset(months=1), periods=len(k_proj), freq='ME')

    # 4. RECESSION PROBABILITY (Joined)
    def calc_risk(k, z):
        # Normalizing to ensure 0-100 scale
        score = (k / np.std(k)) + (z / np.std(z))
        return 100 * norm.cdf(-score / 1.6)

    hist_risk = calc_risk(k_hist, z_hist)
    proj_risk = calc_risk(k_proj, z_proj)

    # 5. VISUALIZATION
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Cycles Panel
    ax1.plot(hist_dates, k_hist, color='teal', label="Kitchin (Historical)")
    ax1.plot(hist_dates, z_hist, color='crimson', linewidth=2.5, label="Kuznets (Historical)")
    ax1.plot(proj_dates, k_proj, color='teal', linestyle=':', label="Kitchin Forecast")
    ax1.plot(proj_dates, z_proj, color='crimson', linestyle=':', linewidth=2.5, label="Kuznets Forecast")
    ax1.axvline(hist_dates[-1], color='black', alpha=0.3, label="Surgical Join Point")
    ax1.set_title("1973-2031 Seamless Cycle Forecast")
    ax1.legend(loc='upper left')

    # Risk Panel
    ax2.plot(hist_dates, hist_risk, color='black', alpha=0.4)
    ax2.plot(proj_dates, proj_risk, color='red', linewidth=2, label="2026-2031 Recession Risk")
    ax2.fill_between(proj_dates, 50, proj_risk, where=(proj_risk > 50), color='red', alpha=0.2)
    ax2.axhline(80, color='darkred', linestyle='--', label="80% Danger Zone")
    ax2.set_ylabel("Recession Probability (%)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_seamless_macro_model()