import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram, sawtooth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

# ===========================
# 1. Load Data
# ===========================
df = pd.read_csv('home_prices.csv')
df = df.rename(columns={'Year': 'Date', 'Home_Prices': 'Real_HPI'})
df = df.dropna().sort_values('Date')

# ===========================
# 2. HP Filter: Cyclical Component
# ===========================
hpi_cycle, hpi_trend = hpfilter(df['Real_HPI'], lamb=129600)  # annual data

# ===========================
# 3. Spectral Analysis for Dominant Cycle
# ===========================
fs = 1 / df['Date'].diff().median()  # sampling frequency in 1/year
frequencies, power_density = periodogram(hpi_cycle, fs=fs)
periods = 1 / frequencies[frequencies > 0]
psd = power_density[frequencies > 0]

mask = (periods >= 10) & (periods <= 25)  # look for cycles 10-25 years
peak_cycle_length = periods[mask][np.argmax(psd[mask])]

# ===========================
# 4. Regime Detection (GMM)
# ===========================
df['LogReturn'] = np.log(df['Real_HPI'] / df['Real_HPI'].shift(1))
df['Volatility'] = df['LogReturn'].rolling(window=3).std()  # small annual window
df_ml = df.dropna().copy()

X = StandardScaler().fit_transform(df_ml[['LogReturn', 'Volatility']])
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
df_ml['Regime_ID'] = gmm.predict(X)

# Map regimes: highest mean log return = Expansion, lowest = Contraction
means = df_ml.groupby('Regime_ID')['LogReturn'].mean()
mapping = {means.idxmax(): 'Expansion', means.idxmin(): 'Contraction'}
for i in range(3):
    if i not in mapping:
        mapping[i] = 'Neutral'
df_ml['Regime'] = df_ml['Regime_ID'].map(mapping)

# ===========================
# 5. Fit Asymmetric Sawtooth Cycle
# ===========================
def asymmetric_sawtooth(t, amplitude, period, phase, duty):
    """Amplitude-scaled asymmetric sawtooth wave."""
    return amplitude * sawtooth(2 * np.pi * (t - phase) / period, width=duty)

t = np.arange(len(df))
y_actual = hpi_cycle.values

# Initial guesses
amplitude_guess = y_actual.std()
period_guess = peak_cycle_length
bottom_year = 2012
bottom_idx = df.index[df['Date'] == bottom_year][0]  # align bottom
phase_guess = bottom_idx
duty_guess = 0.7

p0 = [amplitude_guess, period_guess, phase_guess, duty_guess]

# Fit curve
params, _ = curve_fit(
    asymmetric_sawtooth, t, y_actual,
    p0=p0, bounds=([0, 5, 0, 0.1], [np.inf, 50, len(t), 0.9])
)
amplitude_fit, period_fit, phase_fit, duty_fit = params

print(f"--- Cycle Analysis ---")
print(f"Detected Peak Cycle Length (spectral): {peak_cycle_length:.2f} years")
print(f"Fitted Parameters:")
print(f"Amplitude: {amplitude_fit:.2f}")
print(f"Period: {period_fit:.2f} years")
print(f"Phase (index of bottom): {phase_fit:.0f}")
print(f"Duty cycle: {duty_fit:.2f}")

# ===========================
# 6. Extend Prediction 4 Years
# ===========================
years_extended = 4
t_extended = np.arange(len(df) + years_extended)
predicted_extended = asymmetric_sawtooth(t_extended, amplitude_fit, period_fit, phase_fit, duty_fit)

# Extend Date index
last_year = df['Date'].iloc[-1]
date_extended = np.arange(df['Date'].iloc[0], last_year + years_extended + 1)

# ===========================
# 7. Visualization
# ===========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Regimes
colors = {'Expansion': 'forestgreen', 'Neutral': 'royalblue', 'Contraction': 'crimson'}
for label, color in colors.items():
    mask = df_ml['Regime'] == label
    ax1.scatter(df_ml['Date'][mask], df_ml['Real_HPI'][mask], color=color, label=label, s=50)
ax1.set_title(f'Market Regime Detection (Cycle Length: {peak_cycle_length:.2f} Years)', fontsize=14)
ax1.set_ylabel('Home Price Index')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cyclical Component + Fitted & Extended
ax2.plot(df['Date'], hpi_cycle, color='teal', label='Actual Cyclical Component')
ax2.plot(date_extended, predicted_extended, color='blue', linestyle='--', alpha=0.7, label='Fitted + Extended Predicted Cycle')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.fill_between(df['Date'], 0, hpi_cycle, where=(hpi_cycle >= 0), color='green', alpha=0.15)
ax2.fill_between(df['Date'], 0, hpi_cycle, where=(hpi_cycle < 0), color='red', alpha=0.15)
ax2.set_title('Underlying Cyclical Pulse vs Fitted Asymmetric Cycle (+4 Years)', fontsize=12)
ax2.set_ylabel('Deviation from Trend')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
