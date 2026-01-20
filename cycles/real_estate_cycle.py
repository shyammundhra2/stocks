import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram, sawtooth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('home_prices.csv')
df = df.rename(columns={'Year': 'Date', 'Home_Prices': 'Real_HPI'})
df = df.dropna().sort_values('Date')

# 2. HP Filter: Cyclical Component
hpi_cycle, hpi_trend = hpfilter(df['Real_HPI'], lamb=129600)  # annual data

# 3. Spectral Analysis for Dominant Cycle
fs = 1 / df['Date'].diff().median()  # sampling frequency in 1/year
frequencies, power_density = periodogram(hpi_cycle, fs=fs)
periods = 1 / frequencies[frequencies > 0]
psd = power_density[frequencies > 0]

mask = (periods >= 10) & (periods <= 25)  # look for cycles 10-25 years
peak_cycle_length = periods[mask][np.argmax(psd[mask])]

# 3b. Predicted Asymmetric Cycle aligned to 2012
t = np.arange(len(df))
duty = 0.7  # 70% expansion, 30% contraction
bottom_year = 2012
bottom_idx = df.index[df['Date'] == bottom_year][0]  # find index of 2012
predicted_cycle = sawtooth(2 * np.pi * (t - bottom_idx) / peak_cycle_length, width=duty) * hpi_cycle.std()

# 4. Regime Detection (GMM)
df['LogReturn'] = np.log(df['Real_HPI'] / df['Real_HPI'].shift(1))
df['Volatility'] = df['LogReturn'].rolling(window=3).std()  # annual, small window
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

# 5. Display Results
print(f"--- Cycle Analysis ---")
print(f"Detected Peak Cycle Length: {peak_cycle_length:.2f} years")
print(f"Predicted bottom aligned to year: {bottom_year}")

# 6. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

colors = {'Expansion': 'forestgreen', 'Neutral': 'royalblue', 'Contraction': 'crimson'}

# Top Plot: Regimes
for label, color in colors.items():
    mask = df_ml['Regime'] == label
    ax1.scatter(df_ml['Date'][mask], df_ml['Real_HPI'][mask], color=color, label=label, s=50)
ax1.set_title(f'Market Regime Detection (Cycle Length: {peak_cycle_length:.2f} Years)', fontsize=14)
ax1.set_ylabel('Home Price Index')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Plot: Cyclical Component + Asymmetric Predicted Cycle
ax2.plot(df['Date'], hpi_cycle, color='teal', label='Cyclical Wave (HP Filter)')
ax2.plot(df['Date'], predicted_cycle, color='blue', linestyle='--', alpha=0.7, label='Predicted Asymmetric Cycle')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.fill_between(df['Date'], 0, hpi_cycle, where=(hpi_cycle >= 0), color='green', alpha=0.15)
ax2.fill_between(df['Date'], 0, hpi_cycle, where=(hpi_cycle < 0), color='red', alpha=0.15)
ax2.set_title('Underlying Cyclical Pulse (Deviation from Trend)', fontsize=12)
ax2.set_ylabel('Deviation')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
