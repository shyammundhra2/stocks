import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import periodogram
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 1. Load Data
df = pd.read_csv('home_prices.csv', skiprows=7, usecols=[0, 1], names=['Date', 'Real_HPI'])
df = df.dropna().apply(pd.to_numeric, errors='coerce').dropna().sort_values('Date')

# 2. Identify Cycle Length (Spectral Analysis)
# Isolate the cyclical wave using HP Filter
hpi_cycle, hpi_trend = hpfilter(df['Real_HPI'], lamb=129600)
fs = 1 / df['Date'].diff().median() 
frequencies, power_density = periodogram(hpi_cycle, fs=fs)
periods = 1 / frequencies[frequencies > 0]
psd = power_density[frequencies > 0]

# Identify the peak period (the most dominant cycle length)
mask = (periods >= 10) & (periods <= 25)
peak_cycle_length = periods[mask][np.argmax(psd[mask])]

# 3. Identify Regimes (ML - Gaussian Mixture Model)
df['LogReturn'] = np.log(df['Real_HPI'] / df['Real_HPI'].shift(1))
df['Volatility'] = df['LogReturn'].rolling(window=12).std()
df_ml = df.dropna().copy()

# Cluster based on returns and risk
X = StandardScaler().fit_transform(df_ml[['LogReturn', 'Volatility']])
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
df_ml['Regime_ID'] = gmm.predict(X)

# Map clusters to meaningful labels
means = df_ml.groupby('Regime_ID')['LogReturn'].mean()
mapping = {means.idxmax(): 'Expansion', means.idxmin(): 'Contraction'}
for i in range(3):
    if i not in mapping: mapping[i] = 'Neutral'
df_ml['Regime'] = df_ml['Regime_ID'].map(mapping)

# 4. Display Results
print(f"--- Cycle Analysis ---")
print(f"Detected Peak Cycle Length: {peak_cycle_length:.2f} years")

# 5. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top Plot: Regimes
colors = {'Expansion': 'forestgreen', 'Neutral': 'royalblue', 'Contraction': 'crimson'}
for label, color in colors.items():
    mask = df_ml['Regime'] == label
    ax1.scatter(df_ml['Date'][mask], df_ml['Real_HPI'][mask], color=color, label=label, s=12)
ax1.set_title(f'Market Regime Detection (Cycle Length: {peak_cycle_length:.2f} Years)', fontsize=14)
ax1.set_ylabel('Real Price Index')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Plot: Cyclical Component
ax2.plot(df['Date'], hpi_cycle, color='teal', label='Cyclical Wave (HP Filter)')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
ax2.set_title('Underlying Cyclical Pulse (Deviation from Trend)', fontsize=12)
ax2.set_ylabel('Deviation')
ax2.set_xlabel('Year')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
