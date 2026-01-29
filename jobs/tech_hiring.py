import os
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================
# FREE DATA SOURCES
# =============================================

def load_fred_data(start="2005-01-01"):
    """Load comprehensive labor market data from FRED (FREE)"""
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        raise ValueError("FRED_API_KEY environment variable not set")

    fred = Fred(api_key=api_key)

    series = {
        'unrate': 'UNRATE',
        'initial_claims': 'ICSA',
        'cont_claims': 'CCSA',
        'fed_rate': 'DFF',
        'nfp': 'PAYEMS',
        'jolts_total': 'JTSJOL',
        'avg_hours': 'AWHAETP',
        'labor_force': 'CLF16OV',
    }

    data = {}
    failed = []

    for name, series_id in series.items():
        try:
            series_data = fred.get_series(series_id, observation_start=start)
            if series_data is not None and len(series_data) > 0:
                data[name] = series_data
            else:
                failed.append(name)
        except Exception as e:
            print(f"⚠️  Failed to load {name} ({series_id}): {e}")
            failed.append(name)

    if not data:
        raise ValueError("No FRED data loaded successfully")

    print(f"✓ Loaded {len(data)}/{len(series)} FRED series")
    if failed:
        print(f"  Skipped: {', '.join(failed)}")

    df = pd.DataFrame(data)
    df = df.resample('ME').mean()

    if 'unrate' in df.columns:
        df['unrate_chg'] = df['unrate'].pct_change(fill_method=None)
    if 'jolts_total' in df.columns:
        df['jolts_chg'] = df['jolts_total'].pct_change(fill_method=None)
    if 'initial_claims' in df.columns:
        df['claims_chg'] = df['initial_claims'].pct_change(fill_method=None)
    if 'nfp' in df.columns:
        df['nfp_chg'] = df['nfp'].pct_change(fill_method=None)

    return df.dropna()


def load_yf_market_data(start="2005-01-01"):
    """Load market data from Yahoo Finance (FREE)"""
    tickers = ['SPY', '^VIX', 'QQQ', 'HYG']
    ticker_names = ['spy', 'vix', 'qqq', 'hyg']

    data = {}
    failed = []

    for name, ticker in zip(ticker_names, tickers):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, auto_adjust=False)

            if not hist.empty:
                if 'Close' in hist.columns:
                    series = hist['Close']
                    series.index = series.index.tz_localize(None)
                    data[name] = series
                else:
                    print(f"⚠️  No Close column for {name}")
                    failed.append(name)
            else:
                print(f"⚠️  No data for {name}")
                failed.append(name)
        except Exception as e:
            print(f"⚠️  Failed to load {name}: {e}")
            failed.append(name)

    if not data:
        raise ValueError("No market data loaded successfully")

    print(f"✓ Loaded {len(data)}/{len(tickers)} market series")
    if failed:
        print(f"  Skipped: {', '.join(failed)}")

    df = pd.DataFrame(data)
    df = df.resample('ME').last()

    if 'spy' in df.columns:
        df['spy_ret'] = df['spy'].pct_change(fill_method=None)
    if 'qqq' in df.columns:
        df['qqq_ret'] = df['qqq'].pct_change(fill_method=None)
        if 'spy' in df.columns:
            df['qqq_spy_ratio'] = df['qqq'] / df['spy']
    if 'vix' in df.columns:
        df['vix_chg'] = df['vix'].pct_change(fill_method=None)
        df['vix_level'] = df['vix']
    if 'hyg' in df.columns:
        df['credit_spread'] = df['hyg'].pct_change(fill_method=None)

    return df.dropna()


def build_free_dataset():
    """Combine all free data sources"""
    print("\n📊 Loading data sources...")

    fred_data = load_fred_data()
    market_data = load_yf_market_data()

    X = fred_data.join(market_data, how='inner')

    print(f"\n✓ Combined dataset: {len(X)} months")
    print(f"  Date range: {X.index.min().strftime('%Y-%m')} to {X.index.max().strftime('%Y-%m')}")
    print(f"  Features: {X.shape[1]}")

    if 'unrate_chg' in X.columns:
        y = -X['unrate_chg'].shift(-3)
    elif 'unrate' in X.columns:
        y = -X['unrate'].pct_change(fill_method=None).shift(-3)
    else:
        raise ValueError("No unemployment data available for target")

    y = 1 / (1 + np.exp(-10 * y))
    y = y.rename('hiring_prob')

    data = X.join(y, how='inner').dropna()

    if len(data) < 50:
        raise ValueError(f"Insufficient data: only {len(data)} months available")

    return data.drop(columns='hiring_prob'), data['hiring_prob']


def train_model(X, y):
    """Train with walk-forward validation"""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    preds, actuals = [], []

    print("\n🔄 Training with walk-forward validation...")

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        p = model.predict(X_test)

        preds.extend(p)
        actuals.extend(y_test)

        fold_corr = np.corrcoef(y_test, p)[0, 1]
        print(f"  Fold {i}: corr={fold_corr:.3f}, test_size={len(y_test)}")

    corr = np.corrcoef(actuals, preds)[0, 1]
    print(f"\n✓ Overall walk-forward correlation: {corr:.3f}")

    model.fit(X, y)

    feature_imp = pd.Series(
        model.named_steps['gbm'].feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print(f"\n📊 Top 5 Features:")
    for feat, imp in feature_imp.head(5).items():
        print(f"  {feat}: {imp:.3f}")

    return model


# =============================================
# WAVELET DECOMPOSITION & CYCLE EXTRACTION
# =============================================

def wavelet_decomposition(signal_data, wavelet='db4', level=4):
    """
    Decompose signal using discrete wavelet transform

    Parameters:
    - signal_data: 1D array of values
    - wavelet: wavelet type (db4 = Daubechies 4)
    - level: decomposition level

    Returns:
    - coeffs: wavelet coefficients
    - reconstructed: dictionary of reconstructed components
    """
    print(f"\n🌊 Performing wavelet decomposition (wavelet={wavelet}, level={level})...")

    # Perform multi-level discrete wavelet transform
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    # Reconstruct individual components
    reconstructed = {}

    # Approximation (trend)
    approx = pywt.upcoef('a', coeffs[0], wavelet, level=level, take=len(signal_data))
    reconstructed['trend'] = approx[:len(signal_data)]

    # Details (cycles from high to low frequency)
    for i in range(1, len(coeffs)):
        detail = pywt.upcoef('d', coeffs[i], wavelet, level=level - i + 1, take=len(signal_data))
        reconstructed[f'cycle_{i}'] = detail[:len(signal_data)]

    print(f"✓ Extracted {len(reconstructed)} components")

    return coeffs, reconstructed


def identify_dominant_cycles(signal_data, dates, sampling_period=1):
    """
    Identify dominant cycles using FFT

    Parameters:
    - signal_data: 1D array
    - dates: DatetimeIndex
    - sampling_period: months per sample (default=1)

    Returns:
    - DataFrame with cycle periods and amplitudes
    """
    print("\n📈 Identifying dominant cycles using FFT...")

    # Detrend the signal
    detrended = signal.detrend(signal_data)

    # Perform FFT
    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, sampling_period)

    # Get positive frequencies only
    positive_freq_idx = xf > 0
    xf_pos = xf[positive_freq_idx]
    yf_pos = np.abs(yf[positive_freq_idx])

    # Convert frequency to period (in months)
    periods = 1 / xf_pos

    # Find peaks (dominant cycles)
    peaks, properties = signal.find_peaks(yf_pos, height=np.max(yf_pos) * 0.1)

    if len(peaks) > 0:
        cycle_df = pd.DataFrame({
            'period_months': periods[peaks],
            'period_years': periods[peaks] / 12,
            'amplitude': yf_pos[peaks]
        }).sort_values('amplitude', ascending=False)

        print("\n🔍 Dominant Cycles Detected:")
        print(cycle_df.head(5).to_string())

        return cycle_df
    else:
        print("⚠️  No significant cycles detected")
        return pd.DataFrame()


def extrapolate_cycles(reconstructed, dates, future_periods=60):
    """
    Extrapolate each cycle component into the future

    Parameters:
    - reconstructed: dict of cycle components
    - dates: historical dates
    - future_periods: months to forecast (default=60 for 5 years)

    Returns:
    - DataFrame with extrapolated components
    """
    print(f"\n🔮 Extrapolating cycles {future_periods} months into future...")

    last_date = dates[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=future_periods,
        freq='ME'
    )

    extrapolated = {}

    for component_name, component_signal in reconstructed.items():
        if component_name == 'trend':
            # Linear extrapolation of trend
            x = np.arange(len(component_signal))
            z = np.polyfit(x[-36:], component_signal[-36:], 1)  # Use last 3 years
            p = np.poly1d(z)

            future_x = np.arange(len(component_signal), len(component_signal) + future_periods)
            future_trend = p(future_x)
            extrapolated['trend'] = future_trend

        else:
            # Periodic extrapolation for cycles
            # Fit sine wave to last portion of signal
            recent_signal = component_signal[-72:] if len(component_signal) >= 72 else component_signal

            # Use FFT to estimate frequency
            yf = fft(recent_signal)
            xf = fftfreq(len(recent_signal), 1)

            # Find dominant frequency
            positive_idx = xf > 0
            if positive_idx.sum() > 0:
                dominant_freq_idx = np.argmax(np.abs(yf[positive_idx]))
                dominant_freq = xf[positive_idx][dominant_freq_idx]

                # Estimate amplitude and phase
                amplitude = np.std(recent_signal) * 2
                phase = np.angle(yf[positive_idx][dominant_freq_idx])

                # Generate future signal
                future_t = np.arange(future_periods)
                future_cycle = amplitude * np.sin(2 * np.pi * dominant_freq *
                                                  (future_t + len(component_signal)) + phase)

                # Add damping factor (cycles tend to decay over time)
                damping = np.exp(-future_t / (future_periods * 2))
                future_cycle *= damping
            else:
                # Fallback to zeros if no frequency detected
                future_cycle = np.zeros(future_periods)

            extrapolated[component_name] = future_cycle

    print(f"✓ Extrapolated {len(extrapolated)} components")

    return pd.DataFrame(extrapolated, index=future_dates)


def wavelet_forecast(X, y, model, years_ahead=5):
    """
    Generate forecast using wavelet cycle extraction

    Parameters:
    - X: feature data
    - y: target historical
    - model: trained model
    - years_ahead: years to forecast

    Returns:
    - forecast_df: DataFrame with forecast
    - components_df: DataFrame with decomposed components
    """
    print("\n" + "=" * 70)
    print("WAVELET-BASED CYCLE EXTRACTION & FORECASTING")
    print("=" * 70)

    # Get historical predictions
    y_pred_hist = model.predict(X)

    # Perform wavelet decomposition on historical predictions
    coeffs, reconstructed = wavelet_decomposition(y_pred_hist, wavelet='db4', level=4)

    # Identify dominant cycles
    cycle_info = identify_dominant_cycles(y_pred_hist, X.index)

    # Extrapolate cycles
    future_periods = years_ahead * 12
    future_components = extrapolate_cycles(reconstructed, X.index, future_periods)

    # Sum components to get final forecast
    forecast_values = future_components.sum(axis=1).values

    # Clip to valid probability range [0, 1]
    forecast_values = np.clip(forecast_values, 0, 1)

    forecast_df = pd.DataFrame({
        'probability': forecast_values
    }, index=future_components.index)

    # Create historical components DataFrame for visualization
    hist_components = pd.DataFrame(reconstructed, index=X.index)

    print(f"\n✓ Generated {len(forecast_df)} month forecast ({years_ahead} years)")

    return forecast_df, hist_components, future_components, cycle_info


# =============================================
# PLOTTING
# =============================================

def plot_wavelet_forecast(X, y, model, forecast_df, hist_components,
                          future_components, years_history=10):
    """
    Create comprehensive plot with wavelet components and forecast
    """
    y_pred_hist = model.predict(X)

    # Filter to recent history
    cutoff_date = X.index[-1] - pd.DateOffset(years=years_history)
    hist_mask = X.index >= cutoff_date

    X_recent = X[hist_mask]
    y_pred_recent = y_pred_hist[hist_mask]

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # ========== Main Plot ==========
    ax1 = fig.add_subplot(gs[0])

    # Plot historical predictions
    ax1.plot(X_recent.index, y_pred_recent,
             linewidth=2.5, color='#2E86AB',
             label='Historical Predictions', zorder=3)

    # Plot future forecast
    ax1.plot(forecast_df.index, forecast_df['probability'],
             linewidth=2.5, color='#F18F01', linestyle='--',
             label='Wavelet-Based Forecast', zorder=4)

    # Add vertical line at present
    ax1.axvline(X.index[-1], color='gray', linestyle=':', linewidth=2,
                alpha=0.7, label='Present', zorder=1)

    # Add horizontal reference lines
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    ax1.axhline(0.6, color='green', linestyle='--', linewidth=1, alpha=0.3)
    ax1.axhline(0.4, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # Shade regions
    ax1.fill_between(X_recent.index, 0, 1, where=(y_pred_recent > 0.6),
                     alpha=0.1, color='green')
    ax1.fill_between(X_recent.index, 0, 1, where=(y_pred_recent < 0.4),
                     alpha=0.1, color='red')

    ax1.set_ylabel('Hiring Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Tech Hiring Forecast with Wavelet Cycle Extraction (5-Year Outlook)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['0%', '20%', '40%', '50%', '60%', '80%', '100%'])
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)

    # ========== Historical Components ==========
    ax2 = fig.add_subplot(gs[1])

    # Plot trend and major cycles
    recent_components = hist_components[hist_mask]

    ax2.plot(recent_components.index, recent_components['trend'],
             linewidth=2, label='Trend', color='#E63946')

    for col in recent_components.columns:
        if col.startswith('cycle_') and col in ['cycle_1', 'cycle_2']:
            ax2.plot(recent_components.index, recent_components[col],
                     linewidth=1, alpha=0.7, label=col.replace('_', ' ').title())

    ax2.axvline(X.index[-1], color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Component Value', fontsize=10, fontweight='bold')
    ax2.set_title('Historical Wavelet Components', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    # ========== Future Components ==========
    ax3 = fig.add_subplot(gs[2])

    ax3.plot(future_components.index, future_components['trend'],
             linewidth=2, label='Trend', color='#E63946', linestyle='--')

    for col in future_components.columns:
        if col.startswith('cycle_') and col in ['cycle_1', 'cycle_2']:
            ax3.plot(future_components.index, future_components[col],
                     linewidth=1, alpha=0.7, linestyle='--',
                     label=col.replace('_', ' ').title())

    ax3.set_ylabel('Component Value', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3.set_title('Extrapolated Future Components', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=9)

    # Format x-axes
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_path = 'tech_hiring_wavelet_forecast.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to {output_path}")

    return fig


def create_yearly_forecast_table(forecast_df, X):
    """Create yearly summary table for 5-year forecast"""

    # Resample to yearly
    yearly = forecast_df.resample('YE').mean()

    yearly['signal'] = yearly['probability'].apply(
        lambda x: '🟢 Strong' if x > 0.6 else ('🔴 Weak' if x < 0.4 else '🟡 Neutral')
    )

    yearly['probability_pct'] = (yearly['probability'] * 100).round(1)

    print("\n" + "=" * 60)
    print("5-YEAR WAVELET-BASED FORECAST")
    print("=" * 60)
    print(f"{'Year':<10} {'Avg Probability':<20} {'Signal':<15}")
    print("-" * 60)

    for date, row in yearly.iterrows():
        year = date.year
        prob = row['probability_pct']
        signal = row['signal']
        print(f"  {year:<8} {prob:>6.1f}%              {signal:<15}")

    print("=" * 60)

    return yearly


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    if not os.getenv('FRED_API_KEY'):
        print("❌ Error: FRED_API_KEY environment variable not set")
        print("\nGet your free API key from: https://fredaccount.stlouisfed.org/apikeys")
        exit(1)

    try:
        X, y = build_free_dataset()

        print(f"\nFeatures ({len(X.columns)}): {', '.join(X.columns[:10])}{'...' if len(X.columns) > 10 else ''}")

        # Train model
        model = train_model(X, y)

        # Generate wavelet-based forecast
        forecast_df, hist_components, future_components, cycle_info = \
            wavelet_forecast(X, y, model, years_ahead=5)

        # Create yearly summary
        yearly_forecast = create_yearly_forecast_table(forecast_df, X)

        # Create comprehensive plot
        fig = plot_wavelet_forecast(X, y, model, forecast_df,
                                    hist_components, future_components,
                                    years_history=10)

        # Save outputs
        import joblib

        joblib.dump(model, 'hiring_wavelet_model.pkl')
        forecast_df.to_csv('wavelet_forecast_5years.csv')
        yearly_forecast.to_csv('yearly_wavelet_forecast.csv')
        hist_components.to_csv('historical_components.csv')
        future_components.to_csv('future_components.csv')

        if not cycle_info.empty:
            cycle_info.to_csv('dominant_cycles.csv')

        print("\n✓ Model saved to hiring_wavelet_model.pkl")
        print("✓ 5-year forecast saved to wavelet_forecast_5years.csv")
        print("✓ Yearly forecast saved to yearly_wavelet_forecast.csv")
        print("✓ Components saved to historical_components.csv & future_components.csv")
        if not cycle_info.empty:
            print("✓ Cycle analysis saved to dominant_cycles.csv")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()