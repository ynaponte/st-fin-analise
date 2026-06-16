"""
finalise.smoothing.window
-------------------------
Calculates the optimal smoothing window size using Cumulative Power Spectral Density (PSD).
"""

import numpy as np
import pandas as pd
from scipy.signal import periodogram


def find_optimal_window(series: pd.Series, threshold: float = 0.99) -> int:
    """
    Discovers the optimal (smallest) window size by analyzing the cumulative 
    Power Spectral Density (PSD) of the series.

    The function identifies the frequency that captures `threshold` (default 80%) 
    of the total signal power (variance). The corresponding period (1 / frequency) 
    is returned as the ideal window size to smooth out higher-frequency noise.

    Parameters
    ----------
    series : pd.Series
        The time series to analyze.
    threshold : float
        The cumulative power threshold (0.0 to 1.0). Default is 0.99.

    Returns
    -------
    int
        The recommended window size (minimum 2).
    """
    clean_series = series.dropna()
    
    if len(clean_series) < 4:
        return 2
        
    # Compute PSD using periodogram. 
    # Detrend 'linear' removes the drift (trend) typical in random walks
    freqs, psd = periodogram(clean_series, detrend='linear')
    
    # Ignore DC component
    freqs = freqs[1:]
    psd = psd[1:]
    
    cum_psd = np.cumsum(psd)
    total_power = cum_psd[-1]
    
    if total_power == 0:
        return 2
        
    normalized_cum_psd = cum_psd / total_power
    
    # Find index where cumulative power exceeds threshold
    idx = np.searchsorted(normalized_cum_psd, threshold)
    
    if idx >= len(freqs):
        idx = len(freqs) - 1
        
    target_freq = freqs[idx]
    
    # Avoid division by zero if freq is somehow 0
    if target_freq <= 0:
        return 60
        
    # The optimal window is the period corresponding to the target frequency
    optimal_window = int(round(1.0 / target_freq))
    
    # Cap between 2 and 60 days for daily financial time series
    return max(2, min(optimal_window, 60))
