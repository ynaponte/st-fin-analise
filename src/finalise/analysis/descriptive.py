"""
analysis.descriptive
--------------------
Descriptive statistics and scaling exponent analysis (Hurst).
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import anderson


def stats(series: pd.Series) -> dict:
    """
    Computes mean, standard deviation, skewness, and Pearson kurtosis.
    Runs the Anderson-Darling normality test.
    
    Parameters
    ----------
    series : pd.Series – The time series to analyze.
    
    Returns
    -------
    dict with keys:
        ``mean``                : float
        ``std``                 : float
        ``skewness``            : float
        ``kurtosis``            : float (Pearson kurtosis, not excess)
        ``is_normal``           : bool (based on 5% significance)
        ``ad_statistic``        : float
        ``ad_critical_value_5`` : float
    """
    series_clean = series.dropna()
    
    mean_val = float(series_clean.mean())
    std_val = float(series_clean.std())
    # Use pandas skew and kurtosis (convert excess kurtosis to Pearson by adding 3)
    skew_val = float(series_clean.skew())
    kurtosis_val = float(series_clean.kurt() + 3)
    
    n = len(series_clean)
    if n < 3:
        return {
            "mean": mean_val,
            "std": std_val,
            "skewness": skew_val,
            "kurtosis": kurtosis_val,
            "is_normal": False,
            "ad_statistic": float('nan'),
            "ad_critical_value_5": float('nan'),
        }
        
    # Anderson-Darling test for normality
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        ad_result = anderson(series_clean, dist='norm')
    
    sig_levels = ad_result.significance_level
    crit_vals = ad_result.critical_values
    
    # Look up critical value for 5% significance level
    idx_5 = 2
    if 5.0 in sig_levels:
        idx_5 = list(sig_levels).index(5.0)
    crit_5 = crit_vals[idx_5]
    
    # If the statistic is larger than critical value, reject normality
    is_normal = bool(ad_result.statistic < crit_5)
    
    return {
        "mean": mean_val,
        "std": std_val,
        "skewness": skew_val,
        "kurtosis": kurtosis_val,
        "is_normal": is_normal,
        "ad_statistic": float(ad_result.statistic),
        "ad_critical_value_5": float(crit_5),
    }


def hurst(series: pd.Series) -> dict:
    """
    Computes the Hurst exponent using Detrended Fluctuation Analysis (DFA).

    The input series must be stationary (I(0)), e.g. log-returns.
    The DFA integrates the series internally (cumulative sum of deviations
    from the mean) and then estimates the scaling exponent α from
    fluctuation vs. scale in log-log space.

    Mapping of the DFA scaling exponent α to the Hurst exponent H
    (Peng et al., 1994; Kantelhardt, 2002):
      - For stationary input (fGn / I(0)):  α ∈ (0, 1)  →  H = α
      - For non-stationary input (fBm / I(1)):  α ∈ (1, 2)  →  H = α − 1

    Parameters
    ----------
    series : pd.Series – The time series to analyze.

    Returns
    -------
    dict with keys:
      ``hurst``         : Hurst exponent clipped to (0.001, 0.999).
      ``distance_05``   : |H - 0.5|.
    """
    x = np.asarray(series.dropna())
    N = len(x)
    
    if N < 16:
        return {"hurst": 0.5, "distance_05": 0.0}
        
    # 1. Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(x - np.mean(x))
    
    # 2. Define scale sizes (s)
    # Kantelhardt recommends scale_min = 4 and scale_max = N // 4
    scale_min = 4
    scale_max = max(scale_min + 2, N // 4)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), num=20, dtype=int))
    
    # Ensure at least a few points for regression
    if len(scales) < 4:
        scales = np.arange(scale_min, max(scale_min + 4, scale_max + 1))
    
    fluctuations = []
    for s in scales:
        num_segments = N // s
        F2 = 0.0
        for i in range(num_segments):
            segment = y[i*s : (i+1)*s]
            x_seg = np.arange(s)
            # Linear fit
            poly = np.polyfit(x_seg, segment, 1)
            fit = np.polyval(poly, x_seg)
            F2 += np.mean((segment - fit) ** 2)
        fluctuations.append(np.sqrt(F2 / num_segments))
        
    # Fit log(F) vs log(scale) to find alpha safely
    log_scales = np.log(scales)
    log_flucts = np.log(np.clip(fluctuations, 1e-15, None))
    
    try:
        alpha, _ = np.polyfit(log_scales, log_flucts, 1)
        if np.isnan(alpha) or np.isinf(alpha):
            alpha = 0.5
    except Exception:
        alpha = 0.5
        
    # Map DFA scaling exponent α to Hurst exponent H.
    # Stationary input (fGn / I(0)):  α ∈ (0, 1) → H = α
    # Non-stationary input (fBm / I(1)):  α ∈ (1, 2) → H = α − 1
    if alpha >= 1.0:
        H = alpha - 1.0
    else:
        H = alpha
        
    H = float(np.clip(H, 0.001, 0.999))
    
    return {
        "hurst": H,
        "distance_05": float(abs(H - 0.5))
    }
