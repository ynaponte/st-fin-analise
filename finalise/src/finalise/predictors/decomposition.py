import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def stl(series: pd.Series, period: int) -> dict:
    """
    Computes a causal STL decomposition of the closing prices.
    Returns a dictionary of pandas Series: {'trend', 'seasonal', 'residual'}.
    No value at time t uses data from after time t.
    """
    n = len(series)
    trend = np.full(n, np.nan)
    seasonal = np.full(n, np.nan)
    residual = np.full(n, np.nan)
    
    # STL requires at least 2 * period observations
    min_obs = max(10, 2 * period)
    
    # Progressive expanding window fit to ensure strict causality (no look-ahead)
    for t in range(min_obs, n + 1):
        sub_series = series.iloc[:t]
        res = STL(sub_series, period=period, robust=False).fit()
        trend[t-1] = res.trend.iloc[-1]
        seasonal[t-1] = res.seasonal.iloc[-1]
        residual[t-1] = res.resid.iloc[-1]
        
    # Fill starting values (where STL could not run) so that:
    # trend + seasonal + residual = series
    nan_mask = np.isnan(trend)
    trend[nan_mask] = series.values[nan_mask]
    seasonal[nan_mask] = 0.0
    residual[nan_mask] = 0.0
    
    return {
        "trend": pd.Series(trend, index=series.index),
        "seasonal": pd.Series(seasonal, index=series.index),
        "residual": pd.Series(residual, index=series.index)
    }
