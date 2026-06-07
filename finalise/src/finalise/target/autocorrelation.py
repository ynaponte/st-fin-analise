import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

def acf_pacf(series: pd.Series, lags: int, alpha: float = 0.05) -> dict:
    """
    Computes ACF and PACF for the series and the squared series.
    Returns a dictionary of coefficients and confidence intervals.
    """
    # Drop any NaNs
    series_clean = series.dropna()
    
    # Limit lags to len(series) // 2 - 1 to avoid statsmodels error for small series
    n = len(series_clean)
    effective_lags = min(lags, n // 2 - 1)
    if effective_lags < 1:
        effective_lags = 1
        
    acf_vals, acf_conf = acf(series_clean, nlags=effective_lags, alpha=alpha)
    pacf_vals, pacf_conf = pacf(series_clean, nlags=effective_lags, alpha=alpha)
    
    series_sq = series_clean ** 2
    acf_sq_vals, acf_sq_conf = acf(series_sq, nlags=effective_lags, alpha=alpha)
    pacf_sq_vals, pacf_sq_conf = pacf(series_sq, nlags=effective_lags, alpha=alpha)
    
    return {
        "acf": acf_vals.tolist(),
        "acf_confint": acf_conf.tolist(),
        "pacf": pacf_vals.tolist(),
        "pacf_confint": pacf_conf.tolist(),
        "acf_sq": acf_sq_vals.tolist(),
        "acf_sq_confint": acf_sq_conf.tolist(),
        "pacf_sq": pacf_sq_vals.tolist(),
        "pacf_sq_confint": pacf_sq_conf.tolist()
    }
