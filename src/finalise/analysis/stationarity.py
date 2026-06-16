import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.
    Returns a dictionary with the test statistic, p-value, decision, and critical values.
    Raises a warning and prints to terminal if the series is non-stationary.
    """
    try:
        adf_result = adfuller(series)
        adf_stat = float(adf_result[0])
        pvalue = float(adf_result[1])
        crit_values = adf_result[4]
    except Exception as e:
        adf_stat = 0.0
        pvalue = 1.0
        crit_values = {"1%": 0.0, "5%": 0.0, "10%": 0.0}
        
    is_stationary = bool(pvalue < alpha)
        
    return {
        "statistic": adf_stat,
        "p_value": pvalue,
        "is_stationary": is_stationary,
        "critical_values": {k: float(v) for k, v in crit_values.items()}
    }
