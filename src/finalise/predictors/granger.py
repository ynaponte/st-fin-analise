import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def test(x: pd.Series, y: pd.Series, lag: int, alpha: float = 0.05) -> dict:
    """
    Performs Granger causality test of x -> y at the specified lag.
    Returns a dictionary with p-value, decision (is_causal), and test statistic.
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ['y', 'x']
    
    if lag < 1:
        lag = 1
        
    if len(df) <= 3 * lag + 2:
        return {
            "p_value": 1.0,
            "is_causal": False,
            "statistic": 0.0
        }
        
    try:
        res = grangercausalitytests(df[['y', 'x']], maxlag=[lag], verbose=False)
        stats = res[lag][0]
        p_val = float(stats['ssr_ftest'][1])
        stat_val = float(stats['ssr_ftest'][0])
    except Exception:
        p_val = 1.0
        stat_val = 0.0
        
    is_causal = bool(p_val < alpha)
    return {
        "p_value": p_val,
        "is_causal": is_causal,
        "statistic": stat_val
    }
