import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def engle_granger(price_x: pd.Series, price_y: pd.Series, alpha: float = 0.05) -> dict:
    """
    Performs the Engle-Granger cointegration test between price_x and price_y.
    Input must be raw prices (strictly positive, non-stationary), not log-returns.
    Raises ValueError if negative values are present in input series.
    """
    if (price_x < 0).any() or (price_y < 0).any():
        raise ValueError("A entrada do teste de cointegração deve ser de preços brutos (positivos), não log-retornos.")
        
    df = pd.concat([price_y, price_x], axis=1).dropna()
    df.columns = ['y', 'x']
    
    if len(df) < 10:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "is_cointegrated": False,
            "spread": pd.Series(dtype='float64'),
            "beta": 0.0,
            "alpha_const": 0.0
        }
        
    coint_t, pvalue, _ = coint(df['y'], df['x'])
    
    X = sm.add_constant(df['x'])
    model = sm.OLS(df['y'], X).fit()
    spread = model.resid
    
    is_cointegrated = bool(pvalue < alpha)
    
    return {
        "statistic": float(coint_t),
        "p_value": float(pvalue),
        "is_cointegrated": is_cointegrated,
        "spread": spread,
        "beta": float(model.params['x']),
        "alpha_const": float(model.params['const'])
    }
