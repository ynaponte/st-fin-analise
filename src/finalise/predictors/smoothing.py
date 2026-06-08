import pandas as pd
import numpy as np
from scipy.signal import group_delay as scipy_group_delay
from scipy.signal.windows import gaussian

def apply(series: pd.Series, window: int, method: str) -> pd.Series:
    """
    Applies moving average smoothing without look-ahead bias.
    Supported methods: 'SMA', 'EMA', 'DEMA', 'TMA', 'Gaussian' (or 'Gaussiana').
    """
    method_upper = method.upper()
    
    if method_upper == "SMA":
        return series.rolling(window=window, min_periods=1).mean()
        
    elif method_upper == "EMA":
        return series.ewm(span=window, adjust=False, min_periods=1).mean()
        
    elif method_upper == "DEMA":
        ema1 = series.ewm(span=window, adjust=False, min_periods=1).mean()
        ema2 = ema1.ewm(span=window, adjust=False, min_periods=1).mean()
        return 2 * ema1 - ema2
        
    elif method_upper == "TMA":
        w1 = (window + 1) // 2
        w2 = window // 2 + 1 if window % 2 == 0 else w1
        return series.rolling(window=w1, min_periods=1).mean().rolling(window=w2, min_periods=1).mean()
        
    elif method_upper in ["GAUSSIAN", "GAUSSIANA"]:
        return series.rolling(window=window, win_type='gaussian', min_periods=1).mean(std=window / 6.0)
        
    else:
        raise ValueError(f"Método de suavização desconhecido: {method}")

def group_delay(window: int, method: str) -> float:
    """
    Returns the phase/group delay in samples for the specified filter and window size.
    """
    method_upper = method.upper()
    
    if method_upper == "SMA":
        b = np.ones(window) / window
        a = [1]
    elif method_upper == "EMA":
        alpha = 2 / (window + 1)
        b = [alpha]
        a = [1, -(1 - alpha)]
    elif method_upper == "DEMA":
        alpha = 2 / (window + 1)
        b = [alpha * (2 - alpha), -2 * alpha * (1 - alpha)]
        a = [1, -2 * (1 - alpha), (1 - alpha)**2]
    elif method_upper == "TMA":
        w1 = (window + 1) // 2
        w2 = window // 2 + 1 if window % 2 == 0 else w1
        b = np.convolve(np.ones(w1) / w1, np.ones(w2) / w2)
        a = [1]
    elif method_upper in ["GAUSSIAN", "GAUSSIANA"]:
        b = gaussian(window, std=window / 6.0)
        b /= b.sum()
        a = [1]
    else:
        raise ValueError(f"Método de suavização desconhecido: {method}")
        
    _, gd = scipy_group_delay((b, a), w=[0.0])
    return float(gd[0])
