import numpy as np
import pandas as pd

def compute(prices: pd.Series, k: int, overlapping: bool = False) -> pd.Series:
    """
    Computes log-returns.
    Daily: r_1(t) = log(P(t) / P(t-1)).
    For k > 1: sum of daily log-returns over window k.
    If overlapping is False, returns non-overlapping returns.
    """
    # Compute daily log-returns
    r1 = np.log(prices / prices.shift(1))
    
    if k == 1:
        return r1.dropna()
        
    r1_clean = r1.dropna()
    
    if not overlapping:
        # Non-overlapping: rolling sum of window k, sampled every k-th element starting from k-1
        r_k = r1_clean.rolling(window=k).sum()
        return r_k.iloc[k - 1::k].dropna()
    else:
        # Overlapping
        r_k = r1_clean.rolling(window=k).sum()
        return r_k.dropna()
