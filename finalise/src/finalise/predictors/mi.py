import numpy as np
import pandas as pd
from finalise.target.entropy import ksg_mi

def cross_mi_lags(x: pd.Series, y: pd.Series, lag_max: int, alpha: float = 0.05, k: int = 5) -> dict:
    """
    Computes cross-mutual information between x(t-lag) and y(t) for lag = 1..lag_max.
    Uses Kraskov k-NN estimator.
    Applies Bonferroni correction for significance: threshold = alpha / lag_max.
    """
    mi_profile = []
    
    for lag in range(1, lag_max + 1):
        # Align series: y(t) and x(t-lag)
        df = pd.concat([y, x.shift(lag)], axis=1).dropna()
        if len(df) <= k + 2:
            mi_profile.append(0.0)
            continue
            
        x_lagged = df.iloc[:, 1].values
        y_aligned = df.iloc[:, 0].values
        mi_profile.append(ksg_mi(x_lagged, y_aligned, k=k))
        
    lag_opt = int(np.argmax(mi_profile) + 1) if len(mi_profile) > 0 else 1
    max_mi = mi_profile[lag_opt - 1] if len(mi_profile) > 0 else 0.0
    
    # Bonferroni threshold
    threshold = alpha / lag_max
    is_significant = bool(max_mi > threshold)
    
    return {
        "mi_profile": mi_profile,
        "lag_opt": lag_opt,
        "is_significant": is_significant,
        "threshold": threshold
    }
