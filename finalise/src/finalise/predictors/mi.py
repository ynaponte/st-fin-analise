import numpy as np
import pandas as pd
from finalise.target.entropy import ksg_mi

def cross_mi_lags(
    x: pd.Series,
    y: pd.Series,
    lag_max: int,
    alpha: float = 0.05,
    k: int = 5,
    n_permutations: int = 100
) -> dict:
    """
    Computes cross-mutual information between x(t-lag) and y(t) for lag = 1..lag_max.
    Uses Kraskov k-NN estimator.
    Applies Bonferroni correction on the permutation-based p-value of the optimal lag:
    threshold = alpha / lag_max.
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
    
    # Align the optimal lag for the permutation test
    df_opt = pd.concat([y, x.shift(lag_opt)], axis=1).dropna()
    if len(df_opt) <= k + 2 or max_mi <= 0.0:
        p_value = 1.0
        is_significant = False
    else:
        x_lagged_opt = df_opt.iloc[:, 1].values
        y_aligned_opt = df_opt.iloc[:, 0].values
        
        # Permutation test at optimal lag
        rng = np.random.default_rng(42)
        count = 0
        for _ in range(n_permutations):
            x_perm = rng.permutation(x_lagged_opt)
            mi_perm = ksg_mi(x_perm, y_aligned_opt, k=k)
            if mi_perm >= max_mi:
                count += 1
        p_value = float((count + 1) / (n_permutations + 1))
        # Bonferroni corrected threshold for p-value: alpha / lag_max
        is_significant = bool(p_value < (alpha / lag_max))
        
    return {
        "mi_profile": mi_profile,
        "lag_opt": lag_opt,
        "is_significant": is_significant,
        "threshold": alpha / lag_max,
        "p_value": p_value
    }
