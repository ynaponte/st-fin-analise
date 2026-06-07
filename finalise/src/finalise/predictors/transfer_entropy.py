import numpy as np
import pandas as pd

def bin_series(series: pd.Series, n_bins: int) -> np.ndarray:
    """
    Bins series values into integer indices 0..n_bins-1.
    """
    min_val, max_val = series.min(), series.max()
    if min_val == max_val:
        return np.zeros(len(series), dtype=int)
    edges = np.linspace(min_val, max_val, n_bins + 1)
    bins = np.digitize(series.values, edges) - 1
    return np.clip(bins, 0, n_bins - 1)

def entropy_joint(arrays: list[np.ndarray]) -> float:
    """
    Calculates joint Shannon entropy of a list of 1D binned arrays.
    """
    n = len(arrays[0])
    stacked = np.column_stack(arrays)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def compute_te(x: pd.Series, y: pd.Series, lag: int) -> float:
    """
    Computes Transfer Entropy from x -> y at lag using Freedman-Diaconis bins.
    """
    df = pd.concat([y, y.shift(1), x.shift(lag)], axis=1).dropna()
    if len(df) < 10:
        return 0.0
        
    y_t = df.iloc[:, 0]
    y_lag = df.iloc[:, 1]
    x_lag = df.iloc[:, 2]
    
    from finalise.target.entropy import fd_bins
    bx = fd_bins(x_lag)
    by = fd_bins(y_t)
    
    x_binned = bin_series(x_lag, bx)
    y_t_binned = bin_series(y_t, by)
    y_lag_binned = bin_series(y_lag, by)
    
    h_y_t_y_lag = entropy_joint([y_t_binned, y_lag_binned])
    h_y_lag = entropy_joint([y_lag_binned])
    h_y_t_y_lag_x_lag = entropy_joint([y_t_binned, y_lag_binned, x_binned])
    h_y_lag_x_lag = entropy_joint([y_lag_binned, x_binned])
    
    te = h_y_t_y_lag - h_y_lag - h_y_t_y_lag_x_lag + h_y_lag_x_lag
    return float(max(0.0, te))

def compute(x: pd.Series, y: pd.Series, lag: int, alpha: float = 0.05, n_permutations: int = 500) -> dict:
    """
    Computes Transfer Entropy from x -> y at lag.
    Runs a permutation test (N=n_permutations) to determine significance.
    Returns a dictionary.
    """
    te_obs = compute_te(x, y, lag)
    
    df = pd.concat([y, y.shift(1), x.shift(lag)], axis=1).dropna()
    if len(df) < 10:
        return {
            "te": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "null_distribution": [0.0] * n_permutations,
            "threshold": 0.0
        }
        
    y_t = df.iloc[:, 0]
    y_lag = df.iloc[:, 1]
    x_lag = df.iloc[:, 2]
    
    from finalise.target.entropy import fd_bins
    bx = fd_bins(x_lag)
    by = fd_bins(y_t)
    
    x_binned = bin_series(x_lag, bx)
    y_t_binned = bin_series(y_t, by)
    y_lag_binned = bin_series(y_lag, by)
    
    h_y_t_y_lag = entropy_joint([y_t_binned, y_lag_binned])
    h_y_lag = entropy_joint([y_lag_binned])
    
    null_dist = []
    rng = np.random.default_rng(42)  # Seed for reproducibility
    for _ in range(n_permutations):
        x_perm = rng.permutation(x_binned)
        h_joint3 = entropy_joint([y_t_binned, y_lag_binned, x_perm])
        h_joint2 = entropy_joint([y_lag_binned, x_perm])
        te_perm = h_y_t_y_lag - h_y_lag - h_joint3 + h_joint2
        null_dist.append(max(0.0, te_perm))
        
    threshold = float(np.percentile(null_dist, (1.0 - alpha) * 100))
    is_significant = bool(te_obs > threshold)
    p_val = float(np.mean(np.array(null_dist) >= te_obs))
    
    return {
        "te": te_obs,
        "p_value": p_val,
        "is_significant": is_significant,
        "null_distribution": null_dist,
        "threshold": threshold
    }
