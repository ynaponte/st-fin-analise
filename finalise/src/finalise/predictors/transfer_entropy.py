import numpy as np
import pandas as pd
from typing import Union, List

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

def compute_te(x: pd.Series, y: pd.Series, lag: int, y_lags: Union[int, list[int], None] = 1) -> float:
    """
    Computes Transfer Entropy from x -> y at lag using Freedman-Diaconis bins,
    conditioning on the specified past lags of y (y_lags).
    """
    if isinstance(y_lags, int):
        lags_list = list(range(1, y_lags + 1))
    elif isinstance(y_lags, list):
        lags_list = y_lags
    else:
        lags_list = [1]
        
    cols = [y]
    col_names = ["y_t"]
    for l in lags_list:
        cols.append(y.shift(l))
        col_names.append(f"y_lag_{l}")
    cols.append(x.shift(lag))
    col_names.append("x_lag")
    
    df = pd.concat(cols, axis=1)
    df.columns = col_names
    df = df.dropna()
    
    if len(df) < 10:
        return 0.0
        
    from finalise.target.entropy import fd_bins
    bx = fd_bins(df["x_lag"])
    by_t = fd_bins(df["y_t"])
    
    x_binned = bin_series(df["x_lag"], bx)
    y_t_binned = bin_series(df["y_t"], by_t)
    y_lags_binned = [bin_series(df[f"y_lag_{l}"], by_t) for l in lags_list]
    
    h_y_t_y_past = entropy_joint([y_t_binned] + y_lags_binned)
    h_y_past = entropy_joint(y_lags_binned)
    h_y_t_y_past_x_past = entropy_joint([y_t_binned] + y_lags_binned + [x_binned])
    h_y_past_x_past = entropy_joint(y_lags_binned + [x_binned])
    
    te = h_y_t_y_past - h_y_past - h_y_t_y_past_x_past + h_y_past_x_past
    return float(max(0.0, te))

def compute(
    x: pd.Series,
    y: pd.Series,
    lag: int,
    alpha: float = 0.05,
    n_permutations: int = 500,
    y_lags: Union[int, list[int], None] = 1
) -> dict:
    """
    Computes Transfer Entropy from x -> y at lag.
    Runs a permutation test (N=n_permutations) to determine significance.
    Returns a dictionary.
    """
    te_obs = compute_te(x, y, lag, y_lags=y_lags)
    
    if isinstance(y_lags, int):
        lags_list = list(range(1, y_lags + 1))
    elif isinstance(y_lags, list):
        lags_list = y_lags
    else:
        lags_list = [1]
        
    cols = [y]
    col_names = ["y_t"]
    for l in lags_list:
        cols.append(y.shift(l))
        col_names.append(f"y_lag_{l}")
    cols.append(x.shift(lag))
    col_names.append("x_lag")
    
    df = pd.concat(cols, axis=1)
    df.columns = col_names
    df = df.dropna()
    
    if len(df) < 10:
        return {
            "te": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "null_distribution": [0.0] * n_permutations,
            "threshold": 0.0
        }
        
    from finalise.target.entropy import fd_bins
    bx = fd_bins(df["x_lag"])
    by_t = fd_bins(df["y_t"])
    
    x_binned = bin_series(df["x_lag"], bx)
    y_t_binned = bin_series(df["y_t"], by_t)
    y_lags_binned = [bin_series(df[f"y_lag_{l}"], by_t) for l in lags_list]
    
    h_y_t_y_past = entropy_joint([y_t_binned] + y_lags_binned)
    h_y_past = entropy_joint(y_lags_binned)
    
    null_dist = []
    rng = np.random.default_rng(42)  # Seed for reproducibility
    for _ in range(n_permutations):
        x_perm = rng.permutation(x_binned)
        h_joint3 = entropy_joint([y_t_binned] + y_lags_binned + [x_perm])
        h_joint2 = entropy_joint(y_lags_binned + [x_perm])
        te_perm = h_y_t_y_past - h_y_past - h_joint3 + h_joint2
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
