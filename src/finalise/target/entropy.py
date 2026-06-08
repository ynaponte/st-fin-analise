import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma
from rich.progress import track
from joblib import Parallel, delayed


def _auto_mi_perm_task(seed, series_values, lag_max, k):
    rng = np.random.default_rng(seed)
    perm_series = rng.permutation(series_values)
    perm_profile = []
    for lag in range(1, lag_max + 1):
        x = perm_series[lag:]
        y = perm_series[:-lag]
        perm_profile.append(ksg_mi(x, y, k=k))
    return max(perm_profile) if perm_profile else 0.0


def fd_bins(series: pd.Series) -> int:
    """
    Estimates the number of bins using the Freedman-Diaconis rule.
    """
    x = np.asarray(series)
    n = len(x)
    if n <= 1:
        return 1
    val_range = np.max(x) - np.min(x)
    if val_range < 1e-12:
        return 1
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        # Fallback to Sturges rule if IQR is 0
        return int(np.ceil(np.log2(n) + 1))
    h = 2 * iqr * (n ** (-1/3))
    if h == 0:
        return 1
    return int(np.clip(np.ceil(val_range / h), 1, 1000))

def shannon(series: pd.Series) -> float:
    """
    Calculates the normalized Shannon entropy of the series in nats,
    estimating bins with the Freedman-Diaconis rule.
    Normalized by log(n_bins) if n_bins > 1 to allow comparison across scales.
    """
    x = np.asarray(series)
    n_bins = fd_bins(series)
    if n_bins <= 1:
        return 0.0
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / len(x)
    probs = probs[probs > 0]
    h_discrete = -np.sum(probs * np.log(probs))
    return float(max(0.0, h_discrete / np.log(n_bins)))

def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Calculates the Mutual Information between x and y using the
    Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator.
    """
    x = x.astype(float).reshape(-1, 1)
    y = y.astype(float).reshape(-1, 1)
    N = len(x)
    if N <= k + 1:
        return 0.0
        
    # Add a tiny jitter to break ties/duplicates (standard KSG practice)
    rng = np.random.default_rng(42)
    x_std = np.std(x)
    y_std = np.std(y)
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x_std if x_std > 0 else 1.0)
    y = y + 1e-10 * rng.standard_normal(y.shape) * (y_std if y_std > 0 else 1.0)
    
    # Combined joint space Z = (X, Y)
    xy = np.hstack([x, y])
    
    # K-D Trees for joint and marginal spaces using Chebyshev (infinity) norm
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    
    # Distance to the k-th nearest neighbor (query returns point itself as 1st, so query k+1)
    distances = tree_xy.query(xy, k + 1, p=np.inf)[0][:, -1]
    
    eps = distances - 1e-15
    eps = np.clip(eps, 1e-15, None)
    
    nx_indices = tree_x.query_ball_point(x, eps, p=np.inf)
    ny_indices = tree_y.query_ball_point(y, eps, p=np.inf)
    
    nx = np.array([len(indices) - 1 for indices in nx_indices])
    ny = np.array([len(indices) - 1 for indices in ny_indices])
        
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return float(max(0.0, mi))

def auto_mi(series: pd.Series, lag_max: int, k: int = 5) -> tuple[list[float], int]:
    """
    Computes auto-mutual information of series with its lagged versions.
    Returns:
      - mi_profile: list of MI values for lags 1..lag_max.
      - opt_lag: the lag that maximizes MI.
    """
    mi_profile = []
    series_values = np.asarray(series)
    for lag in range(1, lag_max + 1):
        if len(series_values) <= lag + k + 1:
            mi_profile.append(0.0)
            continue
        x = series_values[lag:]
        y = series_values[:-lag]
        mi_profile.append(ksg_mi(x, y, k=k))
        
    opt_lag = int(np.argmax(mi_profile) + 1) if len(mi_profile) > 0 else 1
    return mi_profile, opt_lag

def auto_mi_significance(
    series: pd.Series,
    lag_max: int,
    max_mi: float,
    alpha: float = 0.05,
    k: int = 5,
    n_permutations: int = 100
) -> tuple[float, bool]:
    """
    Runs a permutation test to determine the significance of the maximum auto-MI.
    Under the null hypothesis of independence, shuffles the series and computes the max auto-MI.
    Returns:
      - p_value: empirical p-value of the observed maximum auto-MI.
      - is_significant: True if p_value < alpha.
    """
    if len(series) <= lag_max + k + 1 or max_mi <= 0.0:
        return 1.0, False
        
    rng = np.random.default_rng(42)
    series_values = np.asarray(series)
    
    seeds = np.random.default_rng(42).integers(0, 2**31, size=n_permutations)
    max_null_mis = []
    
    tasks = (delayed(_auto_mi_perm_task)(seed, series_values, lag_max, k) for seed in seeds)
    for result in track(Parallel(n_jobs=-1, return_as="generator")(tasks), total=n_permutations, description="[Target] Permutações Auto-MI"):
        max_null_mis.append(result)
        
    # Phipson-Smyth correction: avoids p=0 and is consistent with mi.py / transfer_entropy.py
    count = sum(1 for v in max_null_mis if v >= max_mi)
    p_value = float((count + 1) / (len(max_null_mis) + 1))
    is_significant = bool(p_value < alpha)
    return p_value, is_significant
