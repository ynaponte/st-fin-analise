import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma

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
    Calculates the Shannon entropy of the series in nats,
    estimating bins with the Freedman-Diaconis rule.
    """
    x = np.asarray(series)
    n_bins = fd_bins(series)
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / len(x)
    probs = probs[probs > 0]
    return float(max(0.0, -np.sum(probs * np.log(probs))))

def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Calculates the Mutual Information between x and y using the
    Kraskov-Stögbauer-Grassberger (KSG) k-NN estimator.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    N = len(x)
    
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
