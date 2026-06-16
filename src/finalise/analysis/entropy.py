import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.special import digamma
from typing import Union


# ---------------------------------------------------------------------------
# Low-level estimators (shared utilities)
# ---------------------------------------------------------------------------

def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Kraskov-Stögbauer-Grassberger (KSG) k-NN mutual information estimator.
    Result is in nats; convert to bits by dividing by ln(2).

    Parameters
    ----------
    x, y : 1-D numpy arrays of equal length.
    k    : Number of nearest neighbours.

    Returns
    -------
    mi : float – MI in nats (≥ 0).
    """
    x = x.astype(float).reshape(-1, 1)
    y = y.astype(float).reshape(-1, 1)
    N = len(x)
    if N <= k + 1:
        return 0.0

    rng = np.random.default_rng(42)
    x_std = np.std(x)
    y_std = np.std(y)
    x = x + 1e-10 * rng.standard_normal(x.shape) * (x_std if x_std > 0 else 1.0)
    y = y + 1e-10 * rng.standard_normal(y.shape) * (y_std if y_std > 0 else 1.0)

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x  = cKDTree(x)
    tree_y  = cKDTree(y)

    distances = tree_xy.query(xy, k + 1, p=np.inf)[0][:, -1]
    eps = np.clip(distances - 1e-15, 1e-15, None)

    nx = np.array([len(idx) - 1 for idx in tree_x.query_ball_point(x, eps, p=np.inf)])
    ny = np.array([len(idx) - 1 for idx in tree_y.query_ball_point(y, eps, p=np.inf)])

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return float(max(0.0, mi))


def _bin_series(series: pd.Series, n_bins: int) -> np.ndarray:
    """Bins a series into integer indices 0 … n_bins-1."""
    min_val, max_val = series.min(), series.max()
    if min_val == max_val:
        return np.zeros(len(series), dtype=int)
    edges = np.linspace(min_val, max_val, n_bins + 1)
    bins = np.digitize(series.values, edges) - 1
    return np.clip(bins, 0, n_bins - 1)


def _fd_bins(series: pd.Series) -> int:
    """Freedman-Diaconis bin count estimator."""
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
        return int(np.ceil(np.log2(n) + 1))
    h = 2 * iqr * (n ** (-1 / 3))
    if h == 0:
        return 1
    return int(np.clip(np.ceil(val_range / h), 1, 1000))


def _entropy_joint(arrays: list[np.ndarray]) -> float:
    """Shannon joint entropy (nats) of a list of 1-D binned arrays."""
    n = len(arrays[0])
    stacked = np.column_stack(arrays)
    _, counts = np.unique(stacked, axis=0, return_counts=True)
    probs = counts / n
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mutual_information_lags(
    target: pd.Series,
    test: pd.Series,
    max_lag: int,
    k: int = 5,
) -> list[tuple[int, float]]:
    """
    Calculates Mutual Information (MI) between *target* and lagged versions of
    *test*, iterating from lag 1 up to *max_lag*.

    MI is estimated with the KSG k-NN estimator and expressed in **bits**
    (nats ÷ ln 2).  No cross-permutation / permutation testing is performed
    here; this function simply returns the observed MI at every lag.

    Parameters
    ----------
    target  : pd.Series – The reference series (y).
    test    : pd.Series – The series to be lagged (x).
    max_lag : int       – Maximum lag to evaluate (inclusive).
    k       : int       – Number of nearest neighbours for the KSG estimator.

    Returns
    -------
    List of (lag, mi_bits) tuples for lag = 1 … max_lag.
    """
    # Align on a common index once; subsequent lags shift only test.
    df_base = pd.concat([target.rename("target"), test.rename("test")], axis=1).dropna()
    y_vals  = df_base["target"].values
    x_vals  = df_base["test"].values
    n       = len(y_vals)

    ln2 = np.log(2)
    results: list[tuple[int, float]] = []

    for lag in range(1, max_lag + 1):
        if lag >= n:
            results.append((lag, 0.0))
            continue

        # x(t-lag) aligned with y(t): drop the first `lag` observations of y
        # and the last `lag` observations of x.
        x_lagged = x_vals[: n - lag]
        y_aligned = y_vals[lag:]

        if len(x_lagged) <= k + 2:
            results.append((lag, 0.0))
            continue

        mi_nats = _ksg_mi(x_lagged, y_aligned, k=k)
        mi_bits = mi_nats / ln2
        results.append((lag, float(mi_bits)))

    return results


def transfer_entropy(
    x: pd.Series,
    y: pd.Series,
    lag: int = 1,
    y_lags: Union[int, list[int]] = 1,
) -> float:
    """
    Calculates Transfer Entropy (TE) from *x* → *y* at a given *lag*.

    TE measures the directed information flow from x to y beyond what y's
    own past already provides.  It is estimated via plug-in Shannon entropy
    on histogram-binned data (Freedman-Diaconis bin widths) and expressed
    in **nats**.

    TE(X→Y) = H(Y_t | Y_past) − H(Y_t | Y_past, X_past)
             = H(Y_t, Y_past) − H(Y_past) − H(Y_t, Y_past, X_past) + H(Y_past, X_past)

    Parameters
    ----------
    x      : pd.Series – Source / driver series.
    y      : pd.Series – Target / driven series.
    lag    : int       – How many steps x is lagged when conditioning on it.
    y_lags : int | list[int]
               If int  → conditions on y at lags 1 … y_lags.
               If list → conditions on y at exactly those lags.

    Returns
    -------
    te : float – Transfer entropy in nats (≥ 0).
    """
    if isinstance(y_lags, int):
        lags_list = list(range(1, y_lags + 1))
    else:
        lags_list = list(y_lags)

    cols      = [y.rename("y_t")]
    col_names = ["y_t"]
    for l in lags_list:
        cols.append(y.shift(l).rename(f"y_lag_{l}"))
        col_names.append(f"y_lag_{l}")
    cols.append(x.shift(lag).rename("x_lag"))
    col_names.append("x_lag")

    df = pd.concat(cols, axis=1)
    df.columns = col_names
    df = df.dropna()

    if len(df) < 10:
        return 0.0

    bx   = _fd_bins(df["x_lag"])
    by_t = _fd_bins(df["y_t"])

    x_binned     = _bin_series(df["x_lag"], bx)
    y_t_binned   = _bin_series(df["y_t"],   by_t)
    y_lags_binned = [_bin_series(df[f"y_lag_{l}"], by_t) for l in lags_list]

    h_y_t_y_past         = _entropy_joint([y_t_binned]   + y_lags_binned)
    h_y_past             = _entropy_joint(y_lags_binned)
    h_y_t_y_past_x_past  = _entropy_joint([y_t_binned]   + y_lags_binned + [x_binned])
    h_y_past_x_past      = _entropy_joint(y_lags_binned  + [x_binned])

    te_nats = h_y_t_y_past - h_y_past - h_y_t_y_past_x_past + h_y_past_x_past
    te_bits = te_nats / np.log(2)
    return float(max(0.0, te_bits))


def jensen_shannon_divergence(
    series_list: list[pd.Series],
    bins: Union[int, str] = "fd",
) -> pd.DataFrame:
    """
    Calculates the pairwise Jensen-Shannon Divergence (JSD) between all series
    in *series_list*.
    
    The JSD is computed based on probability distributions estimated using
    histograms over a common range spanning all series.

    Parameters
    ----------
    series_list : list[pd.Series]
        List of series to compare.
    bins : int | str
        Number of bins for the histogram, or "fd" to use the maximum
        Freedman-Diaconis bin count among all series.
        
    Returns
    -------
    pd.DataFrame
        Symmetric matrix (k x k) containing the JSD (in bits) between all pairs.
    """
    named_series = []
    for i, s in enumerate(series_list):
        named_series.append(s.rename(s.name if s.name is not None else f"series_{i}"))
        
    df = pd.concat(named_series, axis=1).dropna()
    names = df.columns.tolist()
    k = len(names)
    
    if k < 2:
        return pd.DataFrame(index=names, columns=names)
        
    global_min = df.min().min()
    global_max = df.max().max()
    
    if global_min == global_max:
        return pd.DataFrame(0.0, index=names, columns=names)
    
    if bins == "fd":
        n_bins = max(_fd_bins(df[col]) for col in names)
    else:
        n_bins = int(bins)
        
    edges = np.linspace(global_min, global_max, n_bins + 1)
    
    distributions = []
    for col in names:
        counts, _ = np.histogram(df[col], bins=edges)
        p = counts / counts.sum()
        distributions.append(p)
        
    jsd_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                jsd_matrix[i, j] = 0.0
            else:
                # scipy jensenshannon returns the distance. Divergence is distance^2.
                # Base 2 for bits.
                js_dist = jensenshannon(distributions[i], distributions[j], base=2)
                js_div = float(js_dist ** 2)
                jsd_matrix[i, j] = js_div
                jsd_matrix[j, i] = js_div
                
    return pd.DataFrame(jsd_matrix, index=names, columns=names)
