import numpy as np
import pandas as pd
from finalise.target.entropy import ksg_mi
from rich.progress import track
from joblib import Parallel, delayed


def _cross_mi_perm_task(seed, x_lagged_opt, y_aligned_opt, k):
    rng = np.random.default_rng(seed)
    x_perm = rng.permutation(x_lagged_opt)
    return ksg_mi(x_perm, y_aligned_opt, k=k)


def _cross_mi_max_perm_task(seed, x_values, y_values, lag_max, k):
    """
    Permutation task for the max-MI null distribution.
    Shuffles x, then computes MI at every lag and returns the maximum.
    This naturally accounts for multiple testing across lags (no extra
    Bonferroni needed, because the null already models the max-MI statistic).
    """
    rng = np.random.default_rng(seed)
    x_perm = rng.permutation(x_values)
    max_perm_mi = 0.0
    for lag in range(1, lag_max + 1):
        n = len(y_values)
        if lag >= n:
            continue
        x_lag = x_perm[: n - lag]
        y_aln = y_values[lag:]
        if len(x_lag) <= k + 2:
            continue
        mi = ksg_mi(x_lag, y_aln, k=k)
        if mi > max_perm_mi:
            max_perm_mi = mi
    return max_perm_mi


def cross_mi_lags(
    x: pd.Series,
    y: pd.Series,
    lag_max: int,
    alpha: float = 0.05,
    k: int = 5,
    n_permutations: int = 200,
) -> dict:
    """
    Computes cross-mutual information between x(t-lag) and y(t) for lag = 1..lag_max.
    Uses Kraskov k-NN estimator.

    Significance is assessed via a max-statistic permutation test:
    the null distribution is the maximum MI across all lags under the permuted
    (independent) x, which implicitly corrects for multiple testing across lags
    without requiring an external Bonferroni factor.
    """
    # Align on common index first, then extract numpy arrays for speed
    df_full = pd.concat([y, x], axis=1).dropna()
    if len(df_full) <= lag_max + k + 2:
        return {
            "mi_profile": [0.0] * lag_max,
            "lag_opt": 1,
            "is_significant": False,
            "threshold_alpha": alpha,
            "p_value": 1.0,
        }

    y_vals = df_full.iloc[:, 0].values
    x_vals = df_full.iloc[:, 1].values
    n = len(y_vals)

    # Observed MI profile
    mi_profile = []
    for lag in range(1, lag_max + 1):
        if lag >= n:
            mi_profile.append(0.0)
            continue
        x_lag = x_vals[: n - lag]
        y_aln = y_vals[lag:]
        if len(x_lag) <= k + 2:
            mi_profile.append(0.0)
        else:
            mi_profile.append(ksg_mi(x_lag, y_aln, k=k))

    lag_opt = int(np.argmax(mi_profile) + 1) if mi_profile else 1
    max_mi = mi_profile[lag_opt - 1] if mi_profile else 0.0

    if max_mi <= 0.0:
        return {
            "mi_profile": mi_profile,
            "lag_opt": lag_opt,
            "is_significant": False,
            "threshold_alpha": alpha,
            "p_value": 1.0,
        }

    # Max-statistic permutation test
    seeds = np.random.default_rng(42).integers(0, 2**31, size=n_permutations)
    tasks = (
        delayed(_cross_mi_max_perm_task)(seed, x_vals, y_vals, lag_max, k)
        for seed in seeds
    )
    null_max_mis = []
    for mi_perm in track(
        Parallel(n_jobs=-1, return_as="generator")(tasks),
        total=n_permutations,
        description="[Preditor] Permutações Cross-MI",
    ):
        null_max_mis.append(mi_perm)

    # Phipson-Smyth corrected p-value (avoids p=0)
    count = sum(1 for v in null_max_mis if v >= max_mi)
    p_value = float((count + 1) / (n_permutations + 1))
    is_significant = bool(p_value < alpha)

    return {
        "mi_profile": mi_profile,
        "lag_opt": lag_opt,
        "is_significant": is_significant,
        "threshold_alpha": alpha,
        "p_value": p_value,
    }
