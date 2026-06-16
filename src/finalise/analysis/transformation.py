"""
analysis.transformation
-----------------------
Time-series transformation utilities, including log-returns and seasonal-trend
decomposition (STL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


# ---------------------------------------------------------------------------
# STL Decomposition  (single series, strictly causal)
# ---------------------------------------------------------------------------

def stl_decomposition(
    series: pd.Series,
    period: int,
    step: int = 5,
    robust: bool = False,
) -> dict:
    """
    Causal STL (Seasonal-Trend decomposition using LOESS) of *series*.

    STL is refitted every *step* observations using only data up to that
    point.  Between fit points, trend and seasonal are forward-filled
    (still causal).  Residual is always recomputed as
    ``series − trend − seasonal`` so the additive identity holds exactly.

    Parameters
    ----------
    series : pd.Series – Input time series (any scale / frequency).
    period : int       – Dominant seasonal period (e.g. 5 for weekly, 252 for annual).
    step   : int       – Refit interval in observations (default 5).
    robust : bool      – Enable LOESS robustness iterations (down-weights outliers).

    Returns
    -------
    dict with keys:

    ``trend``    : pd.Series – Trend component.
    ``seasonal`` : pd.Series – Seasonal component.
    ``residual`` : pd.Series – Remainder = series − trend − seasonal.
    ``period``   : int       – Period used.
    """
    n        = len(series)
    trend    = np.full(n, np.nan)
    seasonal = np.full(n, np.nan)

    min_obs = max(10, 2 * period)
    fit_ts  = list(range(min_obs, n + 1, step))
    if n not in fit_ts:
        fit_ts.append(n)

    for t in fit_ts:
        res          = STL(series.iloc[:t], period=period, robust=robust).fit()
        trend[t - 1]    = float(res.trend.iloc[-1])
        seasonal[t - 1] = float(res.seasonal.iloc[-1])

    # Forward-fill between fit points
    trend    = pd.Series(trend).ffill().values.copy()
    seasonal = pd.Series(seasonal).ffill().values.copy()

    # Fill the initial window (before min_obs) with the raw series / zero
    nan_mask           = np.isnan(trend)
    trend[nan_mask]    = series.values[nan_mask]
    seasonal[nan_mask] = 0.0

    residual = series.values - trend - seasonal

    return {
        "trend":    pd.Series(trend,    index=series.index, name="trend"),
        "seasonal": pd.Series(seasonal, index=series.index, name="seasonal"),
        "residual": pd.Series(residual, index=series.index, name="residual"),
        "period":   period,
    }


def log_returns(
    prices: pd.Series,
    k: int = 1,
    overlapping: bool = False,
) -> pd.Series:
    """
    Computes logarithmic returns from a price series.

    For k = 1:
        r_1(t) = log(P(t) / P(t-1))

    For k > 1:
        Computes the k-period log-return, which mathematically equals the
        rolling sum of daily log-returns over window k.
        - If overlapping=False, returns non-overlapping sampled returns.
        - If overlapping=True, returns the rolling overlapping returns.

    Parameters
    ----------
    prices      : pd.Series
        The raw price series.
    k           : int
        The period window for returns (default 1).
    overlapping : bool
        If True and k > 1, returns rolling overlapping returns.
        If False and k > 1, returns non-overlapping returns.

    Returns
    -------
    pd.Series
        The calculated log-returns, with missing values (NaNs) dropped.
    """
    # Base daily log-returns
    r1 = np.log(prices / prices.shift(1))

    if k == 1:
        return r1.dropna()

    r1_clean = r1.dropna()

    # k-period return is the rolling sum of 1-period returns
    r_k = r1_clean.rolling(window=k).sum()

    if not overlapping:
        # Non-overlapping: sample every k-th element, starting from the first valid rolling sum
        return r_k.iloc[k - 1::k].dropna()
    else:
        # Overlapping: keep all rolling periods
        return r_k.dropna()
