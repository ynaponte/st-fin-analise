import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl(series: pd.Series, period: int, step: int = 5) -> dict:
    """
    Computes a causal STL decomposition of the raw price series.
    Returns a dictionary of pandas Series: {'trend', 'seasonal', 'residual'}.
    No value at time t uses data from after time t.

    To reduce the O(n²) wall-clock time, STL is refitted every `step`
    observations (default: 5). Between fit points, trend and seasonal are
    forward-filled — strictly causal, since fit at t uses only series[:t].
    Residual is recomputed as (series - trend - seasonal) at every point to
    preserve the additive identity exactly.
    """
    n = len(series)
    trend = np.full(n, np.nan)
    seasonal = np.full(n, np.nan)

    # STL requires at least 2 * period observations
    min_obs = max(10, 2 * period)

    # Fit STL at every `step`-th point; always include the final observation
    fit_ts = list(range(min_obs, n + 1, step))
    if n not in fit_ts:
        fit_ts.append(n)

    for t in fit_ts:
        res = STL(series.iloc[:t], period=period, robust=False).fit()
        trend[t - 1] = float(res.trend.iloc[-1])
        seasonal[t - 1] = float(res.seasonal.iloc[-1])

    # Forward-fill gaps between fit points (causal: each gap index inherits
    # the value from the most recent fit, which used no future data)
    trend = pd.Series(trend).ffill().values.copy()
    seasonal = pd.Series(seasonal).ffill().values.copy()

    # Fill starting positions (before min_obs) so that trend + seasonal + residual = series
    nan_mask = np.isnan(trend)
    trend[nan_mask] = series.values[nan_mask]
    seasonal[nan_mask] = 0.0

    # Residual is always exact: guarantees additive identity at every point
    residual = series.values - trend - seasonal

    return {
        "trend": pd.Series(trend, index=series.index),
        "seasonal": pd.Series(seasonal, index=series.index),
        "residual": pd.Series(residual, index=series.index),
    }

