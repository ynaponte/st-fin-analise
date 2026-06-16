"""
finalise.smoothing.methods
--------------------------
Smoothing algorithms implemented via the Strategy (Policy) pattern,
incorporating automatic group-delay compensation.
"""

from typing import Protocol
import pandas as pd
import numpy as np
from scipy.signal import group_delay as scipy_group_delay
from scipy.signal.windows import gaussian


class SmoothingPolicy(Protocol):
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        """Applies the specific smoothing algorithm."""
        ...
        
    def group_delay(self, window: int) -> float:
        """Calculates the group delay (in samples) for this window size."""
        ...


class SMAPolicy:
    """Simple Moving Average (SMA)"""
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=1).mean()
        
    def group_delay(self, window: int) -> float:
        return (window - 1) / 2.0


class EMAPolicy:
    """Exponential Moving Average (EMA)"""
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False, min_periods=1).mean()
        
    def group_delay(self, window: int) -> float:
        alpha = 2 / (window + 1)
        b = [alpha]
        a = [1, -(1 - alpha)]
        _, gd = scipy_group_delay((b, a), w=[0.0])
        return float(gd[0])


class TMAPolicy:
    """Triangular Moving Average (TMA) - effectively an SMA of an SMA"""
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        w1 = (window + 1) // 2
        w2 = window // 2 + 1 if window % 2 == 0 else w1
        return series.rolling(window=w1, min_periods=1).mean().rolling(window=w2, min_periods=1).mean()
        
    def group_delay(self, window: int) -> float:
        w1 = (window + 1) // 2
        w2 = window // 2 + 1 if window % 2 == 0 else w1
        b = np.convolve(np.ones(w1) / w1, np.ones(w2) / w2)
        _, gd = scipy_group_delay((b, [1]), w=[0.0])
        return float(gd[0])


class GaussianPolicy:
    """Gaussian Weighted Moving Average"""
    def apply(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, win_type='gaussian', min_periods=1).mean(std=window / 6.0)
        
    def group_delay(self, window: int) -> float:
        b = gaussian(window, std=window / 6.0)
        b /= b.sum()
        _, gd = scipy_group_delay((b, [1]), w=[0.0])
        return float(gd[0])


def apply_smoothing(series: pd.Series, window: int, policy: SmoothingPolicy) -> pd.Series:
    """
    Applies the smoothing policy to the series and shifts the result backwards
    to completely compensate for the filter's phase/group delay.

    Parameters
    ----------
    series : pd.Series
        The time series to be smoothed.
    window : int
        The window size for the smoothing filter.
    policy : SmoothingPolicy
        The policy object implementing the smoothing logic (e.g., SMAPolicy()).

    Returns
    -------
    pd.Series
        The smoothed series, delay-compensated (non-causal).
    """
    smoothed = policy.apply(series, window)
    
    # Calculate group delay and round to nearest integer sample shift
    delay = int(round(policy.group_delay(window)))
    
    # Shift backwards (negative shift) to align smoothed curve with original data
    shifted = smoothed.shift(-delay)
    
    return shifted
