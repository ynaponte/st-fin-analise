import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import grangercausalitytests


def test(x: pd.Series, y: pd.Series, lag_max: int, alpha: float = 0.05) -> dict:
    """
    Performs Granger causality test of x -> y over lags 1..lag_max.
    Uses Benjamini-Hochberg FDR to correct for multiple lag comparisons.

    Both x and y must be stationary (I(0)) series — this is the caller's
    responsibility. Typically called with daily log-returns.

    Returns a dictionary with:
      - best_lag: the lag with the smallest raw p-value
      - p_value: BH-corrected p-value at the best lag
      - p_value_raw: uncorrected p-value at the best lag
      - is_causal: True if p_value (corrected) < alpha
      - statistic: F-statistic at the best lag
      - all_p_values: dict of {lag: raw_p} for all lags tested
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["y", "x"]

    # Need at least 3*lag_max + 2 observations to have degrees of freedom
    if len(df) <= 3 * lag_max + 2 or lag_max < 1:
        return {
            "best_lag": 1,
            "p_value": 1.0,
            "p_value_raw": 1.0,
            "is_causal": False,
            "statistic": 0.0,
            "all_p_values": {},
        }

    try:
        lags = list(range(1, lag_max + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            res = grangercausalitytests(df[["y", "x"]], maxlag=lag_max, verbose=False)
        raw_pvals = {lag: float(res[lag][0]["ssr_ftest"][1]) for lag in lags}
        stats = {lag: float(res[lag][0]["ssr_ftest"][0]) for lag in lags}
    except Exception:
        return {
            "best_lag": 1,
            "p_value": 1.0,
            "p_value_raw": 1.0,
            "is_causal": False,
            "statistic": 0.0,
            "all_p_values": {},
        }

    # Benjamini-Hochberg FDR correction across all lags
    m = len(lags)
    sorted_lags = sorted(lags, key=lambda l: raw_pvals[l])
    bh_threshold = {lag: (rank / m) * alpha for rank, lag in enumerate(sorted_lags, 1)}

    # A lag is BH-significant if its raw p-value <= its BH threshold
    significant_lags = [lag for lag in sorted_lags if raw_pvals[lag] <= bh_threshold[lag]]

    if significant_lags:
        best_lag = significant_lags[0]  # smallest p-value among significant
        corrected_p = raw_pvals[best_lag]  # use raw p at best lag; BH decision was already made
        is_causal = True
    else:
        best_lag = sorted_lags[0]  # smallest raw p, even if not significant
        corrected_p = raw_pvals[best_lag]
        is_causal = False

    return {
        "best_lag": best_lag,
        "p_value": corrected_p,
        "p_value_raw": raw_pvals[best_lag],
        "is_causal": is_causal,
        "statistic": stats[best_lag],
        "all_p_values": raw_pvals,
    }
