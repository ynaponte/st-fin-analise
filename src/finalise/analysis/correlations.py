import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf


# ---------------------------------------------------------------------------
# Cross-series correlations
# ---------------------------------------------------------------------------

def pearson(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """
    Computes the Pearson product-moment correlation coefficient between *x*
    and *y* after dropping rows where either series is NaN.

    Parameters
    ----------
    x, y : pd.Series – The two series to correlate.

    Returns
    -------
    dict with keys:
        ``correlation`` : float  – Pearson r ∈ [-1, 1].
        ``p_value``     : float  – Two-tailed p-value under H₀: r = 0.
        ``n``           : int    – Number of observations used.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return {"correlation": float("nan"), "p_value": float("nan"), "n": len(df)}

    r, p = stats.pearsonr(df["x"].values, df["y"].values)
    return {"correlation": float(r), "p_value": float(p), "n": len(df)}


def spearman(
    x: pd.Series,
    y: pd.Series,
) -> dict:
    """
    Computes the Spearman rank-order correlation coefficient between *x* and
    *y* after dropping rows where either series is NaN.

    Parameters
    ----------
    x, y : pd.Series – The two series to correlate.

    Returns
    -------
    dict with keys:
        ``correlation`` : float  – Spearman ρ ∈ [-1, 1].
        ``p_value``     : float  – Two-tailed p-value under H₀: ρ = 0.
        ``n``           : int    – Number of observations used.
    """
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return {"correlation": float("nan"), "p_value": float("nan"), "n": len(df)}

    rho, p = stats.spearmanr(df["x"].values, df["y"].values)
    return {"correlation": float(rho), "p_value": float(p), "n": len(df)}


# ---------------------------------------------------------------------------
# Autocorrelation / Partial Autocorrelation
# ---------------------------------------------------------------------------

def acf_pacf(
    series: pd.Series,
    lags: int,
    alpha: float = 0.05,
) -> dict:
    """
    Computes ACF and PACF for *series* and for the squared series (a proxy
    for volatility clustering / nonlinear dependence).

    The maximum number of lags is clamped to ``len(series) // 2 − 1`` to
    avoid numerical errors in statsmodels for short series.

    Parameters
    ----------
    series : pd.Series – Time series to analyse.
    lags   : int       – Desired number of lags.
    alpha  : float     – Confidence level for the confidence bands (default 5 %).

    Returns
    -------
    dict with keys:
        ``acf``              : list[float] – ACF coefficients (lag 0 … lags).
        ``acf_confint``      : list        – Lower/upper confidence intervals.
        ``pacf``             : list[float] – PACF coefficients.
        ``pacf_confint``     : list        – Lower/upper confidence intervals.
        ``acf_sq``           : list[float] – ACF of squared series.
        ``acf_sq_confint``   : list        – Lower/upper confidence intervals.
        ``pacf_sq``          : list[float] – PACF of squared series.
        ``pacf_sq_confint``  : list        – Lower/upper confidence intervals.
        ``effective_lags``   : int         – Actual number of lags computed.
    """
    series_clean = series.dropna()
    n = len(series_clean)

    effective_lags = min(lags, n // 2 - 1)
    if effective_lags < 1:
        effective_lags = 1

    acf_vals,  acf_conf  = acf( series_clean, nlags=effective_lags, alpha=alpha)
    pacf_vals, pacf_conf = pacf(series_clean, nlags=effective_lags, alpha=alpha)

    series_sq = series_clean ** 2
    acf_sq_vals,  acf_sq_conf  = acf( series_sq, nlags=effective_lags, alpha=alpha)
    pacf_sq_vals, pacf_sq_conf = pacf(series_sq, nlags=effective_lags, alpha=alpha)

    return {
        "acf":             acf_vals.tolist(),
        "acf_confint":     acf_conf.tolist(),
        "pacf":            pacf_vals.tolist(),
        "pacf_confint":    pacf_conf.tolist(),
        "acf_sq":          acf_sq_vals.tolist(),
        "acf_sq_confint":  acf_sq_conf.tolist(),
        "pacf_sq":         pacf_sq_vals.tolist(),
        "pacf_sq_confint": pacf_sq_conf.tolist(),
        "effective_lags":  effective_lags,
    }
