"""
analysis.base
-------------
Multivariate time-series tests: Granger causality and Johansen cointegration.
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------------------------------------------------------
# Granger Causality  (target vs multiple drivers)
# ---------------------------------------------------------------------------

def granger_causality(
    target: pd.Series,
    tests: list[pd.Series],
    lag_max: int,
    alpha: float = 0.05,
) -> dict[str, dict]:
    """
    Tests Granger causality for every candidate series in *tests* against
    the *target* series (driver → target).

    At each lag the F-statistic from ``ssr_ftest`` is used.  Multiple-testing
    across lags for a single pair is controlled with Benjamini-Hochberg FDR
    correction.

    Parameters
    ----------
    target  : pd.Series
        The target series (y).
    tests   : list[pd.Series]
        List of candidate driver series (x).
    lag_max : int
        Maximum lag to test (inclusive, ≥ 1).
    alpha   : float
        Family-wise error rate for BH correction (default 5 %).

    Returns
    -------
    dict keyed by driver name.  Each value is a dict with:

    ``best_lag``     : int   – Lag with the smallest raw p-value.
    ``p_value``      : float – Raw p-value at the best lag.
    ``is_causal``    : bool  – True if the best lag survives BH correction.
    ``statistic``    : float – F-statistic at the best lag.
    ``all_p_values`` : dict  – ``{lag: raw_p}`` for every lag tested.
    """
    target_named = target.rename(target.name if target.name is not None else "target")

    results: dict[str, dict] = {}

    for i, driver in enumerate(tests):
        driver_name = str(driver.name if driver.name is not None else f"test_{i}")
        driver_named = driver.rename("driver")

        _empty = {
            "best_lag":     1,
            "p_value":      1.0,
            "is_causal":    False,
            "statistic":    0.0,
            "all_p_values": {},
            "all_f_stats":  {},
        }

        df = pd.concat([target_named.rename("target"), driver_named], axis=1).dropna()

        min_obs = 3 * lag_max + 2
        if lag_max < 1 or len(df) <= min_obs:
            results[driver_name] = _empty
            continue

        lags = list(range(1, lag_max + 1))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = grangercausalitytests(
                    df[["target", "driver"]], maxlag=lag_max, verbose=False
                )
            raw_pvals = {lag: float(raw[lag][0]["ssr_ftest"][1]) for lag in lags}
            f_stats   = {lag: float(raw[lag][0]["ssr_ftest"][0]) for lag in lags}
        except Exception:
            results[driver_name] = _empty
            continue

        # Benjamini-Hochberg FDR across lags
        m           = len(lags)
        ranked      = sorted(lags, key=lambda l: raw_pvals[l])
        bh_thresh   = {lag: (rank / m) * alpha for rank, lag in enumerate(ranked, 1)}
        sig_lags    = [lag for lag in ranked if raw_pvals[lag] <= bh_thresh[lag]]

        best_lag  = sig_lags[0] if sig_lags else ranked[0]
        is_causal = bool(sig_lags)

        results[driver_name] = {
            "best_lag":     best_lag,
            "p_value":      raw_pvals[best_lag],
            "is_causal":    is_causal,
            "statistic":    f_stats[best_lag],
            "all_p_values": raw_pvals,
            "all_f_stats":  f_stats,
        }

    return results


# ---------------------------------------------------------------------------
# Johansen Cointegration  (full multivariate system)
# ---------------------------------------------------------------------------

def johansen_cointegration(
    series: list[pd.Series],
    lag_order: int = 1,
    det_order: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Performs the Johansen cointegration test on a system of *k* ≥ 2 series.

    The Trace statistic is used to determine the cointegrating rank *r*:
    starting from the null H₀: rank = 0 and stepping up until we fail to
    reject.

    All series are aligned on a common index; rows with any NaN are dropped.

    Parameters
    ----------
    series    : list[pd.Series]
        Two or more I(1) price/level series.  Raw prices, NOT log-returns.
    lag_order : int
        Number of lags in the VAR model fitted in first differences (k_ar_diff).
    det_order : int
        Deterministic terms in the VECM:
        ``-1`` no constant, ``0`` constant inside cointegrating relation (default),
        ``1`` constant + linear trend.
    alpha     : float
        Significance level: must be one of 0.10, 0.05, or 0.01.

    Returns
    -------
    dict with keys:

    ``rank``              : int         – Number of cointegrating vectors (0 … k).
    ``is_cointegrated``   : bool        – True if rank ≥ 1.
    ``series_names``      : list[str]   – Names of the series in system order.
    ``trace_stats``       : list[float] – Trace statistics for H₀: rank ≤ i.
    ``critical_values``   : list[float] – Critical values at *alpha* for each H₀.
    ``eigenvectors``      : list[list]  – Cointegrating vectors as rows (k × k matrix).
    ``loadings``          : list[list]  – Adjustment coefficients (alpha matrix).
    ``spreads``           : dict[str, pd.Series]
                            One spread per cointegrating vector found (keyed
                            ``"cv_0"``, ``"cv_1"``, …).  Empty if rank = 0.
    ``alpha_used``        : float       – Significance level used.
    """
    # Name series
    named = []
    for i, s in enumerate(series):
        named.append(s.rename(s.name if s.name is not None else f"s{i}"))

    df = pd.concat(named, axis=1).dropna()
    k  = df.shape[1]

    _empty: dict = {
        "rank":            0,
        "is_cointegrated": False,
        "series_names":    [str(s.name) for s in named],
        "trace_stats":     [],
        "critical_values": [],
        "eigenvectors":    [],
        "loadings":        [],
        "spreads":         {},
        "coint_series":    None,
        "alpha_used":      alpha,
    }

    if k < 2:
        raise ValueError("johansen_cointegration requires at least 2 series.")

    min_obs = max(15, k * (lag_order + 2) + k)
    if len(df) < min_obs:
        return _empty

    # statsmodels critical-value column: 0 → 10 %, 1 → 5 %, 2 → 1 %
    cv_col = {0.10: 0, 0.05: 1, 0.01: 2}.get(alpha, 1)

    try:
        result = coint_johansen(df.values, det_order=det_order, k_ar_diff=lag_order)
    except Exception:
        return _empty

    trace_stats  = result.lr1.tolist()           # length k
    crit_vals    = result.cvt[:, cv_col].tolist() # length k

    # Determine rank: step through hypotheses H₀: rank ≤ 0, ≤ 1, … until fail to reject
    rank = 0
    for i, (ts, cv) in enumerate(zip(trace_stats, crit_vals)):
        if ts > cv:
            rank = i + 1
        else:
            break

    # Build spreads for each accepted cointegrating vector
    spreads: dict[str, pd.Series] = {}
    for r in range(rank):
        evec = result.evec[:, r]
        # Normalize: first non-zero element becomes 1
        pivot = next((v for v in evec if abs(v) > 1e-12), 1.0)
        evec  = evec / pivot
        spread = pd.Series(
            df.values @ evec,
            index=df.index,
            name=f"cv_{r}",
        )
        spreads[f"cv_{r}"] = spread

    # Calculate raw cointegrated series
    coint_series = None
    if rank > 0:
        coint_series = pd.DataFrame(
            df.values @ result.evec[:, :rank],
            index=df.index,
            columns=[f"coint_{i}" for i in range(rank)]
        )

    return {
        "rank":            rank,
        "is_cointegrated": bool(rank >= 1),
        "series_names":    df.columns.tolist(),
        "trace_stats":     trace_stats,
        "critical_values": crit_vals,
        "eigenvectors":    result.evec.T.tolist(),   # rows = eigenvectors
        "loadings":        result.evec.tolist(),     # alpha matrix (k × k)
        "spreads":         spreads,
        "coint_series":    coint_series,
        "alpha_used":      alpha,
    }


