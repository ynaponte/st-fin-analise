from dataclasses import dataclass
import pandas as pd
from typing import List, Optional
from finalise.target import stationarity
from rich.progress import track


@dataclass
class SelectedCandidate:
    ticker: str
    component: str
    series: pd.Series          # series used for selection (daily returns or STL component)
    lag_tau: int               # best predictive lag found
    relation_type: str         # "linear" | "non-linear"
    mi_value: float
    te_value: Optional[float]
    granger_pvalue: float


def select(
    candidates: dict,
    target: pd.Series,           # daily log-returns of the target at horizon k
    target_daily: pd.Series,     # daily log-returns of the target (always k=1)
    config,
    alpha: float,
    results: Optional[dict] = None,
    cointegration: Optional[dict] = None,
) -> List["SelectedCandidate"]:
    """
    Decision pipeline for predictor selection.  Order of tests matters:

    For stationary, non-STL candidates (daily returns):
      1. Granger causality (lags 1..lag_max, BH-corrected).
         → Significant: accept as 'linear'.
      2. If not Granger: Cross-MI (max-statistic permutation test).
         → Significant: run Transfer Entropy at best MI lag.
            → TE significant: accept as 'non-linear'.
      3. Else: discard.

    For STL components (trend/seasonal/residual — may be non-stationary):
      → Skip Granger (requires stationarity).
      → Go straight to Cross-MI → TE path.

    Rationale:
      - Granger is the most direct test of *linear* predictability and should
        be tried first on stationary series.
      - Cross-MI alone is not a sufficient condition for predictability and
        should not gate Granger.  The original code had this backwards.
      - Cross-MI + TE serve as the *non-linear* detection path for cases
        where Granger finds nothing.
      - All Granger tests run on the daily return series (k=1), which has
        ~2000 observations and sufficient power.  The k-period return series
        used for TE/MI preserves the horizon semantics.
    """
    from . import mi, granger, transfer_entropy
    from finalise.target import entropy

    # Compute target auto-MI optimal lag (used as embedding dim in TE)
    _, target_opt_lag = entropy.auto_mi(
        target_daily, lag_max=min(config.lag_max, 10), k=config.knn_k
    )

    selected_list = []

    for (ticker, component), c_series in track(
        candidates.items(), description="Avaliando Preditores..."
    ):
        is_stl = component in ["tendencia", "sazonalidade", "residuo"]
        result_entry: dict = {}

        # ── PATH A: Granger (stationary non-STL series) ──────────────────
        if not is_stl:
            adf_res = stationarity.adf(c_series, alpha=alpha)
            result_entry["adf"] = adf_res

            if adf_res["is_stationary"]:
                # Granger on DAILY returns — use target_daily for power
                # Use c_series aligned to target_daily index
                granger_res = granger.test(
                    c_series,
                    target_daily,
                    lag_max=config.lag_max,
                    alpha=alpha,
                )
                result_entry["granger"] = granger_res

                if results is not None:
                    results[(ticker, component)] = result_entry

                if granger_res["is_causal"]:
                    tau = granger_res["best_lag"]
                    selected_list.append(
                        SelectedCandidate(
                            ticker=ticker,
                            component=component,
                            series=c_series,
                            lag_tau=tau,
                            relation_type="linear",
                            mi_value=0.0,  # not computed in this path
                            te_value=None,
                            granger_pvalue=granger_res["p_value"],
                        )
                    )
                    continue  # accepted; skip MI/TE

        # ── PATH B: Cross-MI → Transfer Entropy (non-linear path) ────────
        # Also used for STL components and non-stationary series
        mi_res = mi.cross_mi_lags(
            c_series,
            target,
            lag_max=config.lag_max,
            alpha=alpha,
            k=config.knn_k,
            n_permutations=config.n_permutations,
        )
        result_entry["mi"] = mi_res

        if results is not None:
            results[(ticker, component)] = result_entry

        if not mi_res["is_significant"]:
            continue

        tau = mi_res["lag_opt"]
        mi_val = mi_res["mi_profile"][tau - 1] if mi_res["mi_profile"] else 0.0

        te_res = transfer_entropy.compute(
            c_series,
            target,
            lag=tau,
            alpha=alpha,
            n_permutations=config.n_permutations,
            y_lags=target_opt_lag,
        )
        result_entry["te"] = te_res

        if results is not None:
            results[(ticker, component)] = result_entry

        if te_res["is_significant"]:
            selected_list.append(
                SelectedCandidate(
                    ticker=ticker,
                    component=component,
                    series=c_series,
                    lag_tau=tau,
                    relation_type="non-linear",
                    mi_value=mi_val,
                    te_value=te_res["te"],
                    granger_pvalue=1.0,
                )
            )

    return selected_list
