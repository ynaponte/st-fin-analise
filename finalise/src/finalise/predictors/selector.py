from dataclasses import dataclass
import pandas as pd
from typing import List, Optional
from finalise.target import stationarity


@dataclass
class SelectedCandidate:
    ticker: str
    component: str
    series: pd.Series
    lag_tau: int
    relation_type: str
    mi_value: float
    te_value: Optional[float]
    granger_pvalue: float

def select(candidates: dict, target: pd.Series, config, alpha: float, results: Optional[dict] = None) -> List[SelectedCandidate]:
    """
    Applies the decision pipeline to select predictor candidates:
    1. Cross-MI signficance (with Bonferroni threshold = alpha / lag_max). Discards if non-significant.
    2. Granger causality at lag tau*. If rejected (p < alpha), includes as 'linear'.
    3. Else, runs Transfer Entropy at lag tau*. If significant, includes as 'non-linear'.
    4. Else, discards.
    """
    from . import mi, granger, transfer_entropy
    from finalise.target import entropy
    
    # Compute target auto-MI optimal lag for Transfer Entropy embedding dimension
    _, target_opt_lag = entropy.auto_mi(target, lag_max=config.lag_max, k=config.knn_k)
    
    selected_list = []
    
    for (ticker, component), c_series in candidates.items():
        # 1. Cross-MI
        mi_res = mi.cross_mi_lags(
            c_series, target,
            lag_max=config.lag_max,
            alpha=alpha,
            k=config.knn_k
        )
        
        if results is not None:
            results[(ticker, component)] = {
                "mi": mi_res
            }
            
        if not mi_res["is_significant"]:
            continue
            
        tau = mi_res["lag_opt"]
        mi_val = mi_res["mi_profile"][tau - 1]
        
        # 2. Granger causality requires stationarity and is not applied to STL components (user directive)
        from finalise.target import stationarity
        adf_res = stationarity.adf(c_series, alpha=alpha)
        
        is_stl_component = component in ["tendencia", "sazonalidade", "residuo"]
        
        if adf_res["is_stationary"] and not is_stl_component:
            granger_res = granger.test(
                c_series, target,
                lag=tau,
                alpha=alpha
            )
        else:
            # Bypass Granger if predictor candidate is non-stationary or is an STL component
            granger_res = {
                "p_value": 1.0,
                "is_causal": False,
                "statistic": 0.0
            }
        
        if results is not None:
            results[(ticker, component)]["granger"] = granger_res
            
        if granger_res["is_causal"]:
            selected_list.append(SelectedCandidate(
                ticker=ticker,
                component=component,
                series=c_series,
                lag_tau=tau,
                relation_type="linear",
                mi_value=mi_val,
                te_value=None,
                granger_pvalue=granger_res["p_value"]
            ))
        else:
            # 3. Transfer Entropy
            te_res = transfer_entropy.compute(
                c_series, target,
                lag=tau,
                alpha=alpha,
                n_permutations=config.n_permutations,
                y_lags=target_opt_lag
            )
            
            if results is not None:
                results[(ticker, component)]["te"] = te_res
                
            if te_res["is_significant"]:
                selected_list.append(SelectedCandidate(
                    ticker=ticker,
                    component=component,
                    series=c_series,
                    lag_tau=tau,
                    relation_type="non-linear",
                    mi_value=mi_val,
                    te_value=te_res["te"],
                    granger_pvalue=granger_res["p_value"]
                ))
                
    return selected_list
