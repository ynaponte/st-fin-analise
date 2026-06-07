from typing import Union
import pandas as pd
from ..config import Config
from . import returns
from . import descriptive
from . import stationarity
from . import hurst
from . import entropy
from . import autocorrelation
from . import selector
from . import report

class TargetAnalysis:
    def __init__(self, prices: Union[dict, pd.Series], config: Config):
        self.prices = prices
        self.config = config
        self.horizon = None
        self.alpha = config.alpha
        self.results = {}

    def run(self):
        # 1. Extract target series if prices is a dict
        if isinstance(self.prices, dict):
            price_series = self.prices[self.config.target_ticker]
        else:
            price_series = self.prices
            
        self.results = {}
        for k in self.config.horizons:
            # a. Compute returns
            ret = returns.compute(price_series, k, overlapping=False)
            if len(ret) < 10:
                continue
                
            # b. Descriptive stats
            desc_stats = descriptive.stats(ret)
            scale_alpha = desc_stats["alpha"]
            
            # c. Stationarity test (ADF)
            stationarity_res = stationarity.adf(ret, alpha=scale_alpha)
            
            scale_results = {
                "returns": ret,
                **desc_stats,
                **stationarity_res
            }
            
            # If not stationary, exclude from subsequent analyses
            if not stationarity_res["is_stationary"]:
                self.results[k] = scale_results
                continue
                
            # d. Hurst Exponent via DFA
            h_val, h_diff = hurst.dfa(ret)
            scale_results["hurst"] = h_val
            scale_results["hurst_diff"] = h_diff
            
            # e. Shannon Entropy
            shannon_val = entropy.shannon(ret)
            scale_results["shannon"] = shannon_val
            
            # f. Auto-MI
            auto_mi_profile, opt_lag = entropy.auto_mi(
                ret,
                lag_max=self.config.lag_max,
                k=self.config.knn_k
            )
            scale_results["auto_mi_profile"] = auto_mi_profile
            scale_results["auto_mi_max"] = max(auto_mi_profile) if auto_mi_profile else 0.0
            scale_results["auto_mi_significant"] = bool(scale_results["auto_mi_max"] > 0.05)
            
            # g. ACF/PACF
            acf_pacf_res = autocorrelation.acf_pacf(
                ret,
                lags=self.config.lag_max,
                alpha=scale_alpha
            )
            scale_results["acf_pacf"] = acf_pacf_res
            
            self.results[k] = scale_results
            
        # 2. Choose the best horizon
        try:
            self.horizon = selector.choose_horizon(self.results)
            self.alpha = self.results[self.horizon]["alpha"]
        except ValueError as e:
            self.horizon = None
            self.alpha = self.config.alpha
            raise e
            
        # 3. Gate check: if no signal is detected, raise error unless force_continue is True
        best_m = self.results[self.horizon]
        h_diff = best_m.get("hurst_diff", 0.0)
        auto_mi_sig = best_m.get("auto_mi_significant", False)
        
        if h_diff < 0.05 and not auto_mi_sig:
            if not self.config.force_continue:
                raise ValueError("Nenhum sinal detectado no ativo alvo.")

    def report(self) -> dict:
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")
        # If run failed to select a horizon, use first available stationary scale for report fallback
        best_h = self.horizon
        if best_h is None:
            stationary_scales = [k for k, v in self.results.items() if v.get("is_stationary", True)]
            if stationary_scales:
                best_h = stationary_scales[0]
            else:
                best_h = list(self.results.keys())[0]
                
        return report.generate(self.results, best_h)
