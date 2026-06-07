from typing import Optional, Union, List
import pandas as pd
from . import decomposition, smoothing, selector, cointegration, report
from .selector import SelectedCandidate

from finalise.target.returns import compute


class PredictorAnalysis:
    def __init__(self, prices_dict: dict[str, pd.Series], target_series: pd.Series, config, alpha: Optional[float] = None):
        self.prices_dict = prices_dict
        self.target_series = target_series
        self.config = config
        self.alpha = alpha if alpha is not None else config.alpha
        
        self.selected: List[SelectedCandidate] = []
        self.cointegration: dict[str, dict] = {}
        self.results: dict = {}
        self.candidates: dict = {}
        
        # Identify the horizon k from target_series
        target_prices = self.prices_dict.get(self.config.target_ticker)
        if target_prices is None or target_prices.empty:
            raise ValueError(f"Preços do ativo alvo '{self.config.target_ticker}' não encontrados em prices_dict.")
            
        self.horizon = None
        for k in self.config.horizons:
            expected_ret = compute(target_prices, k, overlapping=False)
            if expected_ret.index.equals(self.target_series.index):
                self.horizon = k
                break
        if self.horizon is None:
            # Fallback/default to first horizon or 1 if not found
            self.horizon = self.config.horizons[0] if self.config.horizons else 1

    def run(self):
        # 1. Candidate Generation
        candidates = {}
        for ticker in self.config.predictor_tickers:
            if ticker not in self.prices_dict or self.prices_dict[ticker].empty:
                continue
            price_series = self.prices_dict[ticker]
            # Compute log returns at self.horizon
            c_bruto = compute(price_series, k=self.horizon, overlapping=False)
            if len(c_bruto) < 10:
                continue
            
            # 5 candidate series:
            candidates[(ticker, "bruto")] = c_bruto
            candidates[(ticker, "suavizado")] = smoothing.apply(
                c_bruto, 
                window=self.config.smoothing_window, 
                method=self.config.smoothing_method
            )
            
            decomp = decomposition.stl(c_bruto, period=self.config.stl_period)
            candidates[(ticker, "tendencia")] = decomp["trend"]
            candidates[(ticker, "sazonalidade")] = decomp["seasonal"]
            candidates[(ticker, "residuo")] = decomp["residual"]
            
        # 2. Cointegration on raw prices
        self.cointegration = {}
        target_prices = self.prices_dict[self.config.target_ticker]
        for ticker in self.config.predictor_tickers:
            if ticker not in self.prices_dict or self.prices_dict[ticker].empty:
                continue
            pred_prices = self.prices_dict[ticker]
            self.cointegration[ticker] = cointegration.engle_granger(
                pred_prices, target_prices, alpha=self.alpha
            )
            
        # 3. Decision pipeline selection and populating results
        self.candidates = candidates
        self.results = {}
        self.selected = selector.select(
            candidates, self.target_series,
            config=self.config, alpha=self.alpha,
            results=self.results
        )

    def report(self) -> dict:
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")
        return report.generate(
            self.results, self.selected, self.cointegration,
            config=self.config, alpha=self.alpha
        )

__all__ = ["PredictorAnalysis", "SelectedCandidate"]
