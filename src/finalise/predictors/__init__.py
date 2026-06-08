from typing import Optional, Union, List
import pandas as pd
from . import decomposition, smoothing, selector, cointegration, report
from .selector import SelectedCandidate

from finalise.target.returns import compute


class PredictorAnalysis:
    def __init__(self, prices_dict: dict[str, pd.Series], target_series: pd.Series, horizon: int, config, alpha: Optional[float] = None):
        self.prices_dict = prices_dict
        self.target_series = target_series
        self.horizon = horizon
        self.config = config
        self.alpha = alpha if alpha is not None else config.alpha
        
        self.selected: List[SelectedCandidate] = []
        self.cointegration: dict[str, dict] = {}
        self.results: dict = {}
        self.candidates: dict = {}

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
            
            # Decompose RAW prices (A-04), use raw components for Information Theory tests
            decomp = decomposition.stl(price_series, period=self.config.stl_period)
            
            # Align indices with c_bruto
            common_idx = c_bruto.index.intersection(decomp["trend"].index)
            if len(common_idx) > 10:
                candidates[(ticker, "tendencia")] = decomp["trend"].loc[common_idx]
                candidates[(ticker, "sazonalidade")] = decomp["seasonal"].loc[common_idx]
                candidates[(ticker, "residuo")] = decomp["residual"].loc[common_idx]
            
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
