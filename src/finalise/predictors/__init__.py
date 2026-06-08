from typing import Optional, Union, List
import pandas as pd
from . import decomposition, smoothing, selector, cointegration, report
from .selector import SelectedCandidate

from finalise.target.returns import compute


class PredictorAnalysis:
    def __init__(
        self,
        prices_dict: dict[str, pd.Series],
        target_series: pd.Series,
        horizon: int,
        config,
        alpha: Optional[float] = None,
    ):
        self.prices_dict = prices_dict
        self.target_series = target_series   # k-period returns of target (used for TE/MI)
        self.horizon = horizon
        self.config = config
        self.alpha = alpha if alpha is not None else config.alpha

        self.selected: List[SelectedCandidate] = []
        self.cointegration: dict[str, dict] = {}
        self.results: dict = {}
        self.candidates: dict = {}

    def run(self):
        # ── Target daily returns (k=1) — used for Granger (needs stationarity + power) ──
        target_prices = self.prices_dict[self.config.target_ticker]
        target_daily = compute(target_prices, k=1, overlapping=False)

        # ── 1. Candidate Generation ───────────────────────────────────────────────────
        candidates = {}
        for ticker in self.config.predictor_tickers:
            if ticker not in self.prices_dict or self.prices_dict[ticker].empty:
                continue
            price_series = self.prices_dict[ticker]

            # Daily returns (k=1) — primary candidate for Granger path
            c_daily = compute(price_series, k=1, overlapping=False)
            if len(c_daily) < 10:
                continue
            candidates[(ticker, "bruto")] = c_daily

            # Smoothed daily returns
            candidates[(ticker, "suavizado")] = smoothing.apply(
                c_daily,
                window=self.config.smoothing_window,
                method=self.config.smoothing_method,
            )

            # STL decomposition on RAW prices → components for non-linear path
            decomp = decomposition.stl(price_series, period=self.config.stl_period)

            # Align components with the daily returns index
            common_idx = c_daily.index.intersection(decomp["trend"].index)
            if len(common_idx) > 10:
                candidates[(ticker, "tendencia")] = decomp["trend"].loc[common_idx]
                candidates[(ticker, "sazonalidade")] = decomp["seasonal"].loc[common_idx]
                candidates[(ticker, "residuo")] = decomp["residual"].loc[common_idx]

        # ── 2. Cointegration test on raw prices ───────────────────────────────────────
        self.cointegration = {}
        for ticker in self.config.predictor_tickers:
            if ticker not in self.prices_dict or self.prices_dict[ticker].empty:
                continue
            pred_prices = self.prices_dict[ticker]
            self.cointegration[ticker] = cointegration.engle_granger(
                pred_prices, target_prices, alpha=self.alpha
            )

        # ── 3. Decision pipeline ──────────────────────────────────────────────────────
        self.candidates = candidates
        self.results = {}
        self.selected = selector.select(
            candidates,
            target=self.target_series,       # k-period returns (for MI/TE horizon semantics)
            target_daily=target_daily,       # daily returns (for Granger power)
            config=self.config,
            alpha=self.alpha,
            results=self.results,
            cointegration=self.cointegration,
        )

    def report(self) -> dict:
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")
        return report.generate(
            self.results, self.selected, self.cointegration,
            config=self.config, alpha=self.alpha
        )


__all__ = ["PredictorAnalysis", "SelectedCandidate"]
