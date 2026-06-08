import pytest
import pandas as pd
import numpy as np
from finalise.config import Config
from finalise.target import TargetAnalysis

def test_target_analysis_run_success(prices_synthetic):
    # Test TargetAnalysis.run() with a series that has signal
    # We can modify prices_synthetic slightly to make it persistent/trending
    trended_prices = prices_synthetic * np.exp(0.02 * np.arange(len(prices_synthetic)))
    config = Config(
        target_ticker="TEST",
        predictor_tickers=[],
        horizons=[1, 5],
        lag_max=5,
        knn_k=3,
        force_continue=True
    )
    prices = {"TEST": trended_prices}
    
    ta = TargetAnalysis(prices, config)
    ta.run()
    
    assert ta.horizon in [1, 5]
    assert ta.alpha in [0.05, 0.02]
    assert len(ta.results) > 0
    
    figs = ta.report()
    assert isinstance(figs, dict)
    assert "distribution" in figs
    assert "hurst" in figs

def test_target_analysis_no_signal_raises(prices_synthetic):
    # Test that it raises ValueError on prices_synthetic (white noise returns, no signal) when force_continue is False
    config = Config(
        target_ticker="TEST",
        predictor_tickers=[],
        horizons=[1],
        lag_max=5,
        knn_k=3,
        force_continue=False
    )
    prices = {"TEST": prices_synthetic}
    
    ta = TargetAnalysis(prices, config)
    with pytest.raises(ValueError, match="Nenhum sinal detectado"):
        ta.run()

def test_target_analysis_no_signal_warning(prices_synthetic):
    # Test that it issues a warning but continues on prices_synthetic when force_continue is True
    config = Config(
        target_ticker="TEST",
        predictor_tickers=[],
        horizons=[1],
        lag_max=5,
        knn_k=3,
        force_continue=True
    )
    prices = {"TEST": prices_synthetic}
    
    ta = TargetAnalysis(prices, config)
    ta.run() # Should not raise ValueError
    assert ta.horizon == 1
