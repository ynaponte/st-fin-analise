import pytest
import pandas as pd
import numpy as np
from finalise.config import Config
from finalise.analyze import analyze, AnalysisResult
from finalise.predictors.selector import SelectedCandidate

@pytest.mark.e2e
def test_e2e_analyze_full(prices_real):
    # E2E-01 a E2E-06, E2E-09
    config = Config(
        target_ticker="PETR4.SA",
        predictor_tickers=["GC=F", "DX-Y.NYB", "XOM"],
        horizons=[1, 5, 21],
        smoothing_window=30,
        smoothing_method="DEMA",
        stl_period=21,
        alpha=0.05,
        n_permutations=20, # Low for testing speed
        knn_k=3,
        lag_max=5,
        force_continue=True
    )
    
    # prices_real is a DataFrame of prices
    prices_dict = {}
    for col in prices_real.columns:
        prices_dict[col] = prices_real[col].dropna()
        
    result = analyze(prices_dict, config)
    
    assert isinstance(result, AnalysisResult)
    assert result.target.horizon in [1, 5, 21]
    assert result.target.alpha in [0.05, 0.02]
    
    assert isinstance(result.predictors.selected, list)
    for cand in result.predictors.selected:
        assert isinstance(cand, SelectedCandidate)
        assert cand.ticker in config.predictor_tickers
        
    if result.model is not None:
        assert isinstance(result.model.is_outlier, bool)
        
        # Test reports
        figs_m = result.model.report()
        assert isinstance(figs_m, dict)
        
    figs_t = result.target.report()
    assert isinstance(figs_t, dict)
    
    figs_p = result.predictors.report()
    assert isinstance(figs_p, dict)

def test_e2e_analyze_white_noise(white_noise):
    # E2E-07 and E2E-08
    config_strict = Config(
        target_ticker="WN",
        predictor_tickers=[],
        horizons=[1],
        lag_max=5,
        knn_k=3,
        force_continue=False
    )
    # Convert white noise returns to price series
    prices = 100 * np.exp(np.cumsum(white_noise / 100.0))
    prices_dict = {"WN": prices}
    
    # Should raise error because white noise has no signal
    with pytest.raises(ValueError, match="Nenhum sinal detectado"):
        analyze(prices_dict, config_strict)
        
    config_force = Config(
        target_ticker="WN",
        predictor_tickers=[],
        horizons=[1],
        lag_max=5,
        knn_k=3,
        force_continue=True
    )
    
    # Should not raise error, but model will be None since no predictors
    result = analyze(prices_dict, config_force)
    assert result.target.horizon == 1
    assert result.model is None
