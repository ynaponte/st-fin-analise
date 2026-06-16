import numpy as np
import pandas as pd
import pytest
from finalise.analysis.base import johansen_cointegration
from finalise.workflows.te_surrogates import generate_circular_surrogates, test_te_significance
from finalise.workflows.predictors_analysis import PredictorsAnalysis
from finalise.config import Config

def test_run_te_permutation_test_coupled():
    # Test permutation test when driver and target are coupled
    rng = np.random.default_rng(42)
    n = 200
    x = pd.Series(rng.standard_normal(n), name="X")
    y = pd.Series(np.zeros(n), name="Y")
    
    # Coupled relationship: y(t) = 0.5 * y(t-1) + 0.8 * x(t-1)^2 + noise
    for t in range(1, n):
        y.iloc[t] = 0.5 * y.iloc[t-1] + 0.8 * (x.iloc[t-1] ** 2) + 0.1 * rng.standard_normal()
        
    surrs = generate_circular_surrogates(x, n_surrogates=20, seed=42)
    te_obs, p_val = test_te_significance(x, y, surrs, lag=1, y_lags=1)
    
    assert te_obs > 0.0
    assert p_val <= 0.05  # Should be statistically significant
    
def test_run_te_permutation_test_independent():
    # Test permutation test when driver and target are independent
    rng = np.random.default_rng(42)
    n = 100
    x = pd.Series(rng.standard_normal(n), name="X")
    y = pd.Series(rng.standard_normal(n), name="Y")
    
    surrs = generate_circular_surrogates(x, n_surrogates=20, seed=42)
    te_obs, p_val = test_te_significance(x, y, surrs, lag=1, y_lags=1)
    
    # Should not be statistically significant
    assert p_val > 0.05

def test_johansen_cointegration_returns_coint_series():
    # Cointegration system
    rng = np.random.default_rng(42)
    n = 150
    x = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)), name="X")
    y = pd.Series(1.5 * x + rng.standard_normal(n), name="Y")
    
    res = johansen_cointegration([y, x], lag_order=1, alpha=0.05)
    
    assert res["is_cointegrated"] is True
    assert res["rank"] >= 1
    assert res["coint_series"] is not None
    assert isinstance(res["coint_series"], pd.DataFrame)
    assert res["coint_series"].shape[0] == n
    assert res["coint_series"].shape[1] == res["rank"]

def test_predictors_analysis_with_refinements():
    # Integration test for PredictorsAnalysis workflow
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n)
    target = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)), index=dates, name="TARGET")
    pred1 = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)), index=dates, name="PRED1")
    pred2 = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)), index=dates, name="PRED2")
    
    prices_dict = {
        "TARGET": target,
        "PRED1": pred1,
        "PRED2": pred2
    }
    
    config = Config(
        target_ticker="TARGET",
        predictor_tickers=["PRED1", "PRED2"],
        smoothing_method="SMA",
        lag_max=3,
        stl_period=5,
        alpha=0.05,
        knn_k=3
    )
    
    pa = PredictorsAnalysis(prices_dict, config, smoothing_window=5, te_permutations=10)
    result = pa.run()
    
    assert result.lag_consensus >= 1
    assert result.trend_transfer_entropy is not None
    assert result.residual_transfer_entropy is not None
    assert isinstance(result.trend_transfer_entropy, dict)
    assert isinstance(result.residual_transfer_entropy, dict)
