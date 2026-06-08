import numpy as np
import pandas as pd
import pytest
from finalise.predictors import cointegration

def test_ut_coint_success():
    # Cointegration on raw prices: y(t) = 1.5 * x(t) + stationary_error
    rng = np.random.default_rng(42)
    n = 100
    x_prices = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)))
    spread = pd.Series(5.0 + rng.standard_normal(n))
    y_prices = 1.5 * x_prices + spread
    
    res = cointegration.engle_granger(x_prices, y_prices, alpha=0.05)
    assert isinstance(res, dict)
    assert "statistic" in res
    assert "p_value" in res
    assert "is_cointegrated" in res
    assert "spread" in res
    assert "beta" in res
    assert "alpha_const" in res
    
    assert res["is_cointegrated"] is True
    assert res["p_value"] < 0.05
    assert np.isclose(res["beta"], 1.5, atol=0.2)

def test_ut_coint_non_cointegrated():
    # Cointegration on independent random walks
    rng = np.random.default_rng(42)
    n = 100
    x_prices = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)))
    y_prices = pd.Series(100.0 + np.cumsum(rng.standard_normal(n)))
    
    res = cointegration.engle_granger(x_prices, y_prices, alpha=0.05)
    assert isinstance(res, dict)
    # They should not be cointegrated
    assert "is_cointegrated" in res

def test_ut_coint_validation_error():
    # Negative log-returns passed in should raise ValueError
    series_neg = pd.Series([1.0, -0.05, 0.02, 1.1])
    series_pos = pd.Series([1.0, 1.05, 1.02, 1.1])
    
    with pytest.raises(ValueError, match="preços brutos"):
        cointegration.engle_granger(series_neg, series_pos)
        
    with pytest.raises(ValueError, match="preços brutos"):
        cointegration.engle_granger(series_pos, series_neg)

def test_ut_coint_short_series():
    # Short series handling
    x_short = pd.Series([1.0, 2.0, 3.0])
    y_short = pd.Series([2.0, 4.0, 6.0])
    res = cointegration.engle_granger(x_short, y_short)
    assert res["is_cointegrated"] is False
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0
