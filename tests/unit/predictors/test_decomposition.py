import numpy as np
import pandas as pd
import pytest
from finalise.predictors import decomposition

def test_ut_dec_01():
    # STL output format and summation property: trend + seasonal + residual = series
    series = pd.Series(np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100))
    res = decomposition.stl(series, period=10)
    
    assert isinstance(res, dict)
    assert set(res.keys()) == {"trend", "seasonal", "residual"}
    
    for key in res:
        assert isinstance(res[key], pd.Series)
        assert len(res[key]) == len(series)
        assert res[key].index.equals(series.index)
        
    summed = res["trend"] + res["seasonal"] + res["residual"]
    pd.testing.assert_series_equal(summed, series, check_names=False)

def test_ut_dec_02():
    # Starting values where STL cannot run should have trend = series, seasonal = 0, residual = 0
    series = pd.Series(np.random.normal(0, 1, 50))
    period = 12
    res = decomposition.stl(series, period=period)
    
    min_obs = max(10, 2 * period) # 24
    
    # Check that before min_obs, trend equals series, seasonal and residual are 0
    for i in range(min_obs - 1):
        assert np.isclose(res["trend"].iloc[i], series.iloc[i])
        assert np.isclose(res["seasonal"].iloc[i], 0.0)
        assert np.isclose(res["residual"].iloc[i], 0.0)

def test_ut_dec_03():
    # STL decomposition is causal. Changing future values should not change past results.
    series = pd.Series(np.random.normal(0, 1, 60))
    res1 = decomposition.stl(series, period=10)
    
    # Modify values at the end of the series
    series_modified = series.copy()
    series_modified.iloc[-5:] += 10.0
    res2 = decomposition.stl(series_modified, period=10)
    
    # Values before the modified window must be identical
    pd.testing.assert_series_equal(res1["trend"].iloc[:-5], res2["trend"].iloc[:-5])
    pd.testing.assert_series_equal(res1["seasonal"].iloc[:-5], res2["seasonal"].iloc[:-5])
    pd.testing.assert_series_equal(res1["residual"].iloc[:-5], res2["residual"].iloc[:-5])
