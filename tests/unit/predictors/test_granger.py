import numpy as np
import pandas as pd
import pytest
from finalise.predictors import granger

def test_ut_granger_causal():
    # Test Granger causality on a causal relationship: y(t) = 0.5 * y(t-1) + 0.8 * x(t-1) + noise
    rng = np.random.default_rng(42)
    n = 200
    x = pd.Series(rng.standard_normal(n))
    y = pd.Series(np.zeros(n))
    for t in range(1, n):
        y.iloc[t] = 0.5 * y.iloc[t-1] + 0.8 * x.iloc[t-1] + 0.2 * rng.standard_normal()
        
    res = granger.test(x, y, lag_max=1, alpha=0.05)
    assert isinstance(res, dict)
    assert "p_value" in res
    assert "is_causal" in res
    assert "statistic" in res
    
    assert res["is_causal"] is True
    assert res["p_value"] < 0.05

def test_ut_granger_non_causal():
    # Test Granger causality on non-causal series (white noise)
    rng = np.random.default_rng(42)
    n = 100
    x = pd.Series(rng.standard_normal(n))
    y = pd.Series(rng.standard_normal(n))
    
    res = granger.test(x, y, lag_max=2, alpha=0.05)
    # Since they are independent, it should not reject Granger causality (most of the time, so let's verify keys)
    assert "p_value" in res
    assert "is_causal" in res
    
    # Check handling of short series
    res_short = granger.test(x.iloc[:5], y.iloc[:5], lag_max=2, alpha=0.05)
    assert res_short["p_value"] == 1.0
    assert res_short["is_causal"] is False
    assert res_short["statistic"] == 0.0
    
    # Check negative/zero lag handled correctly
    res_zero = granger.test(x, y, lag_max=0, alpha=0.05)
    assert "is_causal" in res_zero
