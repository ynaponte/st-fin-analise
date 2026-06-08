import numpy as np
import pandas as pd
import pytest
from finalise.predictors import mi

def test_ut_mi_cross():
    # Test cross_mi_lags on related series
    rng = np.random.default_rng(42)
    x = pd.Series(rng.standard_normal(200))
    # y is x lagged by 2 plus some noise
    y = x.shift(2) + 0.1 * rng.standard_normal(200)
    y = y.dropna()
    x = x.loc[y.index]
    
    res = mi.cross_mi_lags(x, y, lag_max=5, alpha=0.05, k=3, n_permutations=200)
    assert isinstance(res, dict)
    assert "mi_profile" in res
    assert "lag_opt" in res
    assert "is_significant" in res
    assert "threshold_alpha" in res
    
    assert len(res["mi_profile"]) == 5
    # The optimal lag should be 2
    assert res["lag_opt"] == 2
    assert res["is_significant"] is True
    assert res["threshold_alpha"] == 0.05

def test_ut_mi_unrelated():
    # Test cross_mi_lags on unrelated series (white noise)
    rng = np.random.default_rng(42)
    x = pd.Series(rng.standard_normal(200))
    y = pd.Series(rng.standard_normal(200))
    
    res = mi.cross_mi_lags(x, y, lag_max=3, alpha=0.01, k=3)
    # Since they are unrelated, MI should be small or non-significant
    assert res["threshold_alpha"] == 0.01
    # Check handling of short series
    res_short = mi.cross_mi_lags(x.iloc[:5], y.iloc[:5], lag_max=3, alpha=0.05, k=3)
    assert res_short["is_significant"] is False
    assert all(val == 0.0 for val in res_short["mi_profile"])
