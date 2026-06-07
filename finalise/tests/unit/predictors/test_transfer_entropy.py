import numpy as np
import pandas as pd
import pytest
from finalise.predictors import transfer_entropy

def test_ut_te_helpers():
    # Test bin_series
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    binned = transfer_entropy.bin_series(s, n_bins=3)
    assert len(binned) == len(s)
    assert np.all((binned >= 0) & (binned < 3))
    
    # Constant series
    s_const = pd.Series([2.0, 2.0, 2.0])
    binned_const = transfer_entropy.bin_series(s_const, n_bins=3)
    assert np.all(binned_const == 0)

    # Test entropy_joint
    arr1 = np.array([0, 1, 0, 1])
    arr2 = np.array([0, 1, 0, 1])
    # Since they are perfectly correlated, joint entropy equals individual entropy
    h_joint = transfer_entropy.entropy_joint([arr1, arr2])
    h_single = transfer_entropy.entropy_joint([arr1])
    assert np.isclose(h_joint, h_single)

def test_ut_te_compute():
    # Test compute_te and compute functions with a nonlinear lag relationship
    rng = np.random.default_rng(42)
    n = 250
    x = pd.Series(rng.standard_normal(n))
    y = pd.Series(np.zeros(n))
    
    # Nonlinear relationship: y(t) = y(t-1) + x(t-1)^2 + noise
    for t in range(1, n):
        y.iloc[t] = 0.3 * y.iloc[t-1] + 0.8 * (x.iloc[t-1] ** 2) + 0.1 * rng.standard_normal()
        
    te_val = transfer_entropy.compute_te(x, y, lag=1)
    assert te_val >= 0.0
    
    # Permutation test significance
    res = transfer_entropy.compute(x, y, lag=1, alpha=0.05, n_permutations=50)
    assert isinstance(res, dict)
    assert "te" in res
    assert "p_value" in res
    assert "is_significant" in res
    assert "null_distribution" in res
    assert "threshold" in res
    
    assert len(res["null_distribution"]) == 50
    assert res["te"] == te_val
    
    # Test short series fallback
    res_short = transfer_entropy.compute(x.iloc[:5], y.iloc[:5], lag=1, alpha=0.05, n_permutations=10)
    assert res_short["te"] == 0.0
    assert res_short["p_value"] == 1.0
    assert res_short["is_significant"] is False
