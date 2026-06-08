import numpy as np
import pandas as pd
import pytest
from scipy.stats import t
from hypothesis import given, strategies as st
from finalise.target import descriptive

def test_ut_desc_01():
    # UT-DESC-01: known normal series stats checks
    # Mean = 0, Std = 1
    # We use a large enough sample size to have stable stats
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 1.0, 5000)
    series = pd.Series(data)
    res = descriptive.stats(series)
    
    assert np.isclose(res["mean"], 0.0, atol=0.05)
    assert np.isclose(res["std"], 1.0, atol=0.05)
    # normal skewness should be near 0
    assert np.isclose(res["skewness"], 0.0, atol=0.1)
    # Pearson kurtosis for normal is 3
    assert np.isclose(res["kurtosis"], 3.0, atol=0.2)

def test_ut_desc_02():
    # UT-DESC-02: Anderson-Darling returns is_normal=True for normal series
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0.0, 1.0, 1000))
    res = descriptive.stats(series)
    assert res["is_normal"] is True
    assert res["alpha"] == 0.05

def test_ut_desc_03():
    # UT-DESC-03: Anderson-Darling returns is_normal=False for t-Student df=3
    # Student's t-distribution with 3 df is heavy-tailed (non-normal)
    rng = np.random.default_rng(42)
    # Use scipy to generate Student's t
    data = t.rvs(df=3, size=1000, random_state=rng)
    series = pd.Series(data)
    res = descriptive.stats(series)
    assert res["is_normal"] is False

def test_ut_desc_04():
    # UT-DESC-04: is_normal=False adjusts alpha to 0.02
    rng = np.random.default_rng(42)
    # heavy-tailed distribution (cauchy or student's t with df=2)
    data = t.rvs(df=2, size=500, random_state=rng)
    series = pd.Series(data)
    res = descriptive.stats(series)
    assert res["is_normal"] is False
    assert res["alpha"] == 0.02

@given(st.integers(min_value=10000, max_value=20000))
def test_ut_desc_p01(n):
    # UT-DESC-P01: Kurtosis converges to 3 with large N
    rng = np.random.default_rng(100)
    data = rng.normal(0.0, 1.0, n)
    series = pd.Series(data)
    res = descriptive.stats(series)
    assert np.isclose(res["kurtosis"], 3.0, atol=0.15)
