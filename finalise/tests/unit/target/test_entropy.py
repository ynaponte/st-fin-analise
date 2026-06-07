import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from finalise.target import entropy

def test_ut_ent_01():
    # UT-ENT-01: Shannon of uniform is larger than of concentrated
    rng = np.random.default_rng(42)
    uniform_s = pd.Series(rng.uniform(0, 10, 1000))
    concentrated_s = pd.Series([1.0] * 1000)
    
    h_unif = entropy.shannon(uniform_s)
    h_conc = entropy.shannon(concentrated_s)
    
    assert h_unif > h_conc
    assert np.isclose(h_conc, 0.0)

def test_ut_ent_02():
    # UT-ENT-02: Bins estimated by Freedman-Diaconis (bin count is lower / bin width is larger for larger IQR)
    # Both series have range 20 and length 100
    s1 = pd.Series([-10.0] + list(np.linspace(-0.05, 0.05, 98)) + [10.0]) # Concentrated: small IQR
    s2 = pd.Series([-10.0] + list(np.linspace(-5.0, 5.0, 98)) + [10.0]) # Spread: larger IQR
    
    b1 = entropy.fd_bins(s1)
    b2 = entropy.fd_bins(s2)
    
    # Larger IQR => Larger bin width => Fewer bins
    assert b2 < b1

def test_ut_ent_03(white_noise):
    # UT-ENT-03: auto-MI of white noise is near zero for all lags
    mi_profile, opt_lag = entropy.auto_mi(white_noise, lag_max=5, k=5)
    assert len(mi_profile) == 5
    assert all(val < 0.1 for val in mi_profile)

def test_ut_ent_04():
    # UT-ENT-04: auto-MI of AR(1) has peak at lag=1
    rng = np.random.default_rng(42)
    ar1 = np.zeros(1000)
    for t in range(1, 1000):
        ar1[t] = 0.8 * ar1[t-1] + rng.standard_normal()
    series = pd.Series(ar1)
    
    mi_profile, opt_lag = entropy.auto_mi(series, lag_max=5, k=5)
    assert opt_lag == 1
    assert mi_profile[0] == max(mi_profile)
    assert mi_profile[0] > 0.3

def test_ut_ent_05(white_noise):
    # UT-ENT-05: auto-MI profile length equals lag_max
    mi_profile, _ = entropy.auto_mi(white_noise, lag_max=12, k=5)
    assert len(mi_profile) == 12

@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=10, max_size=100))
def test_ut_ent_p01(lst):
    # PB-02: Shannon entropy is always >= 0
    series = pd.Series(lst)
    h_val = entropy.shannon(series)
    assert h_val >= 0.0
