import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from finalise.target import hurst

def test_ut_hurst_01(white_noise, persistent_series, antipersistent_series):
    # UT-HURST-01: DFA returns H in (0, 1) for any valid series
    for s in [white_noise, persistent_series, antipersistent_series]:
        H, diff = hurst.dfa(s)
        assert 0.0 < H < 1.0
        assert np.isclose(diff, abs(H - 0.5))

def test_ut_hurst_02(white_noise):
    # UT-HURST-02: White noise (stationary, I(0)) returns H ≈ 0.5
    H, _ = hurst.dfa(white_noise)
    assert 0.45 <= H <= 0.55

def test_ut_hurst_03(persistent_series):
    # UT-HURST-03: Fractional Gaussian noise with H=0.75 (persistent, I(0)) returns H > 0.55
    H, _ = hurst.dfa(persistent_series)
    assert H > 0.55

def test_ut_hurst_04(antipersistent_series):
    # UT-HURST-04: Antipersistent series returns H < 0.45
    H, _ = hurst.dfa(antipersistent_series)
    assert H < 0.45

def test_ut_hurst_05_random_walk():
    # UT-HURST-05: Random walk (I(1), non-stationary) yields DFA α ≈ 1.5,
    # which maps to H = α − 1 ≈ 0.5, confirming the corrected formula.
    rng = np.random.default_rng(42)
    rw = pd.Series(np.cumsum(rng.standard_normal(1200)))
    H, diff = hurst.dfa(rw)
    assert 0.35 <= H <= 0.65, f"Random walk should map to H ≈ 0.5, got H={H:.4f}"
    assert diff < 0.20

@given(st.lists(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False), min_size=50, max_size=200))
def test_ut_hurst_p01(lst):
    # PB-03: H in (0, 1) for any non-constant float series
    series = pd.Series(lst)
    if series.nunique() <= 1:
        return  # skip constant lists
    H, diff = hurst.dfa(series)
    assert 0.0 < H < 1.0

