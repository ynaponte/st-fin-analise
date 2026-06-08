import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from finalise.target import returns

def test_ut_ret_01():
    # UT-RET-01: r_1(t) = log(P(t)/P(t-1)) produces correct value
    prices = pd.Series([100.0, 105.0, 110.25])
    r1 = returns.compute(prices, k=1)
    expected = pd.Series([np.log(1.05), np.log(1.05)], index=[1, 2])
    pd.testing.assert_series_equal(r1, expected)

def test_ut_ret_02(prices_synthetic):
    # UT-RET-02: length checks for non-overlapping
    r5 = returns.compute(prices_synthetic, k=5, overlapping=False)
    r21 = returns.compute(prices_synthetic, k=21, overlapping=False)
    assert len(r5) == 1199 // 5
    assert len(r21) == 1199 // 21

def test_ut_ret_03(prices_synthetic):
    # UT-RET-03: indices of r_5 are 5 business days apart
    r5 = returns.compute(prices_synthetic, k=5, overlapping=False)
    prices_index = list(prices_synthetic.index)
    positions = [prices_index.index(date) for date in r5.index]
    pos_diffs = np.diff(positions)
    assert np.all(pos_diffs == 5)

def test_ut_ret_04():
    # UT-RET-04: NaN in input is discarded
    prices = pd.Series([100.0, np.nan, 105.0, 110.25])
    r1 = returns.compute(prices, k=1)
    assert len(r1) == 1
    assert np.isclose(r1.iloc[0], np.log(1.05))

def test_ut_ret_05(prices_synthetic):
    # UT-RET-05: sum of log returns equals log(P(t)/P(t-k))
    r5 = returns.compute(prices_synthetic, k=5, overlapping=False)
    for idx, val in r5.items():
        pos = prices_synthetic.index.get_loc(idx)
        p_t = prices_synthetic.iloc[pos]
        p_t5 = prices_synthetic.iloc[pos - 5]
        assert np.isclose(val, np.log(p_t / p_t5))

@given(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=100))
def test_ut_ret_p01(price_list):
    # PB-01 / UT-RET-P01: Log returns of positive prices are finite and real
    prices = pd.Series(price_list)
    r1 = returns.compute(prices, k=1)
    assert np.all(np.isfinite(r1))
    assert np.all(np.isreal(r1))
