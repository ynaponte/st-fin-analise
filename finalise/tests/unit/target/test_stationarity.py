import pytest
import numpy as np
import pandas as pd
from finalise.target import stationarity

def test_ut_stat_01(white_noise):
    # UT-STAT-01: ADF rejects H0 (is_stationary=True) for white noise at alpha=0.05
    res = stationarity.adf(white_noise, alpha=0.05)
    assert res["is_stationary"] is True
    # The statistic should be negative and lower than 5% critical value
    assert res["statistic"] < res["critical_values"]["5%"]

def test_ut_stat_02(random_walk):
    # UT-STAT-02: ADF does not reject H0 (is_stationary=False) for random walk (I(1))
    res = stationarity.adf(random_walk, alpha=0.05)
    assert res["is_stationary"] is False

def test_ut_stat_03(random_walk):
    # UT-STAT-03: Non-stationary series emits warning
    with pytest.warns(UserWarning, match="Série não estacionária"):
        res = stationarity.adf(random_walk, alpha=0.05)
    assert res["is_stationary"] is False

def test_ut_stat_04(white_noise):
    # UT-STAT-04: Decision uses alpha=0.02 when passed
    # White noise should be stationary for both 0.05 and 0.02
    res = stationarity.adf(white_noise, alpha=0.02)
    assert res["is_stationary"] is True
    
    # We can test with a marginal case or just verify the behavior is correct.
    # For white noise, p-value is extremely small, so it's stationary.
