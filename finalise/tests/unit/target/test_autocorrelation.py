import pytest
import numpy as np
import pandas as pd
from finalise.target import autocorrelation

def test_ut_acf_01(white_noise):
    # UT-ACF-01: ACF of white noise is within confidence interval (except lag 0)
    res = autocorrelation.acf_pacf(white_noise, lags=10, alpha=0.05)
    acf_vals = res["acf"]
    acf_conf = res["acf_confint"]
    
    # Lag 0 is always 1.0
    assert np.isclose(acf_vals[0], 1.0)
    
    # For lags 1..10, the values should lie inside the confidence intervals
    # acf_conf is a list of [lower, upper]
    for i in range(1, len(acf_vals)):
        lower, upper = acf_conf[i]
        assert lower <= acf_vals[i] <= upper

def test_ut_acf_02():
    # UT-ACF-02: ACF of AR(1) with phi=0.5 has lag=1 coefficient close to 0.5
    rng = np.random.default_rng(42)
    ar1 = np.zeros(2000)
    for t in range(1, 2000):
        ar1[t] = 0.5 * ar1[t-1] + rng.standard_normal()
    series = pd.Series(ar1)
    
    res = autocorrelation.acf_pacf(series, lags=5, alpha=0.05)
    assert np.isclose(res["acf"][1], 0.5, atol=0.06)

def test_ut_acf_03():
    # UT-ACF-03: ACF of r^2 of GARCH series has significant coefficients
    # Simulate a GARCH(1,1) process
    rng = np.random.default_rng(42)
    n = 2000
    w = 0.1
    alpha = 0.2
    beta = 0.7
    
    h = np.zeros(n)
    eps = np.zeros(n)
    h[0] = w / (1.0 - alpha - beta)
    
    for t in range(1, n):
        h[t] = w + alpha * (eps[t-1]**2) + beta * h[t-1]
        eps[t] = rng.standard_normal() * np.sqrt(h[t])
        
    series = pd.Series(eps)
    res = autocorrelation.acf_pacf(series, lags=10, alpha=0.05)
    
    # ACF of squared returns (acf_sq) should be significant at lag 1
    acf_sq_val = res["acf_sq"][1]
    # Significant means it lies outside the confidence interval under H0 of white noise
    se = 1.0 / np.sqrt(n)
    margin = 1.96 * se
    assert acf_sq_val > margin or acf_sq_val < -margin

def test_ut_acf_04():
    # UT-ACF-04: PACF cuts off after lag 1 for true AR(1)
    rng = np.random.default_rng(42)
    ar1 = np.zeros(2000)
    for t in range(1, 2000):
        ar1[t] = 0.6 * ar1[t-1] + rng.standard_normal()
    series = pd.Series(ar1)
    
    res = autocorrelation.acf_pacf(series, lags=5, alpha=0.05)
    pacf_vals = res["pacf"]
    pacf_conf = res["pacf_confint"]
    
    # PACF at lag 1 is significant (close to 0.6)
    assert np.isclose(pacf_vals[1], 0.6, atol=0.06)
    
    # PACF at lags 2..5 should be inside the confidence intervals (cut off)
    for i in range(2, len(pacf_vals)):
        lower, upper = pacf_conf[i]
        assert lower <= pacf_vals[i] <= upper
