import numpy as np
import pandas as pd
import pytest
from finalise.predictors import smoothing

def test_ut_smooth_apply():
    # Test all smoothing methods
    series = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    
    # SMA
    sma = smoothing.apply(series, window=3, method="SMA")
    assert len(sma) == len(series)
    assert np.isclose(sma.iloc[0], 10.0) # min_periods=1
    assert np.isclose(sma.iloc[2], 11.0) # (10+11+12)/3
    
    # EMA
    ema = smoothing.apply(series, window=3, method="EMA")
    assert len(ema) == len(series)
    
    # DEMA
    dema = smoothing.apply(series, window=3, method="DEMA")
    assert len(dema) == len(series)
    
    # TMA
    tma = smoothing.apply(series, window=3, method="TMA")
    assert len(tma) == len(series)
    
    # Gaussian
    gauss = smoothing.apply(series, window=3, method="Gaussian")
    assert len(gauss) == len(series)
    
    # Unknown method
    with pytest.raises(ValueError, match="desconhecido"):
        smoothing.apply(series, window=3, method="UNKNOWN")

def test_ut_smooth_group_delay():
    # Test group delay values
    assert smoothing.group_delay(window=5, method="SMA") == pytest.approx(2.0, abs=1e-5)
    assert smoothing.group_delay(window=5, method="EMA") == pytest.approx(2.0, abs=1e-5)
    assert smoothing.group_delay(window=5, method="TMA") == pytest.approx(2.0, abs=1e-5)
    assert smoothing.group_delay(window=5, method="Gaussian") == pytest.approx(2.0, abs=1e-5)
    assert smoothing.group_delay(window=5, method="DEMA") == pytest.approx(0.0, abs=1e-5)
    
    with pytest.raises(ValueError, match="desconhecido"):
        smoothing.group_delay(window=5, method="UNKNOWN")
