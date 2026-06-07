import pytest
import warnings
from finalise.target import selector

def test_ut_sel_t_01():
    # UT-SEL-T-01: Horizon with largest |H - 0.5| is selected
    results = {
        1: {
            "is_stationary": True,
            "hurst_diff": 0.04,
            "auto_mi_max": 0.1,
            "shannon": 2.5,
            "alpha": 0.05
        },
        5: {
            "is_stationary": True,
            "hurst_diff": 0.12, # largest |H - 0.5|
            "auto_mi_max": 0.01,
            "shannon": 2.4,
            "alpha": 0.05
        },
        21: {
            "is_stationary": True,
            "hurst_diff": 0.08,
            "auto_mi_max": 0.05,
            "shannon": 2.6,
            "alpha": 0.05
        }
    }
    
    best = selector.choose_horizon(results)
    assert best == 5

def test_ut_sel_t_02():
    # UT-SEL-T-02: Gate emits warning when |H-0.5| < 0.05 and auto-MI is not significant
    results = {
        1: {
            "is_stationary": True,
            "hurst_diff": 0.03, # < 0.05
            "auto_mi_max": 0.01, # < 0.02 (not significant)
            "auto_mi_significant": False,
            "shannon": 2.5,
            "alpha": 0.05
        }
    }
    
    with pytest.warns(UserWarning, match="Nenhum sinal detectado"):
        best = selector.choose_horizon(results)
    assert best == 1

def test_ut_sel_t_03():
    # UT-SEL-T-03: Horizons with ADF non-stationary are excluded
    results = {
        1: {
            "is_stationary": False, # Excluded!
            "hurst_diff": 0.2,
            "auto_mi_max": 0.5,
            "shannon": 2.1,
            "alpha": 0.05
        },
        5: {
            "is_stationary": True,
            "hurst_diff": 0.05,
            "auto_mi_max": 0.02,
            "shannon": 2.5,
            "alpha": 0.05
        }
    }
    
    best = selector.choose_horizon(results)
    assert best == 5
