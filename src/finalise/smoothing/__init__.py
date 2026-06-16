"""
finalise.smoothing
------------------
Smoothing filters and optimal window size analysis.

Modules
-------
methods – SMA, TMA, EMA, Gaussian smoothing with strategy pattern and delay compensation.
window  – Cumulative PSD analysis to find the smallest optimal window.
"""

from .methods import (
    SmoothingPolicy, 
    SMAPolicy, 
    EMAPolicy, 
    TMAPolicy, 
    GaussianPolicy, 
    apply_smoothing
)
from .window import find_optimal_window
