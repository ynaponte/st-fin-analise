import numpy as np
import pandas as pd

def dfa(series: pd.Series) -> tuple[float, float]:
    """
    Computes the Hurst exponent using Detrended Fluctuation Analysis (DFA).
    Returns a tuple (H, abs(H - 0.5)).
    
    If the computed scaling exponent alpha is >= 1.0, it is mapped to H = alpha - 1.0.
    H is clipped to the interval (0.001, 0.999).
    """
    x = np.asarray(series)
    N = len(x)
    
    # 1. Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(x - np.mean(x))
    
    # 2. Define scale sizes (s)
    scale_min = 10
    scale_max = max(scale_min + 5, N // 6)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), num=20, dtype=int))
    
    fluctuations = []
    for s in scales:
        num_segments = N // s
        F2 = 0.0
        for i in range(num_segments):
            segment = y[i*s : (i+1)*s]
            x_seg = np.arange(s)
            # Linear fit
            poly = np.polyfit(x_seg, segment, 1)
            fit = np.polyval(poly, x_seg)
            F2 += np.mean((segment - fit) ** 2)
        fluctuations.append(np.sqrt(F2 / num_segments))
        
    # Fit log(F) vs log(scale) to find alpha safely
    log_scales = np.log(scales)
    log_flucts = np.log(np.clip(fluctuations, 1e-15, None))
    
    try:
        alpha, _ = np.polyfit(log_scales, log_flucts, 1)
        if np.isnan(alpha) or np.isinf(alpha):
            alpha = 0.5
    except Exception:
        alpha = 0.5
        
    # Map alpha to H
    if alpha >= 1.0:
        H = 0.5 + (alpha - 1.0) / 2.0
    else:
        H = alpha
        
    H = float(np.clip(H, 0.001, 0.999))
    
    return H, float(abs(H - 0.5))
