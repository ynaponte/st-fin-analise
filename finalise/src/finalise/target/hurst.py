import numpy as np
import pandas as pd

def dfa(series: pd.Series) -> tuple[float, float]:
    """
    Computes the Hurst exponent using Detrended Fluctuation Analysis (DFA).

    The input series must be stationary (I(0)), e.g. log-returns.
    The DFA integrates the series internally (cumulative sum of deviations
    from the mean) and then estimates the scaling exponent α from
    fluctuation vs. scale in log-log space.

    Mapping of the DFA scaling exponent α to the Hurst exponent H
    (Peng et al., 1994; Kantelhardt, 2002):
      - For stationary input (fGn / I(0)):  α ∈ (0, 1)  →  H = α
      - For non-stationary input (fBm / I(1)):  α ∈ (1, 2)  →  H = α − 1

    Returns a tuple ``(H, |H − 0.5|)``.
    H is clipped to the interval (0.001, 0.999).
    """
    x = np.asarray(series)
    N = len(x)
    
    # 1. Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(x - np.mean(x))
    
    # 2. Define scale sizes (s)
    # Kantelhardt recommends scale_min = 4 and scale_max = N // 4
    scale_min = 4
    scale_max = max(scale_min + 2, N // 4)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), num=20, dtype=int))
    
    # Ensure at least a few points for regression
    if len(scales) < 4:
        scales = np.arange(scale_min, max(scale_min + 4, scale_max + 1))
    
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
        
    # Map DFA scaling exponent α to Hurst exponent H.
    # Stationary input (fGn / I(0)):  α ∈ (0, 1) → H = α
    # Non-stationary input (fBm / I(1)):  α ∈ (1, 2) → H = α − 1
    if alpha >= 1.0:
        H = alpha - 1.0
    else:
        H = alpha
        
    H = float(np.clip(H, 0.001, 0.999))
    
    return H, float(abs(H - 0.5))
