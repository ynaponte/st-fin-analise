import warnings
from rich.console import Console

def choose_horizon(results: dict) -> int:
    """
    Ranks horizons by the vector (|H - 0.5|, maximum auto-MI, Shannon entropy).
    Non-stationary horizons are excluded from the ranking.
    Raises a warning if no signal is detected (i.e., |H - 0.5| < 0.05 and auto-MI is not significant).
    """
    # 1. Exclude non-stationary scales
    stationary_results = {
        k: v for k, v in results.items() if v.get("is_stationary", True)
    }
    
    if not stationary_results:
        raise ValueError("Nenhum horizonte de tempo é estacionário.")
        
    # 2. Rank remaining horizons
    def get_sorting_tuple(k):
        metrics = stationary_results[k]
        h_diff = metrics.get("hurst_diff", 0.0)
        auto_mi_max = metrics.get("auto_mi_max", 0.0)
        shannon_val = metrics.get("shannon", 0.0)
        return (h_diff, auto_mi_max, shannon_val)
        
    sorted_horizons = sorted(
        stationary_results.keys(),
        key=get_sorting_tuple,
        reverse=True
    )
    best_horizon = sorted_horizons[0]
    
    # 3. Check for signal detection
    best_metrics = stationary_results[best_horizon]
    h_diff = best_metrics.get("hurst_diff", 0.0)
    auto_mi_sig = best_metrics.get("auto_mi_significant", False)
    
    # Fallback to a threshold heuristic if not explicitly set
    if not auto_mi_sig and best_metrics.get("auto_mi_max", 0.0) > 0.02:
        auto_mi_sig = True
        
    if h_diff < 0.05 and not auto_mi_sig:
        console = Console()
        console.print(
            f"[bold yellow]Aviso: Nenhum sinal detectado no horizonte selecionado k={best_horizon} "
            f"(|H - 0.5|={h_diff:.4f} < 0.05 e auto-MI não significativa).[/bold yellow]"
        )
        warnings.warn("Nenhum sinal detectado", UserWarning)
        
    return best_horizon
