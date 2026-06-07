import pandas as pd
import scipy.stats as stats_api
from scipy.stats import anderson
from rich.console import Console
from rich.table import Table

def stats(series: pd.Series) -> dict:
    """
    Computes mean, standard deviation, skewness, and Pearson kurtosis.
    Runs the Anderson-Darling normality test.
    Returns a dictionary of statistics and prints a formatted summary using rich.
    """
    mean_val = float(series.mean())
    std_val = float(series.std())
    # Use pandas skew and kurtosis (convert excess kurtosis to Pearson by adding 3)
    skew_val = float(series.skew())
    kurtosis_val = float(series.kurt() + 3)
    
    # Anderson-Darling test for normality
    ad_result = anderson(series, dist='norm')
    
    sig_levels = ad_result.significance_level
    crit_vals = ad_result.critical_values
    
    # Look up critical value for 5% significance level
    idx_5 = 2
    if 5.0 in sig_levels:
        idx_5 = list(sig_levels).index(5.0)
    crit_5 = crit_vals[idx_5]
    
    # If the statistic is larger than critical value, reject normality
    is_normal = bool(ad_result.statistic < crit_5)
    alpha = 0.05 if is_normal else 0.02
    
    res = {
        "mean": mean_val,
        "std": std_val,
        "skewness": skew_val,
        "kurtosis": kurtosis_val,
        "is_normal": is_normal,
        "alpha": alpha,
        "ad_statistic": float(ad_result.statistic),
        "ad_critical_values": [float(x) for x in crit_vals],
        "ad_significance_levels": [float(x) for x in sig_levels]
    }
    
    # Rich print
    console = Console()
    table = Table(title="Estatísticas Descritivas & Teste de Normalidade")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="magenta")
    
    table.add_row("Média", f"{mean_val:.6f}")
    table.add_row("Desvio Padrão", f"{std_val:.6f}")
    table.add_row("Assimetria (Skewness)", f"{skew_val:.6f}")
    table.add_row("Curtose (Pearson)", f"{kurtosis_val:.6f}")
    table.add_row("Estatística AD", f"{ad_result.statistic:.6f}")
    table.add_row("Valor Crítico AD (5%)", f"{crit_5:.6f}")
    
    status_str = "[green]Normal (H0 não rejeitada)[/green]" if is_normal else "[red]Não-Normal (H0 rejeitada)[/red]"
    table.add_row("Status Normalidade", status_str)
    table.add_row("Nível de Significância (Alpha)", f"{alpha:.2f}")
    
    console.print(table)
    
    return res
