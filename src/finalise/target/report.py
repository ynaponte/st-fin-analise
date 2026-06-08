import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def generate(results: dict, best_horizon: int) -> dict:
    """
    Generates a terminal report using rich and interactive plots using plotly.
    Returns a dictionary of plotly Figure objects.
    """
    console = Console()
    
    console.print(Panel.fit(
        f"[bold green]Resumo Executivo da Análise do Ativo Alvo (k* = {best_horizon})[/bold green]",
        border_style="green"
    ))
    
    # Create Table of metrics across all scales
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Escala (k)", style="cyan", justify="center")
    table.add_column("Estacionária?", style="yellow", justify="center")
    table.add_column("Normal?", style="yellow", justify="center")
    table.add_column("Média", justify="right")
    table.add_column("Desvio Padrão", justify="right")
    table.add_column("Hurst (H)", justify="right")
    table.add_column("|H - 0.5|", justify="right")
    table.add_column("Auto-MI Max", justify="right")
    table.add_column("Entropia (nats)", justify="right")
    
    for k, v in results.items():
        is_stat = "[green]Sim[/green]" if v.get("is_stationary", True) else "[red]Não[/red]"
        is_norm = "[green]Sim[/green]" if v.get("is_normal", True) else "[red]Não[/red]"
        mean = f"{v.get('mean', 0.0):.6f}"
        std = f"{v.get('std', 0.0):.6f}"
        h = f"{v.get('hurst', 0.5):.4f}"
        h_diff = f"{v.get('hurst_diff', 0.0):.4f}"
        mi_max = f"{v.get('auto_mi_max', 0.0):.4f}"
        ent = f"{v.get('shannon', 0.0):.4f}"
        
        table.add_row(str(k), is_stat, is_norm, mean, std, h, h_diff, mi_max, ent)
        
    console.print(table)
    
    best_m = results.get(best_horizon, {})
    justification = (
        f"O horizonte k={best_horizon} foi selecionado como a escala ótima porque apresenta "
        f"maior desvio em relação ao ruído branco (|H - 0.5| = {best_m.get('hurst_diff', 0.0):.4f}), "
        f"auto-informação mútua máxima de {best_m.get('auto_mi_max', 0.0):.4f} e "
        f"entropia de Shannon de {best_m.get('shannon', 0.0):.4f} nats."
    )
    console.print(Panel(justification, title="Justificativa da Seleção", border_style="blue"))
    
    figures = {}
    
    # 2. Histogram + KDE plot
    if "returns" in best_m:
        ret = best_m["returns"]
        fig_dist = go.Figure()
        # Import fd_bins locally to avoid circular dependencies
        from .entropy import fd_bins
        n_bins = fd_bins(ret)
        fig_dist.add_trace(go.Histogram(
            x=ret,
            nbinsx=n_bins,
            histnorm='probability density',
            name='Log-retornos',
            opacity=0.6,
            marker_color='#1f77b4'
        ))
        try:
            kde = gaussian_kde(ret)
            kde_x = np.linspace(ret.min(), ret.max(), 200)
            kde_y = kde(kde_x)
            fig_dist.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color='#ff7f0e', width=2)
            ))
        except Exception:
            pass
        fig_dist.update_layout(
            title=f"Distribuição de Log-retornos (k={best_horizon}) com Histograma + KDE",
            xaxis_title="Retorno",
            yaxis_title="Densidade de Probabilidade",
            template="plotly_white"
        )
        figures["distribution"] = fig_dist

    # 3. Auto-MI Profile plot
    if "auto_mi_profile" in best_m:
        mi_profile = best_m["auto_mi_profile"]
        lags = list(range(1, len(mi_profile) + 1))
        fig_mi = go.Figure()
        fig_mi.add_trace(go.Bar(
            x=lags,
            y=mi_profile,
            name='Auto-MI',
            marker_color='#2ca02c'
        ))
        fig_mi.update_layout(
            title=f"Perfil de Auto-Informação Mútua (k={best_horizon})",
            xaxis_title="Lag",
            yaxis_title="Mutual Information (nats)",
            template="plotly_white"
        )
        figures["auto_mi"] = fig_mi

    # 4. ACF & PACF plot
    if "acf_pacf" in best_m:
        ap = best_m["acf_pacf"]
        acf_vals = ap["acf"]
        pacf_vals = ap["pacf"]
        
        fig_acf = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
        lags_idx = list(range(len(acf_vals)))
        
        fig_acf.add_trace(go.Bar(x=lags_idx, y=acf_vals, name="ACF", marker_color='#1f77b4'), row=1, col=1)
        fig_acf.add_trace(go.Bar(x=lags_idx, y=pacf_vals, name="PACF", marker_color='#ff7f0e'), row=1, col=2)
        
        n = len(best_m.get("returns", []))
        if n > 0:
            ci = 1.96 / np.sqrt(n)
            for col in [1, 2]:
                fig_acf.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=col)
                fig_acf.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=col)
                
        fig_acf.update_layout(
            title_text=f"Autocorrelação e Autocorrelação Parcial (k={best_horizon})",
            template="plotly_white",
            showlegend=False
        )
        figures["acf_pacf"] = fig_acf

    # 5. Hurst Comparison plot
    fig_hurst = go.Figure()
    ks = list(results.keys())
    hursts = [results[k].get("hurst", 0.5) for k in ks]
    fig_hurst.add_trace(go.Bar(
        x=[f"k={k}" for k in ks],
        y=hursts,
        name="Hurst Exponent",
        marker_color='#9467bd'
    ))
    fig_hurst.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="Ruído Branco (H=0.5)")
    fig_hurst.update_layout(
        title="Comparação de Expoente de Hurst por Escala",
        xaxis_title="Escala (k em dias)",
        yaxis_title="Hurst Exponent",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )
    figures["hurst"] = fig_hurst
    
    return figures
