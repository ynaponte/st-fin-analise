import plotly.graph_objects as go
import plotly.express as px
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import numpy as np

def generate(results: dict) -> dict:
    """
    Gera o relatório do modelo usando `rich` no terminal e retorna 
    um dicionário com gráficos `plotly`.
    """
    console = Console()
    
    is_outlier = results.get("is_outlier", False)
    z_score = results.get("z_score", 0.0)
    model_return = results.get("model_return", 0.0)
    agents_mean = results.get("agents_mean", 0.0)
    agents_std = results.get("agents_std", 0.0)
    feature_importance = results.get("feature_importance", {})
    agent_returns = results.get("agent_returns", np.array([]))
    
    status_color = "green" if is_outlier else "red"
    status_text = "VÁLIDO (Outlier)" if is_outlier else "INVÁLIDO (Não supera agentes aleatórios)"
    
    console.print(Panel.fit(
        f"[bold {status_color}]Resumo Executivo do Modelo de Machine Learning[/bold {status_color}]\n"
        f"Status: [bold]{status_text}[/bold]\n"
        f"Retorno Acumulado do Modelo: {model_return:.4f}\n"
        f"Z-Score contra Agentes: {z_score:.2f}σ\n"
        f"Distribuição dos Agentes: μ = {agents_mean:.4f}, σ = {agents_std:.4f}",
        border_style=status_color
    ))
    
    # Feature Importance Table
    if feature_importance:
        table = Table(title="Importância das Features", show_header=True, header_style="bold magenta")
        table.add_column("Feature", style="cyan")
        table.add_column("Importância", justify="right")
        
        # Sort by importance descending
        sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_fi:
            table.add_row(feat, f"{imp:.4f}")
            
        console.print(table)
        
    figures = {}
    
    # Gráfico 1: Distribuição dos agentes com modelo marcado
    if len(agent_returns) > 0:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=agent_returns,
            name="Agentes Aleatórios",
            marker_color="lightslategray",
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_dist.add_vline(
            x=model_return, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Modelo (z={z_score:.2f})", 
            annotation_position="top right"
        )
        
        fig_dist.update_layout(
            title="Distribuição de Retornos (Modelo vs Agentes Aleatórios)",
            xaxis_title="Retorno Acumulado",
            yaxis_title="Frequência",
            template="plotly_white"
        )
        figures["agent_distribution"] = fig_dist
        
    # Gráfico 2: Importância das Features
    if feature_importance:
        feats = [x[0] for x in sorted_fi]
        imps = [x[1] for x in sorted_fi]
        
        fig_fi = go.Figure(go.Bar(
            x=imps,
            y=feats,
            orientation='h',
            marker_color="indigo"
        ))
        # Revert y-axis to show most important on top
        fig_fi.update_layout(
            title="Importância das Features (Decision Tree)",
            xaxis_title="Importância (Gini/MSE)",
            yaxis_title="Feature",
            template="plotly_white",
            yaxis=dict(autorange="reversed")
        )
        figures["feature_importance"] = fig_fi
        
    return figures
