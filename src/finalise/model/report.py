import plotly.graph_objects as go
import plotly.express as px
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import numpy as np

def generate(
    results: dict,
    target_analysis=None,
    predictor_analysis=None,
    target_series=None,
    config=None,
    horizon=None,
) -> dict:
    """
    Gera o relatório consolidado usando `rich` no terminal e retorna 
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
    
    console.print()
    console.print(Panel(
        "[bold white]RESUMO EXECUTIVO CONSOLIDADO - PIPELINE FINALISE[/bold white]",
        style="bold white on blue",
        expand=False
    ))
    
    # 1. Fase 1: Análise do Alvo
    if target_analysis is not None:
        h = target_analysis.horizon
        alpha = target_analysis.alpha
        target_ticker = target_analysis.config.target_ticker
        
        best_res = target_analysis.results.get(h, {})
        is_stationary = "Sim" if best_res.get("is_stationary", True) else "Não"
        hurst = best_res.get("hurst", 0.5)
        entropy = best_res.get("shannon", 0.0)
        auto_mi_max = best_res.get("auto_mi_max", 0.0)
        
        t_table = Table(show_header=True, header_style="bold green", box=None)
        t_table.add_column("Métrica", style="cyan")
        t_table.add_column("Valor", justify="right")
        t_table.add_row("Horizonte Escolhido (k*)", f"{h} dias")
        t_table.add_row("Significância (alpha)", f"{alpha}")
        t_table.add_row("Estacionário?", f"{is_stationary}")
        t_table.add_row("Expoente de Hurst (H)", f"{hurst:.4f}")
        t_table.add_row("Entropia de Shannon", f"{entropy:.4f} nats")
        t_table.add_row("Auto-Informação Mútua Max", f"{auto_mi_max:.4f}")
        
        console.print(Panel(
            t_table,
            title=f"[bold green]Fase 1: Análise do Ativo Alvo ({target_ticker})[/bold green]",
            border_style="green"
        ))
        
    # 2. Fase 2: Seleção de Preditores
    if predictor_analysis is not None:
        selected = predictor_analysis.selected
        cointegration = predictor_analysis.cointegration
        
        p_table = Table(show_header=True, header_style="bold blue", box=None)
        p_table.add_column("Ticker", style="cyan")
        p_table.add_column("Componente", style="yellow")
        p_table.add_column("Lag (tau*)", justify="center")
        p_table.add_column("Tipo de Relação", justify="center")
        p_table.add_column("Estatística/Métrica", justify="right")
        
        for sel in selected:
            val_str = f"MI: {sel.mi_value:.4f}" if sel.relation_type == "non-linear" else f"Granger p-val: {sel.granger_pvalue:.4f}"
            p_table.add_row(sel.ticker, sel.component, str(sel.lag_tau), sel.relation_type.capitalize(), val_str)
            
        coint_tickers = [t for t, c in cointegration.items() if c.get("is_cointegrated", False)]
        coint_str = ", ".join(coint_tickers) if coint_tickers else "Nenhum"
        
        console.print(Panel(
            p_table,
            title=f"[bold blue]Fase 2: Seleção de Preditores (Cointegrados: {coint_str})[/bold blue]",
            border_style="blue"
        ))
        
    # 3. Fase 3: Validação do Modelo
    m_table = Table(show_header=False, box=None)
    m_table.add_row("[cyan]Status do Modelo[/cyan]", f"[bold {status_color}]{status_text}[/bold {status_color}]")
    m_table.add_row("[cyan]Retorno Acumulado do Modelo[/cyan]", f"{model_return:.4f}")
    m_table.add_row("[cyan]Z-Score contra Agentes[/cyan]", f"{z_score:.2f} desvios (sigma)")
    m_table.add_row("[cyan]Distribuição dos Agentes[/cyan]", f"media = {agents_mean:.4f}, desvio padrao = {agents_std:.4f}")
    
    console.print(Panel(
        m_table,
        title=f"[bold {status_color}]Fase 3: Validação do Modelo (Machine Learning)[/bold {status_color}]",
        border_style=status_color
    ))
    
    # 4. Janela de Previsão Futura
    h_win = horizon
    if h_win is None and target_analysis is not None:
        h_win = target_analysis.horizon
    if h_win is None and predictor_analysis is not None:
        h_win = predictor_analysis.horizon
    if h_win is None and config is not None and getattr(config, "horizons", None):
        h_win = config.horizons[0]
    if h_win is None:
        h_win = 5
        
    last_date = None
    if target_series is not None and not target_series.empty:
        last_date = target_series.index[-1]
    elif target_analysis is not None and hasattr(target_analysis, "prices_dict") and config is not None:
        target_prices = target_analysis.prices_dict.get(config.target_ticker)
        if target_prices is not None and not target_prices.empty:
            last_date = target_prices.index[-1]
            
    if last_date is not None:
        try:
            import pandas as pd
            last_date_dt = pd.to_datetime(last_date)
            start_pred = last_date_dt + pd.offsets.BDay(1)
            end_pred = last_date_dt + pd.offsets.BDay(h_win)
            
            start_str = start_pred.strftime('%Y-%m-%d')
            end_str = end_pred.strftime('%Y-%m-%d')
            last_str = last_date_dt.strftime('%Y-%m-%d')
            
            target_ticker_str = config.target_ticker if config else "Alvo"
            window_text = (
                f"O modelo treinado utilizará os dados de fechamento disponíveis até [bold cyan]{last_str}[/bold cyan] "
                f"para prever o retorno acumulado do ativo [bold yellow]{target_ticker_str}[/bold yellow] "
                f"para a seguinte janela de tempo futura:\n\n"
                f"[bold green]JANELA DE PREVISÃO ({h_win} dias úteis): de {start_str} a {end_str}[/bold green]"
            )
        except Exception:
            window_text = (
                f"O modelo utilizará as últimas observações disponíveis para prever o retorno "
                f"do horizonte de [bold green]{h_win}[/bold green] períodos à frente."
            )
            
        console.print(Panel(
            window_text,
            title="PROJEÇÃO / JANELA DE PREVISÃO FUTURA",
            border_style="yellow"
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
