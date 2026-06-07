import plotly.graph_objects as go
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def generate(results: dict, selected: list, cointegration: dict, config, alpha: float) -> dict:
    """
    Generates terminal summary using rich and interactive plots using plotly.
    Returns a dictionary of plotly Figure objects.
    """
    console = Console()
    
    console.print(Panel.fit(
        f"[bold green]Resumo Executivo da Seleção de Preditores (alpha = {alpha})[/bold green]",
        border_style="green"
    ))
    
    # 1. Table of Candidates and Selection Results
    c_table = Table(show_header=True, header_style="bold magenta", title="Candidatos Avaliados")
    c_table.add_column("Ticker", style="cyan")
    c_table.add_column("Componente", style="yellow")
    c_table.add_column("MI Max", justify="right")
    c_table.add_column("Lag Ótimo (τ*)", justify="center")
    c_table.add_column("MI Signif.?", justify="center")
    c_table.add_column("Granger p-val", justify="right")
    c_table.add_column("TE Obs.", justify="right")
    c_table.add_column("TE Limiar", justify="right")
    c_table.add_column("Relação", justify="center", style="bold green")
    
    for (ticker, component), res in results.items():
        mi_res = res["mi"]
        mi_max = mi_res["mi_profile"][mi_res["lag_opt"] - 1] if mi_res["mi_profile"] else 0.0
        lag_opt = mi_res["lag_opt"]
        mi_sig = "[green]Sim[/green]" if mi_res["is_significant"] else "[red]Não[/red]"
        
        # Granger
        granger_p = "-"
        if "granger" in res:
            granger_p = f"{res['granger']['p_value']:.4f}"
            
        # TE
        te_val = "-"
        te_thresh = "-"
        if "te" in res:
            te_val = f"{res['te']['te']:.4f}"
            te_thresh = f"{res['te']['threshold']:.4f}"
            
        # Relation type
        rel = "[red]Descartado[/red]"
        for sel in selected:
            if sel.ticker == ticker and sel.component == component:
                if sel.relation_type == "linear":
                    rel = "[bold blue]Linear[/bold blue]"
                else:
                    rel = "[bold green]Não-Linear[/bold green]"
                break
                
        c_table.add_row(
            ticker, component, f"{mi_max:.4f}", str(lag_opt), mi_sig,
            granger_p, te_val, te_thresh, rel
        )
        
    console.print(c_table)
    
    # 2. Table of Cointegration
    if cointegration:
        coint_table = Table(show_header=True, header_style="bold blue", title="Teste de Cointegração Engle-Granger (Preços Brutos)")
        coint_table.add_column("Par (X vs Target)", style="cyan")
        coint_table.add_column("Estatística t", justify="right")
        coint_table.add_column("p-valor", justify="right")
        coint_table.add_column("Cointegrado?", justify="center")
        coint_table.add_column("Beta (OLS)", justify="right")
        coint_table.add_column("Alpha (Const)", justify="right")
        
        for ticker, coint_res in cointegration.items():
            is_coint = "[green]Sim[/green]" if coint_res["is_cointegrated"] else "[red]Não[/red]"
            stat = f"{coint_res['statistic']:.4f}"
            p_val = f"{coint_res['p_value']:.4f}"
            beta = f"{coint_res['beta']:.4f}"
            alpha_const = f"{coint_res['alpha_const']:.4f}"
            
            coint_table.add_row(
                f"{ticker} vs {config.target_ticker}",
                stat, p_val, is_coint, beta, alpha_const
            )
        console.print(coint_table)
        
    # Justification summary
    n_selected = len(selected)
    justification = (
        f"Foram analisados {len(results)} candidatos a preditores. "
        f"Desses, {n_selected} foram selecionados com base no pipeline causal de decisão.\n"
    )
    if n_selected > 0:
        justification += "Candidatos selecionados:\n"
        for sel in selected:
            justification += f" - {sel.ticker} ({sel.component}): lag={sel.lag_tau}, tipo={sel.relation_type}, MI={sel.mi_value:.4f}\n"
    else:
        justification += "[yellow]Aviso: Nenhum preditor foi selecionado.[/yellow]\n"
        
    console.print(Panel(justification.strip(), title="Justificativa da Seleção de Preditores", border_style="cyan"))
    
    figures = {}
    
    # --- Plot 1: Heatmap of Cross-MI by Lag ---
    if results:
        # Build matrix of MI values: rows = candidates, cols = lags
        candidate_labels = []
        mi_matrix = []
        lags = list(range(1, config.lag_max + 1))
        
        for (ticker, component), res in results.items():
            mi_profile = res["mi"]["mi_profile"]
            if len(mi_profile) < config.lag_max:
                # pad with zeros if shorter
                mi_profile = list(mi_profile) + [0.0] * (config.lag_max - len(mi_profile))
            candidate_labels.append(f"{ticker} ({component})")
            mi_matrix.append(mi_profile[:config.lag_max])
            
        fig_mi = go.Figure(data=go.Heatmap(
            z=mi_matrix,
            x=[f"Lag {l}" for l in lags],
            y=candidate_labels,
            colorscale='Viridis',
            colorbar=dict(title="MI (nats)")
        ))
        fig_mi.update_layout(
            title="Heatmap de Informação Mútua Cruzada por Lag",
            xaxis_title="Defasagem (Lag)",
            yaxis_title="Candidato a Preditor",
            template="plotly_white"
        )
        figures["cross_mi_heatmap"] = fig_mi
        
    # --- Plot 2: Granger Causality p-values ---
    granger_candidates = []
    granger_pvals = []
    for (ticker, component), res in results.items():
        if "granger" in res:
            granger_candidates.append(f"{ticker} ({component})")
            granger_pvals.append(res["granger"]["p_value"])
            
    if granger_candidates:
        fig_granger = go.Figure()
        fig_granger.add_trace(go.Bar(
            x=granger_candidates,
            y=granger_pvals,
            marker_color=['#1f77b4' if p < alpha else '#ff7f0e' for p in granger_pvals],
            name="p-valor de Granger"
        ))
        fig_granger.add_hline(y=alpha, line_dash="dash", line_color="red", annotation_text=f"Limiar alpha ({alpha})")
        fig_granger.update_layout(
            title="p-valores do Teste de Causalidade de Granger no Lag Ótimo τ*",
            xaxis_title="Candidato",
            yaxis_title="p-valor",
            template="plotly_white",
            yaxis=dict(range=[0, 1.05])
        )
        figures["granger_pvalues"] = fig_granger
        
    # --- Plot 3: Transfer Entropy Permutation Distribution ---
    te_candidates = []
    te_observed = []
    te_null_dists = []
    for (ticker, component), res in results.items():
        if "te" in res:
            te_candidates.append(f"{ticker} ({component})")
            te_observed.append(res["te"]["te"])
            te_null_dists.append(res["te"]["null_distribution"])
            
    if te_candidates:
        fig_te = go.Figure()
        for i, (cand, obs, dist) in enumerate(zip(te_candidates, te_observed, te_null_dists)):
            # Boxplot of the null distribution
            fig_te.add_trace(go.Box(
                y=dist,
                name=cand,
                boxpoints=False,
                line_color='#7f7f7f',
                fillcolor='rgba(127,127,127,0.2)'
            ))
            # Marker for observed TE
            fig_te.add_trace(go.Scatter(
                x=[cand],
                y=[obs],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name="Observed TE" if i == 0 else "",
                showlegend=(i == 0)
            ))
            
        fig_te.update_layout(
            title="Distribuição Nula de Permutação de TE vs. Valor Observado (Estrela Vermelha)",
            xaxis_title="Candidato",
            yaxis_title="Transfer Entropy (nats)",
            template="plotly_white",
            showlegend=True
        )
        figures["te_permutation"] = fig_te
        
    return figures
