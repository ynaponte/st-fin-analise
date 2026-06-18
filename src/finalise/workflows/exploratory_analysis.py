"""
workflows.exploratory_analysis
------------------------------
Relatório Factual Exploratório Expandido.
Computa métricas descritivas (média, desvio, normalidade), estacionariedade (ADF, Hurst),
correlações (Pearson, Spearman) e métricas de informação (JSD, MI, Granger).
Trabalha com log-retornos NÃO suavizados, servindo de baseline informativo.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from finalise.config import Config
from finalise.analysis.transformation import log_returns
from finalise.analysis.entropy import mutual_information_lags, jensen_shannon_divergence
from finalise.analysis.base import granger_causality
from finalise.analysis.stationarity import adf
from finalise.analysis.descriptive import hurst


class ExploratoryAnalysis:
    """
    Gera um relatório factual robusto de relações e propriedades estatísticas.
    """

    def __init__(
        self,
        prices_dict: dict[str, pd.Series],
        config: Config
    ):
        self.prices_dict = prices_dict
        self.config = config
        self.console = Console()

    def run(self):
        """Executa a análise exploratória completa e exibe as tabelas."""
        self.console.print(Panel("[bold cyan]Análise Exploratória Factual (Baseline de Retornos Não Suavizados)[/bold cyan]", border_style="cyan"))

        target_ticker = self.config.target_ticker
        candidate_tickers = [t for t in self.config.predictor_tickers if t != target_ticker]

        # 1. Log-retornos BRUTOS (sem smoothing)
        returns = {}
        for ticker in [target_ticker] + candidate_tickers:
            if ticker in self.prices_dict:
                ret = log_returns(self.prices_dict[ticker], k=1)
                if not ret.empty:
                    returns[ticker] = ret

        if target_ticker not in returns:
            self.console.print("[red]Erro: Série alvo vazia após transformação em log-retorno bruto.[/red]")
            return

        target_ret = returns[target_ticker]
        
        # ── Tabela 1: Descritivas, Normalidade e Estacionariedade ────────────
        self.console.print("\n[bold yellow]Tabela 1: Propriedades das Séries (Log-Retornos Brutos)[/bold yellow]")
        
        t1 = Table(show_header=True, header_style="bold magenta")
        t1.add_column("Série", style="cyan")
        t1.add_column("Média", justify="right")
        t1.add_column("Desv. Padrão", justify="right")
        t1.add_column("Curtose", justify="right")
        t1.add_column("Assimetria", justify="right")
        t1.add_column("A-D Stat (Norm)", justify="right")
        t1.add_column("ADF p-valor", justify="right")
        t1.add_column("Hurst", justify="right")

        for ticker in [target_ticker] + candidate_tickers:
            if ticker not in returns:
                continue
            r = returns[ticker]
            # Estatísticas Descritivas
            mean_val = r.mean()
            std_val = r.std()
            kurt_val = scipy_stats.kurtosis(r, fisher=True)
            skew_val = scipy_stats.skew(r)
            
            # Anderson-Darling (Normalidade)
            try:
                ad_res = scipy_stats.anderson(r, dist='norm')
                ad_stat = ad_res.statistic
            except Exception:
                ad_stat = np.nan
                
            # ADF Estacionariedade
            adf_res = adf(r, alpha=self.config.alpha)
            adf_pval = adf_res['p_value']
            
            # Hurst (feito nos preços brutos ou retornos? Hurst normalmente nos preços faz sentido para random walk, mas aqui fazemos nos retornos brutos conforme solicitado "estacionariedade da serie e hurst movidos pra ca")
            h_res = hurst(self.prices_dict[ticker])
            h_val = h_res['hurst']

            t1.add_row(
                f"[bold]{ticker}[/bold]" if ticker == target_ticker else ticker,
                f"{mean_val:.4f}",
                f"{std_val:.4f}",
                f"{kurt_val:.4f}",
                f"{skew_val:.4f}",
                f"{ad_stat:.2f}",
                f"[green]{adf_pval:.4f}[/green]" if adf_pval < self.config.alpha else f"[red]{adf_pval:.4f}[/red]",
                f"{h_val:.4f}"
            )
        self.console.print(t1)

        # ── Tabela 2: Relações Candidato -> Alvo ────────────
        self.console.print("\n[bold yellow]Tabela 2: Relações Iniciais (Candidato vs Alvo)[/bold yellow]")
        
        t2 = Table(show_header=True, header_style="bold magenta")
        t2.add_column("Candidato", style="cyan")
        t2.add_column("Pearson (r)", justify="right")
        t2.add_column("Spearman (rho)", justify="right")
        t2.add_column("JSD", justify="right")
        t2.add_column("Max MI (Bits)", justify="right")
        t2.add_column("Granger Lags Sig.", justify="left")

        for ticker in candidate_tickers:
            if ticker not in returns:
                continue
            c_ret_raw = returns[ticker]

            # Alinhar as séries
            aligned = pd.concat([c_ret_raw, target_ret], axis=1, join='inner').dropna()
            if aligned.empty or len(aligned) < 2:
                continue
            c_ret = aligned.iloc[:, 0]
            target_aligned = aligned.iloc[:, 1]

            # Correlações Simples
            pearson_val, _ = scipy_stats.pearsonr(c_ret, target_aligned)
            spearman_val, _ = scipy_stats.spearmanr(c_ret, target_aligned)

            # JSD
            jsd_df = jensen_shannon_divergence([c_ret, target_aligned])
            jsd_val = float(jsd_df.iloc[0, 1])

            # MI
            mi_result = mutual_information_lags(target_aligned, c_ret, max_lag=self.config.lag_max, k=self.config.knn_k)
            max_mi = max([mi for _, mi in mi_result]) if mi_result else 0.0

            # Granger
            g_result = granger_causality(target_aligned, [c_ret], lag_max=self.config.lag_max, alpha=self.config.alpha)
            if not g_result:
                granger_str = "[red]Erro[/red]"
            else:
                driver_key = next(iter(g_result))
                g_data = g_result[driver_key]
                sig_lags = [str(lag) for lag, p_val in g_data.get("all_p_values", {}).items() if p_val < self.config.alpha]
                granger_str = ", ".join(sig_lags) if sig_lags else "[red]Nenhum[/red]"

            t2.add_row(
                ticker,
                f"{pearson_val:.4f}",
                f"{spearman_val:.4f}",
                f"{jsd_val:.4f}",
                f"{max_mi:.4f}",
                granger_str
            )
        self.console.print(t2)
        self.console.print("\n")

        self.plot_visualizations(returns)

    def plot_visualizations(self, returns: dict[str, pd.Series]):
        self.console.print("[bold cyan]Gerando visualizações...[/bold cyan]")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy.stats import norm
        from finalise.analysis.entropy import _ksg_mi
        import numpy as np

        tickers = list(self.prices_dict.keys())
        n_tickers = len(tickers)
        
        # 1. Séries com e sem suavização
        fig1 = make_subplots(rows=n_tickers, cols=1, shared_xaxes=True, subplot_titles=[f'Original vs Suavizada: {t}' for t in tickers])
        for i, ticker in enumerate(tickers, start=1):
            series = self.prices_dict[ticker]
            smoothed = series.rolling(window=21, min_periods=1).mean()
            fig1.add_trace(go.Scatter(x=series.index, y=series, name=f'Original {ticker}', opacity=0.6, line=dict(color='blue')), row=i, col=1)
            fig1.add_trace(go.Scatter(x=smoothed.index, y=smoothed, name=f'MM21 {ticker}', line=dict(color='red', width=2)), row=i, col=1)
        fig1.update_layout(height=300 * n_tickers, title_text="Séries com e sem suavização", showlegend=False)
        fig1.show()

        # 2. Log-retornos sobrepostos pela normal
        ret_tickers = [t for t in tickers if t in returns]
        n_ret_tickers = len(ret_tickers)
        if n_ret_tickers > 0:
            fig2 = make_subplots(rows=n_ret_tickers, cols=1, subplot_titles=[f'Log-Retornos: {t}' for t in ret_tickers])
            for i, ticker in enumerate(ret_tickers, start=1):
                ret = returns[ticker]
                fig2.add_trace(go.Histogram(x=ret, histnorm='probability density', name=f'Retornos {ticker}', opacity=0.5, marker_color='blue'), row=i, col=1)
                
                mu, std = norm.fit(ret)
                xmin, xmax = ret.min(), ret.max()
                x_vals = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x_vals, mu, std)
                fig2.add_trace(go.Scatter(x=x_vals, y=p, mode='lines', name=f'Normal μ={mu:.4f}, σ={std:.4f}', line=dict(color='black', width=2)), row=i, col=1)
            fig2.update_layout(height=300 * n_ret_tickers, title_text="Distribuição dos Log-Retornos", showlegend=False)
            fig2.show()

        # Alinhar todos os retornos para as matrizes
        ret_df = pd.DataFrame(returns).dropna()
        if ret_df.empty:
            self.console.print("[red]Erro: Sem dados alinhados para as matrizes de calor.[/red]")
            return
            
        ret_cols = ret_df.columns.tolist()
        
        # 3. Matriz de calor da JSD
        jsd_df = jensen_shannon_divergence([ret_df[c] for c in ret_cols])
        fig3 = go.Figure(data=go.Heatmap(z=jsd_df.values, x=ret_cols, y=ret_cols, colorscale='YlOrRd', text=np.round(jsd_df.values, 4), texttemplate="%{text}"))
        fig3.update_layout(title="Heatmap da Jensen-Shannon Divergence (JSD)", width=600, height=600)
        fig3.show()
        
        # 4. Matriz de calor para cada correlação (Pearson, Spearman)
        pearson_corr = ret_df.corr(method='pearson')
        spearman_corr = ret_df.corr(method='spearman')
        
        fig4 = make_subplots(rows=1, cols=2, subplot_titles=["Correlação de Pearson", "Correlação de Spearman"])
        fig4.add_trace(go.Heatmap(z=pearson_corr.values, x=ret_cols, y=ret_cols, colorscale='RdBu', zmin=-1, zmax=1, text=np.round(pearson_corr.values, 4), texttemplate="%{text}", coloraxis="coloraxis"), row=1, col=1)
        fig4.add_trace(go.Heatmap(z=spearman_corr.values, x=ret_cols, y=ret_cols, colorscale='RdBu', zmin=-1, zmax=1, text=np.round(spearman_corr.values, 4), texttemplate="%{text}", coloraxis="coloraxis"), row=1, col=2)
        fig4.update_layout(title_text="Heatmaps de Correlações", coloraxis=dict(colorscale='RdBu', cmin=-1, cmax=1), width=1000, height=500)
        fig4.show()
        
        # 5. Matriz de calor para Informação Mútua
        mi_matrix = pd.DataFrame(0.0, index=ret_cols, columns=ret_cols)
        for c1 in ret_cols:
            for c2 in ret_cols:
                if c1 == c2:
                    mi_matrix.loc[c1, c2] = np.nan # Evita calcular auto-informação mutua (entropia)
                else:
                    # Instante MI usando KSG em nats, converter para bits
                    val = _ksg_mi(ret_df[c1].values, ret_df[c2].values, k=self.config.knn_k) / np.log(2)
                    mi_matrix.loc[c1, c2] = val
                    
        fig5 = go.Figure(data=go.Heatmap(z=mi_matrix.values, x=ret_cols, y=ret_cols, colorscale='Viridis', text=np.round(mi_matrix.values, 4), texttemplate="%{text}"))
        fig5.update_layout(title="Heatmap da Informação Mútua (bits)", width=600, height=600)
        fig5.show()

