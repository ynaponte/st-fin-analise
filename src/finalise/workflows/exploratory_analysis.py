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
            c_ret = returns[ticker]

            # Correlações Simples
            pearson_val, _ = scipy_stats.pearsonr(c_ret, target_ret)
            spearman_val, _ = scipy_stats.spearmanr(c_ret, target_ret)

            # JSD
            jsd_df = jensen_shannon_divergence([c_ret, target_ret])
            jsd_val = float(jsd_df.iloc[0, 1])

            # MI
            mi_result = mutual_information_lags(target_ret, c_ret, max_lag=self.config.lag_max, k=self.config.knn_k)
            max_mi = max([mi for _, mi in mi_result]) if mi_result else 0.0

            # Granger
            g_result = granger_causality(target_ret, [c_ret], lag_max=self.config.lag_max, alpha=self.config.alpha)
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
