"""
workflows.predictors_analysis
-----------------------------
Pipeline de seleção de preditores baseado em scoring contínuo.

Substitui ``finalise.predictors.PredictorAnalysis``.  Dado um alvo e N
séries candidatas (preços de fechamento), executa:

    0. Smoothing nos preços brutos
    1. Log-retornos (k=1)
    2. Hard gates   — ADF (estacionariedade) e Hurst (random walk)
    3. JSD          — peso global por candidato (similaridade distribucional)
    4. Auto-MI      — embedding dimension do alvo para a TE
    5. Score matrix — Granger F (gated by p), MI, TE por (candidato × lag)
    6. Normalização — min-max por métrica
    7. Score composto — média ponderada × JSD weight
    8. Lag consensual — argmax da soma dos scores
    9. Top-K        — seleção final no lag consensual
   10. Enriquecimento — STL + Johansen (pós-seleção)
   11. Descritivas  — stats, correlações (relatório)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from finalise.config import Config
from finalise.smoothing.methods import (
    apply_smoothing,
    SMAPolicy,
    EMAPolicy,
    TMAPolicy,
    GaussianPolicy,
    SmoothingPolicy,
)
from finalise.smoothing.window import find_optimal_window
from finalise.analysis.transformation import log_returns, stl_decomposition
from finalise.analysis.stationarity import adf
from finalise.analysis.descriptive import hurst, stats
from finalise.analysis.entropy import (
    mutual_information_lags,
    transfer_entropy,
    jensen_shannon_divergence,
)
from finalise.analysis.base import granger_causality, johansen_cointegration
from finalise.analysis.correlations import pearson, spearman, acf_pacf


# ---------------------------------------------------------------------------
# SelectedCandidate (compatible replica)
# ---------------------------------------------------------------------------
# Defined here to avoid triggering the broken circular import chain
# that goes through ``finalise.predictors.selector`` → ``finalise.target``.
# Structurally identical to ``finalise.predictors.selector.SelectedCandidate``.

@dataclass
class SelectedCandidate:
    """Candidato selecionado pelo pipeline de scoring."""

    ticker: str
    component: str
    series: pd.Series
    lag_tau: int
    relation_type: str          # "linear" | "non-linear"
    mi_value: float
    te_value: Optional[float]
    granger_pvalue: float


# ---------------------------------------------------------------------------
# Smoothing policy registry
# ---------------------------------------------------------------------------

_SMOOTHING_POLICIES: dict[str, SmoothingPolicy] = {
    "SMA": SMAPolicy(),
    "EMA": EMAPolicy(),
    "TMA": TMAPolicy(),
    "GAUSSIAN": GaussianPolicy(),
    "GAUSSIANA": GaussianPolicy(),
}


def _resolve_policy(method: str) -> SmoothingPolicy:
    """Maps a method name string to a ``SmoothingPolicy`` instance."""
    key = method.upper()
    if key not in _SMOOTHING_POLICIES:
        supported = ", ".join(sorted({k for k in _SMOOTHING_POLICIES}))
        raise ValueError(
            f"Método de suavização '{method}' não suportado. "
            f"Opções: {supported}"
        )
    return _SMOOTHING_POLICIES[key]


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Resultado completo do pipeline de seleção de preditores."""

    selected: List[SelectedCandidate]
    """Candidatos selecionados, todos com ``lag_tau = lag_consensus``."""

    lag_consensus: int
    """Lag consensual (ℓ*) que maximiza a soma dos scores."""

    score_matrix: pd.DataFrame
    """
    Matriz de scores compostos ``S(c, l)`` com shape
    ``(n_candidatos, lag_max)``.  Índice = ticker, colunas = ``lag_1 … lag_L``.
    """

    jsd_weights: Dict[str, float]
    """Pesos JSD ``(1 - JSD)`` por candidato."""

    auto_mi_lag: int
    """Lag ótimo da auto-MI do alvo, usado como ``y_lags`` na TE."""

    smoothing_window: int
    """Janela de suavização efetivamente utilizada."""

    smoothing_method: str
    """Nome do método de suavização utilizado."""

    stl_components: Dict[str, Dict[str, pd.Series]]
    """
    Componentes STL dos candidatos selecionados.
    ``{ticker: {"trend": …, "seasonal": …, "residual": …}}``.
    """

    cointegration: Dict[str, Any]
    """Resultado do teste de Johansen (pós-seleção)."""

    descriptives: Dict[str, Any]
    """Estatísticas descritivas dos candidatos selecionados."""

    dropped: Dict[str, str]
    """Candidatos descartados → motivo."""


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class PredictorsAnalysis:
    """
    Pipeline de seleção de preditores baseado em scoring contínuo.

    Parameters
    ----------
    prices_dict : dict[str, pd.Series]
        Dicionário ``{ticker: preços_fechamento}``.  Deve incluir o alvo.
    config : Config
        Configuração global (fornece target_ticker, predictor_tickers,
        smoothing_method, stl_period, alpha, knn_k, lag_max).
    smoothing_window : int | None
        Janela de suavização.  Se ``None``, calcula automaticamente via
        ``find_optimal_window`` sobre os preços do alvo.
    top_k : int
        Número máximo de candidatos a selecionar.
    score_threshold : float
        Score mínimo no lag consensual para seleção.
    hurst_tolerance : float
        Tolerância do Hurst gate: descarta se ``|H - 0.5| < tolerance``.
    w_granger, w_mi, w_te : float
        Pesos do score composto (devem somar 1.0).
    """

    def __init__(
        self,
        prices_dict: dict[str, pd.Series],
        config: Config,
        *,
        smoothing_window: Optional[int] = None,
        top_k: int = 5,
        score_threshold: float = 0.10,
        hurst_tolerance: float = 0.05,
        w_granger: float = 0.50,
        w_mi: float = 0.25,
        w_te: float = 0.25,
    ):
        self.prices_dict = prices_dict
        self.config = config

        self.smoothing_window = smoothing_window
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.hurst_tolerance = hurst_tolerance
        self.w_granger = w_granger
        self.w_mi = w_mi
        self.w_te = w_te

        self.result: Optional[PipelineResult] = None
        self.console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        """Executa o pipeline completo (etapas 0–11)."""
        self.console.print(Panel("[bold cyan]Iniciando Pipeline PredictorsAnalysis (v4)[/bold cyan]", border_style="cyan"))

        target_ticker = self.config.target_ticker
        candidate_tickers = [
            t for t in self.config.predictor_tickers if t != target_ticker
        ]
        alpha = self.config.alpha
        lag_max = self.config.lag_max
        knn_k = self.config.knn_k

        # ── Etapa 0: Smoothing (preços brutos) ───────────────────────
        self.console.print("\n[bold yellow]Etapa 0: Smoothing (Filtro Passa-Baixa)[/bold yellow]")
        policy = _resolve_policy(self.config.smoothing_method)

        window = self.smoothing_window
        if window is None:
            window = find_optimal_window(self.prices_dict[target_ticker])
            self.console.print(f"Janela ótima calculada via alvo: [bold green]{window}[/bold green]")
        else:
            self.console.print(f"Janela de suavização definida pelo usuário: [bold green]{window}[/bold green]")
            
        self.console.print(f"Método de suavização: [bold green]{self.config.smoothing_method}[/bold green]")

        smoothed: dict[str, pd.Series] = {}
        
        num_series = len(self.prices_dict)
        fig_smooth = make_subplots(rows=num_series, cols=1, shared_xaxes=True, subplot_titles=list(self.prices_dict.keys()))
        
        for i, (ticker, prices) in enumerate(self.prices_dict.items()):
            smoothed_series = apply_smoothing(prices, window, policy)
            smoothed[ticker] = smoothed_series
            
            # Adicionar ao plot
            fig_smooth.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name=f'{ticker} (Bruto)', line=dict(color='rgba(150,150,150,0.4)', width=1)), row=i+1, col=1)
            fig_smooth.add_trace(go.Scatter(x=smoothed_series.index, y=smoothed_series, mode='lines', name=f'{ticker} (Suavizado)', line=dict(width=2)), row=i+1, col=1)
            
        fig_smooth.update_layout(height=max(400, 200*num_series), title="Etapa 0: Preços Brutos vs Suavizados (Separados por Escala)", template="plotly_white", showlegend=False)
        fig_smooth.show()

        # ── Etapa 1: Log-retornos (k=1) ──────────────────────────────
        self.console.print("\n[bold yellow]Etapa 1: Transformação para Log-Retornos[/bold yellow]")
        returns: dict[str, pd.Series] = {}
        hist_data = []
        group_labels = []
        
        for ticker, prices_s in smoothed.items():
            ret = log_returns(prices_s, k=1)
            if not ret.empty:
                returns[ticker] = ret
                hist_data.append(ret.dropna().values)
                group_labels.append(ticker)

        self.console.print(f"Log-retornos calculados para {len(returns)} séries.")

        if target_ticker not in returns:
            raise ValueError(
                f"Séries de retorno do alvo '{target_ticker}' estão vazias "
                "após smoothing + log-retornos."
            )

        target_ret = returns[target_ticker]
        
        # Plotar Histograma em Subplots
        if hist_data:
            try:
                num_ret = len(returns)
                fig_hist = make_subplots(rows=num_ret, cols=1, shared_xaxes=True, subplot_titles=group_labels)
                for i, (ticker, data) in enumerate(zip(group_labels, hist_data)):
                    fig_hist.add_trace(go.Histogram(x=data, nbinsx=100, name=ticker), row=i+1, col=1)
                fig_hist.update_layout(height=max(400, 150*num_ret), title="Etapa 1: Distribuição dos Log-Retornos", template="plotly_white", showlegend=False)
                fig_hist.show()
            except Exception as e:
                self.console.print(f"[red]Erro ao gerar histograma: {e}[/red]")

        # ── Etapa 2: Hard Gates (ADF + Hurst) ────────────────────────
        self.console.print("\n[bold yellow]Etapa 2: Hard Gates (Estacionariedade e Random Walk)[/bold yellow]")
        dropped: dict[str, str] = {}
        candidates: list[str] = []

        table_gates = Table(show_header=True, header_style="bold magenta")
        table_gates.add_column("Candidato", style="cyan")
        table_gates.add_column("ADF p-value", justify="right")
        table_gates.add_column("Hurst", justify="right")
        table_gates.add_column("|H - 0.5|", justify="right")
        table_gates.add_column("Status", justify="center")

        hurst_vals = {}
        hurst_colors = []

        for ticker in candidate_tickers:
            if ticker not in returns or len(returns[ticker]) < 10:
                dropped[ticker] = "Dados insuficientes após smoothing"
                table_gates.add_row(ticker, "-", "-", "-", "[red]Dados insuf.[red]")
                continue

            # ADF nos log-retornos
            adf_res = adf(returns[ticker], alpha=alpha)
            # Hurst nos preços suavizados
            h_res = hurst(smoothed[ticker])
            
            p_val = adf_res['p_value']
            h_val = h_res['hurst']
            h_dist = h_res['distance_05']
            
            hurst_vals[ticker] = h_val

            if not adf_res["is_stationary"]:
                dropped[ticker] = (
                    f"Não-estacionário (ADF p={p_val:.4f})"
                )
                table_gates.add_row(ticker, f"[red]{p_val:.4f}[/red]", f"{h_val:.4f}", f"{h_dist:.4f}", "[red]Reprovado (ADF)[/red]")
                hurst_colors.append('red')
                continue

            if h_dist < self.hurst_tolerance:
                dropped[ticker] = (
                    f"Random walk (H={h_val:.4f}, "
                    f"|H−0.5|={h_dist:.4f} < {self.hurst_tolerance})"
                )
                table_gates.add_row(ticker, f"[green]{p_val:.4f}[/green]", f"[red]{h_val:.4f}[/red]", f"[red]{h_dist:.4f}[/red]", "[red]Reprovado (Hurst)[/red]")
                hurst_colors.append('red')
                continue

            candidates.append(ticker)
            table_gates.add_row(ticker, f"[green]{p_val:.4f}[/green]", f"[green]{h_val:.4f}[/green]", f"[green]{h_dist:.4f}[/green]", "[green]Aprovado[/green]")
            hurst_colors.append('blue')

        self.console.print(table_gates)
        
        # Plot Hurst
        if hurst_vals:
            fig_hurst = go.Figure()
            fig_hurst.add_trace(go.Bar(
                x=list(hurst_vals.keys()),
                y=list(hurst_vals.values()),
                marker_color=hurst_colors,
                name="Hurst Exponent"
            ))
            fig_hurst.add_hline(y=0.5, line_dash="dash", line_color="black", annotation_text="Passeio Aleatório (H=0.5)")
            fig_hurst.add_hrect(y0=0.5 - self.hurst_tolerance, y1=0.5 + self.hurst_tolerance, line_width=0, fillcolor="red", opacity=0.2, annotation_text="Tolerância de Rejeição")
            fig_hurst.update_layout(title="Etapa 2: Expoente de Hurst dos Candidatos", template="plotly_white", yaxis_title="Hurst (H)", yaxis_range=[0, 1])
            fig_hurst.show()

        # Early exit: nenhum candidato sobreviveu
        if not candidates:
            self.result = self._empty_result(
                window=window, dropped=dropped
            )
            return self.result

        # ── Etapa 3: JSD (peso global) ───────────────────────────────
        self.console.print("\n[bold yellow]Etapa 3: Jensen-Shannon Divergence (Pesos Globais)[/bold yellow]")
        self.console.print("[italic dim]Contexto: A JSD mede a distância entre a distribuição de retornos de um candidato e a do alvo. Uma menor divergência (maior peso '1 - JSD') garante que a dinâmica de volatilidade daquele candidato é estatisticamente compatível com a do ativo que queremos prever.[/italic dim]")
        jsd_weights = self._compute_jsd_weights(candidates, returns, target_ret)
        
        table_jsd = Table(show_header=True, header_style="bold magenta")
        table_jsd.add_column("Candidato", style="cyan")
        table_jsd.add_column("Peso JSD (1 - JSD)", justify="right")
        
        for ticker, weight in jsd_weights.items():
            table_jsd.add_row(ticker, f"{weight:.4f}")
            
        self.console.print(table_jsd)
        
        fig_jsd = go.Figure(go.Bar(
            x=list(jsd_weights.keys()),
            y=list(jsd_weights.values()),
            marker_color='purple'
        ))
        fig_jsd.update_layout(title="Etapa 3: Pesos JSD (1 - JSD)", template="plotly_white", yaxis_title="Peso", yaxis_range=[0, 1])
        fig_jsd.show()

        # ── Etapa 4: Auto-MI do alvo ─────────────────────────────────
        self.console.print("\n[bold yellow]Etapa 4: Auto-Análise do Alvo (Auto-MI e ACF/PACF)[/bold yellow]")
        self.console.print("[italic dim]Contexto: A Auto-MI extrai a memória natural (embedding dimension) da série alvo. Esse lag endógeno dita até onde o alvo olha para o seu próprio passado, servindo de filtro temporal condicional (y_lags) para o cálculo rigoroso da Transfer Entropy no próximo passo.[/italic dim]")
        auto_mi_lag, auto_mi_profile = self._compute_auto_mi_lag(target_ret, lag_max, knn_k)
        target_acf = acf_pacf(target_ret, lags=lag_max, alpha=alpha)
        
        self.console.print(f"Lag endógeno com maior Auto-MI selecionado (y_lags da TE): [bold green]{auto_mi_lag}[/bold green]")
        
        # Plot Auto-MI and ACF/PACF
        fig_auto = make_subplots(rows=1, cols=3, subplot_titles=("Perfil Auto-MI", "ACF do Alvo", "PACF do Alvo"))
        
        lags_mi = [lag for lag, _ in auto_mi_profile]
        vals_mi = [val for _, val in auto_mi_profile]
        fig_auto.add_trace(go.Scatter(x=lags_mi, y=vals_mi, mode='lines+markers', name="Auto-MI", marker_color='purple'), row=1, col=1)
        
        acf_vals = target_acf["acf"]
        pacf_vals = target_acf["pacf"]
        lags_acf = list(range(len(acf_vals)))
        
        fig_auto.add_trace(go.Bar(x=lags_acf, y=acf_vals, name="ACF", marker_color='blue'), row=1, col=2)
        fig_auto.add_trace(go.Bar(x=lags_acf, y=pacf_vals, name="PACF", marker_color='orange'), row=1, col=3)
        
        # Add confidence intervals
        conf_int = 1.96 / np.sqrt(len(target_ret))
        fig_auto.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=2)
        fig_auto.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=2)
        fig_auto.add_hline(y=conf_int, line_dash="dash", line_color="red", row=1, col=3)
        fig_auto.add_hline(y=-conf_int, line_dash="dash", line_color="red", row=1, col=3)
        
        fig_auto.update_layout(title="Etapa 4: Autocorrelações e Entropia do Ativo Alvo", template="plotly_white")
        fig_auto.show()

        # ── Etapa 5: Score Matrix ────────────────────────────────────
        self.console.print("\n[bold yellow]Etapa 5, 6 e 7: Score Matrix e Normalização (Linear e Não-Linear)[/bold yellow]")
        granger_f_raw, granger_p_raw, mi_raw, te_raw = self._compute_raw_metrics(
            candidates, returns, target_ret,
            lag_max=lag_max, alpha=alpha, knn_k=knn_k,
            auto_mi_lag=auto_mi_lag,
        )

        # ── Etapa 6–7: Normalizar + Score Composto ───────────────────
        score_matrix = self._build_score_matrix(
            candidates, lag_max,
            granger_f_raw, mi_raw, te_raw,
            jsd_weights,
        )
        
        self.console.print("Matriz de Score Composto calculada com sucesso.")
        
        # Plot Score Matrix Heatmap
        fig_score = go.Figure(data=go.Heatmap(
            z=score_matrix.values,
            x=score_matrix.columns,
            y=score_matrix.index,
            colorscale='Viridis',
            text=np.round(score_matrix.values, 4),
            texttemplate="%{text}",
            colorbar=dict(title="Score Composto")
        ))
        fig_score.update_layout(title="Etapa 7: Matriz de Score Composto S(c, l)", template="plotly_white", xaxis_title="Lags", yaxis_title="Candidatos")
        fig_score.show()

        # ── Etapa 8: Lag Consensual ──────────────────────────────────
        self.console.print("\n[bold yellow]Etapa 8: Identificação do Lag Consensual[/bold yellow]")
        lag_consensus = self._find_consensus_lag(score_matrix)
        
        col_sums = score_matrix.sum(axis=0)
        self.console.print(f"Lag consensual que maximiza a pontuação agregada (l*): [bold green]{lag_consensus}[/bold green]")
        
        fig_lag = go.Figure(go.Scatter(
            x=col_sums.index,
            y=col_sums.values,
            mode='lines+markers',
            marker=dict(size=10, color='blue'),
            line=dict(width=3)
        ))
        # Destacar o lag ótimo
        fig_lag.add_vline(x=f"lag_{lag_consensus}", line_width=2, line_dash="dash", line_color="green")
        fig_lag.update_layout(title="Etapa 8: Soma dos Scores Compostos por Lag", template="plotly_white", xaxis_title="Lags", yaxis_title="Soma de Scores")
        fig_lag.show()

        # ── Etapa 9: Top-K ───────────────────────────────────────────
        self.console.print(f"\n[bold yellow]Etapa 9: Seleção Top-{self.top_k} no Lag Consensual ({lag_consensus})[/bold yellow]")
        selected, newly_dropped = self._select_top_k(
            candidates, score_matrix, lag_consensus,
            returns, granger_f_raw, granger_p_raw, mi_raw, te_raw,
        )
        dropped.update(newly_dropped)
        
        table_sel = Table(show_header=True, header_style="bold green")
        table_sel.add_column("Candidato Selecionado", style="cyan")
        table_sel.add_column("Tipo Relação", justify="center")
        table_sel.add_column("Score", justify="right")
        table_sel.add_column("Granger (p)", justify="right")
        table_sel.add_column("Granger (F)", justify="right")
        table_sel.add_column("MI", justify="right")
        table_sel.add_column("TE", justify="right")
        
        for sel in selected:
            sc = score_matrix.loc[sel.ticker, f"lag_{lag_consensus}"]
            g_f = granger_f_raw[sel.ticker].get(lag_consensus, 0.0)
            table_sel.add_row(
                sel.ticker, 
                sel.relation_type.capitalize(), 
                f"{sc:.4f}",
                f"{sel.granger_pvalue:.4f}",
                f"{g_f:.4f}",
                f"{sel.mi_value:.4f}",
                f"{sel.te_value:.4f}" if sel.te_value else "-"
            )
            
        self.console.print(table_sel)

        # ── Etapa 10: Enriquecimento pós-seleção ─────────────────────
        self.console.print("\n[bold yellow]Etapa 10 e 11: Enriquecimento (STL/Johansen) e Descritivas[/bold yellow]")
        self.console.print("[italic dim]Nota de Transparência: A Cointegração de Johansen e a Decomposição STL são calculadas aqui puramente como contexto auxiliar (enriquecimento) para validação visual humana. Elas não são injetadas como features no classificador preditivo (Etapa 2) para evitar multicolinearidade e vazamento de dados futuros (look-ahead bias da STL).[/italic dim]")
        selected_tickers = [s.ticker for s in selected]

        stl_components = self._compute_stl(selected_tickers, smoothed)
        cointegration = self._compute_johansen(
            selected_tickers, smoothed, target_ticker
        )
        
        if cointegration:
            table_coint = Table(show_header=True, header_style="bold blue", title="Cointegração de Johansen")
            table_coint.add_column("Traço de Coint.", justify="center")
            table_coint.add_row("[green]Processado com Sucesso[/green]")
            self.console.print(table_coint)
            
        if stl_components:
            fig_stl = make_subplots(rows=max(1, len(selected_tickers)), cols=1, shared_xaxes=True, subplot_titles=[f"Decomposição STL: {t}" for t in selected_tickers])
            for i, ticker in enumerate(selected_tickers):
                if "trend" in stl_components[ticker]:
                    fig_stl.add_trace(go.Scatter(x=stl_components[ticker]["trend"].index, y=stl_components[ticker]["trend"].values, name=f"{ticker} Tendência"), row=i+1, col=1)
                    fig_stl.add_trace(go.Scatter(x=stl_components[ticker]["seasonal"].index, y=stl_components[ticker]["seasonal"].values, name=f"{ticker} Sazonalidade"), row=i+1, col=1)
            fig_stl.update_layout(height=300*max(1, len(selected_tickers)), title="Etapa 10: Componentes STL Extraídos", template="plotly_white")
            fig_stl.show()

        # ── Etapa 11: Descritivas ────────────────────────────────────
        descriptives = self._compute_descriptives(
            selected_tickers, returns, target_ret,
            target_acf, lag_max, alpha,
        )

        self.result = PipelineResult(
            selected=selected,
            lag_consensus=lag_consensus,
            score_matrix=score_matrix,
            jsd_weights=jsd_weights,
            auto_mi_lag=auto_mi_lag,
            smoothing_window=window,
            smoothing_method=self.config.smoothing_method,
            stl_components=stl_components,
            cointegration=cointegration,
            descriptives=descriptives,
            dropped=dropped,
        )
        return self.result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self, *, window: int, dropped: dict[str, str]
    ) -> PipelineResult:
        return PipelineResult(
            selected=[],
            lag_consensus=0,
            score_matrix=pd.DataFrame(),
            jsd_weights={},
            auto_mi_lag=1,
            smoothing_window=window,
            smoothing_method=self.config.smoothing_method,
            stl_components={},
            cointegration={},
            descriptives={},
            dropped=dropped,
        )

    # -- Etapa 3 -------------------------------------------------------

    @staticmethod
    def _compute_jsd_weights(
        candidates: list[str],
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
    ) -> dict[str, float]:
        """``jsd_weight(c) = 1 - JSD(c, alvo)``."""
        weights: dict[str, float] = {}
        for ticker in candidates:
            jsd_df = jensen_shannon_divergence(
                [returns[ticker], target_ret]
            )
            jsd_val = float(jsd_df.iloc[0, 1])
            weights[ticker] = max(0.0, 1.0 - jsd_val)
        return weights

    # -- Etapa 4 -------------------------------------------------------

    @staticmethod
    def _compute_auto_mi_lag(
        target_ret: pd.Series, lag_max: int, knn_k: int
    ) -> tuple[int, list]:
        """Lag com maior auto-MI do alvo → ``y_lags`` para TE."""
        results = mutual_information_lags(
            target_ret, target_ret,
            max_lag=min(lag_max, 10), k=knn_k,
        )
        if not results:
            return 1, []
        max_lag = max(results, key=lambda pair: pair[1])[0]
        return max_lag, results

    # -- Etapa 5 -------------------------------------------------------

    @staticmethod
    def _compute_raw_metrics(
        candidates: list[str],
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
        *,
        lag_max: int,
        alpha: float,
        knn_k: int,
        auto_mi_lag: int,
    ) -> tuple[
        dict[str, dict[int, float]],  # granger_f
        dict[str, dict[int, float]],  # granger_p
        dict[str, dict[int, float]],  # mi
        dict[str, dict[int, float]],  # te
    ]:
        """
        Computa métricas brutas para cada (candidato, lag):
        Granger F-stat (gated by p), MI em bits, TE em bits.
        """
        granger_f: dict[str, dict[int, float]] = {}
        granger_p: dict[str, dict[int, float]] = {}
        mi_vals: dict[str, dict[int, float]] = {}
        te_vals: dict[str, dict[int, float]] = {}

        lags = list(range(1, lag_max + 1))

        for ticker in candidates:
            c_ret = returns[ticker]

            # ── 5a: Granger (F-stat gated by p-value) ────────────────
            g_result = granger_causality(
                target_ret, [c_ret], lag_max=lag_max, alpha=alpha
            )
            driver_key = next(iter(g_result))
            g_data = g_result[driver_key]

            granger_f[ticker] = {}
            granger_p[ticker] = {}
            for lag in lags:
                p = g_data["all_p_values"].get(lag, 1.0)
                f = g_data["all_f_stats"].get(lag, 0.0)
                granger_p[ticker][lag] = p
                granger_f[ticker][lag] = f if p < alpha else 0.0

            # ── 5b: MI em bits ───────────────────────────────────────
            mi_result = mutual_information_lags(
                target_ret, c_ret, max_lag=lag_max, k=knn_k
            )
            mi_vals[ticker] = {lag: mi for lag, mi in mi_result}

            # ── 5c: TE em bits (só onde MI > média) ──────────────────
            mi_values = [mi for _, mi in mi_result]
            mi_mean = float(np.mean(mi_values)) if mi_values else 0.0

            te_vals[ticker] = {}
            for lag in lags:
                if mi_vals[ticker].get(lag, 0.0) > mi_mean:
                    te_val = transfer_entropy(
                        x=c_ret, y=target_ret,
                        lag=lag, y_lags=auto_mi_lag,
                    )
                    te_vals[ticker][lag] = te_val
                else:
                    te_vals[ticker][lag] = 0.0

        return granger_f, granger_p, mi_vals, te_vals

    # -- Etapas 6–7 ----------------------------------------------------

    def _build_score_matrix(
        self,
        candidates: list[str],
        lag_max: int,
        granger_f: dict[str, dict[int, float]],
        mi_vals: dict[str, dict[int, float]],
        te_vals: dict[str, dict[int, float]],
        jsd_weights: dict[str, float],
    ) -> pd.DataFrame:
        """
        Normaliza (min-max) e combina as três métricas num score
        composto ponderado pelo JSD weight.
        """
        lags = list(range(1, lag_max + 1))

        # Flatten para encontrar min/max globais
        all_g = [granger_f[t][l] for t in candidates for l in lags]
        all_mi = [mi_vals[t].get(l, 0.0) for t in candidates for l in lags]
        all_te = [te_vals[t].get(l, 0.0) for t in candidates for l in lags]

        g_min, g_max = (min(all_g), max(all_g)) if all_g else (0.0, 0.0)
        mi_min, mi_max = (min(all_mi), max(all_mi)) if all_mi else (0.0, 0.0)
        te_min, te_max = (min(all_te), max(all_te)) if all_te else (0.0, 0.0)

        def _norm(val: float, vmin: float, vmax: float) -> float:
            return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.0

        data: dict[str, dict[str, float]] = {}
        for ticker in candidates:
            data[ticker] = {}
            jw = jsd_weights.get(ticker, 1.0)
            for lag in lags:
                g_n = _norm(granger_f[ticker][lag], g_min, g_max)
                mi_n = _norm(mi_vals[ticker].get(lag, 0.0), mi_min, mi_max)
                te_n = _norm(te_vals[ticker].get(lag, 0.0), te_min, te_max)

                raw = (
                    self.w_granger * g_n
                    + self.w_mi * mi_n
                    + self.w_te * te_n
                )
                data[ticker][f"lag_{lag}"] = jw * raw

        return pd.DataFrame.from_dict(data, orient="index")

    # -- Etapa 8 -------------------------------------------------------

    @staticmethod
    def _find_consensus_lag(score_matrix: pd.DataFrame) -> int:
        """``ℓ* = argmax_l  Σ_c  S(c, l)``.  Empate → menor lag."""
        if score_matrix.empty:
            return 1
        col_sums = score_matrix.sum(axis=0)
        max_score = col_sums.max()
        # Colunas com score máximo (para desempate pelo menor lag)
        ties = col_sums[col_sums == max_score].index.tolist()
        # Extrair lag numérico e retornar o menor
        tie_lags = sorted(int(c.split("_")[1]) for c in ties)
        return tie_lags[0]

    # -- Etapa 9 -------------------------------------------------------

    def _select_top_k(
        self,
        candidates: list[str],
        score_matrix: pd.DataFrame,
        lag_consensus: int,
        returns: dict[str, pd.Series],
        granger_f: dict[str, dict[int, float]],
        granger_p: dict[str, dict[int, float]],
        mi_vals: dict[str, dict[int, float]],
        te_vals: dict[str, dict[int, float]],
    ) -> tuple[list[SelectedCandidate], dict[str, str]]:
        """Seleciona top-K candidatos no lag consensual."""
        col = f"lag_{lag_consensus}"
        scores = score_matrix[col].sort_values(ascending=False)

        selected: list[SelectedCandidate] = []
        newly_dropped: dict[str, str] = {}

        for ticker, score in scores.items():
            if len(selected) >= self.top_k:
                newly_dropped[str(ticker)] = (
                    f"Excedeu top-{self.top_k} (score={score:.4f})"
                )
                continue

            if score < self.score_threshold:
                newly_dropped[str(ticker)] = (
                    f"Score insuficiente no lag {lag_consensus} "
                    f"({score:.4f} < {self.score_threshold})"
                )
                continue

            ticker_str = str(ticker)

            # Determinar tipo de relação dominante
            g_val = granger_f[ticker_str].get(lag_consensus, 0.0)
            mi_val = mi_vals[ticker_str].get(lag_consensus, 0.0)
            te_val = te_vals[ticker_str].get(lag_consensus, 0.0)

            relation_type = (
                "linear"
                if g_val > 0.0
                else "non-linear"
            )

            selected.append(
                SelectedCandidate(
                    ticker=ticker_str,
                    component="retorno",
                    series=returns[ticker_str],
                    lag_tau=lag_consensus,
                    relation_type=relation_type,
                    mi_value=mi_val,
                    te_value=te_val if te_val > 0.0 else None,
                    granger_pvalue=granger_p[ticker_str].get(
                        lag_consensus, 1.0
                    ),
                )
            )

        return selected, newly_dropped

    # -- Etapa 10 ------------------------------------------------------

    def _compute_stl(
        self,
        selected_tickers: list[str],
        smoothed: dict[str, pd.Series],
    ) -> dict[str, dict[str, pd.Series]]:
        """STL decomposition nos preços suavizados dos selecionados."""
        stl_results: dict[str, dict[str, pd.Series]] = {}
        for ticker in selected_tickers:
            try:
                decomp = stl_decomposition(
                    smoothed[ticker], period=self.config.stl_period
                )
                stl_results[ticker] = {
                    "trend": decomp["trend"],
                    "seasonal": decomp["seasonal"],
                    "residual": decomp["residual"],
                }
            except Exception:
                stl_results[ticker] = {}
        return stl_results

    @staticmethod
    def _compute_johansen(
        selected_tickers: list[str],
        smoothed: dict[str, pd.Series],
        target_ticker: str,
    ) -> dict:
        """Johansen cointegration nos preços suavizados."""
        if not selected_tickers:
            return {}

        series_list = [smoothed[target_ticker]]
        for ticker in selected_tickers:
            series_list.append(smoothed[ticker])

        try:
            return johansen_cointegration(series_list, lag_order=1)
        except Exception:
            return {}

    # -- Etapa 11 ------------------------------------------------------

    @staticmethod
    def _compute_descriptives(
        selected_tickers: list[str],
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
        target_acf: dict,
        lag_max: int,
        alpha: float,
    ) -> dict[str, Any]:
        """Estatísticas descritivas dos candidatos selecionados."""
        descriptives: dict[str, Any] = {
            "target_acf_pacf": target_acf,
        }
        for ticker in selected_tickers:
            c_ret = returns[ticker]
            descriptives[ticker] = {
                "stats": stats(c_ret),
                "pearson": pearson(c_ret, target_ret),
                "spearman": spearman(c_ret, target_ret),
                "acf_pacf": acf_pacf(c_ret, lags=lag_max, alpha=alpha),
            }
        return descriptives
