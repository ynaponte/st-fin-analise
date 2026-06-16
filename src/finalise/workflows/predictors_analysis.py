"""
workflows.predictors_analysis
-----------------------------
Pipeline de seleção de preditores baseado em Transfer Entropy e Surrogates.

Dado um alvo e N séries candidatas (preços de fechamento), executa:

    0. Smoothing nos preços brutos
    1. Log-retornos (k=1)
    2. Hard gates   — ADF (estacionariedade) e Hurst (random walk)
    3. Auto-MI      — embedding dimension do alvo para a TE (y_lags)
    4. Geração de Surrogates — Deslocamento circular para cada candidato
    5. TE Score Matrix — TE testada contra Surrogates (candidato × lag)
    6. Lag Endógeno — argmax da soma das TEs significativas
    7. Top-K        — seleção final no lag endógeno
    8. Enriquecimento — STL + Johansen (pós-seleção) e Descritivas (incluindo Granger)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
)
from finalise.analysis.base import granger_causality, johansen_cointegration
from finalise.analysis.correlations import pearson, spearman, acf_pacf

from finalise.workflows.te_surrogates import (
    generate_circular_surrogates,
    test_te_significance
)


# ---------------------------------------------------------------------------
# SelectedCandidate
# ---------------------------------------------------------------------------
@dataclass
class SelectedCandidate:
    """Candidato selecionado pelo pipeline focado em TE."""

    ticker: str
    component: str
    series: pd.Series
    lag_tau: int
    relation_type: str          # "non-linear" (historically used)
    mi_value: float             # maintained for compatibility
    te_value: Optional[float]
    granger_pvalue: float       # evaluated at lag_tau pós-seleção


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
    key = method.upper()
    if key not in _SMOOTHING_POLICIES:
        supported = ", ".join(sorted({k for k in _SMOOTHING_POLICIES}))
        raise ValueError(f"Método de suavização '{method}' não suportado. Opções: {supported}")
    return _SMOOTHING_POLICIES[key]


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Resultado completo do pipeline de seleção."""

    selected: List[SelectedCandidate]
    lag_consensus: int
    score_matrix: pd.DataFrame
    auto_mi_lag: int
    smoothing_window: int
    smoothing_method: str
    stl_components: Dict[str, Dict[str, pd.Series]]
    cointegration: Dict[str, Any]
    descriptives: Dict[str, Any]
    dropped: Dict[str, str]
    trend_transfer_entropy: Dict[str, float]
    residual_transfer_entropy: Dict[str, float]


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------
class PredictorsAnalysis:
    def __init__(
        self,
        prices_dict: dict[str, pd.Series],
        config: Config,
        *,
        smoothing_window: Optional[int] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        hurst_tolerance: float = 0.05,
        n_surrogates: int = 100,
        # Mantendo kwargs legados apenas para não quebrar código que instancia a classe
        w_granger: float = 0.0,
        w_mi: float = 0.0,
        w_te: float = 1.0,
        te_permutations: Optional[int] = None,
    ):
        self.prices_dict = prices_dict
        self.config = config
        self.smoothing_window = smoothing_window
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.hurst_tolerance = hurst_tolerance
        
        # O sistema agora usa surrogates puramente
        self.n_surrogates = te_permutations if te_permutations is not None else n_surrogates
        
        self.result: Optional[PipelineResult] = None
        self.console = Console()

    def run(self) -> PipelineResult:
        self.console.print(Panel("[bold cyan]Iniciando Pipeline PredictorsAnalysis (TE Surrogates v5)[/bold cyan]", border_style="cyan"))

        target_ticker = self.config.target_ticker
        candidate_tickers = [t for t in self.config.predictor_tickers if t != target_ticker]
        alpha = self.config.alpha
        lag_max = self.config.lag_max
        knn_k = self.config.knn_k

        # ── Etapa 0: Smoothing ───────────────────────
        self.console.print("\n[bold yellow]Etapa 0: Smoothing (Filtro Passa-Baixa)[/bold yellow]")
        policy = _resolve_policy(self.config.smoothing_method)
        window = self.smoothing_window
        if window is None:
            window = find_optimal_window(self.prices_dict[target_ticker])
            self.console.print(f"Janela ótima via alvo: [bold green]{window}[/bold green]")
        else:
            self.console.print(f"Janela definida pelo usuário: [bold green]{window}[/bold green]")

        smoothed: dict[str, pd.Series] = {}
        for ticker, prices in self.prices_dict.items():
            smoothed[ticker] = apply_smoothing(prices, window, policy)

        # ── Etapa 1: Log-retornos ──────────────────────────────
        self.console.print("\n[bold yellow]Etapa 1: Transformação para Log-Retornos[/bold yellow]")
        returns: dict[str, pd.Series] = {}
        for ticker, prices_s in smoothed.items():
            ret = log_returns(prices_s, k=1)
            if not ret.empty:
                returns[ticker] = ret

        if target_ticker not in returns:
            raise ValueError(f"Série alvo '{target_ticker}' vazia após smoothing + retornos.")
        target_ret = returns[target_ticker]

        # ── Etapa 2: Preparação de Candidatos (Sem Hard Gates) ───────
        self.console.print("\n[bold yellow]Etapa 2: Preparação dos Candidatos[/bold yellow]")
        dropped: dict[str, str] = {}
        candidates: list[str] = []

        for ticker in candidate_tickers:
            if ticker not in returns or len(returns[ticker]) < 10:
                dropped[ticker] = "Dados insuficientes"
                continue
            candidates.append(ticker)

        self.console.print(f"Candidatos em análise: {len(candidates)}")
        
        if not candidates:
            return self._empty_result(window=window, dropped=dropped)

        # ── Etapa 3: Auto-MI do alvo (y_lags) ─────────────────────────
        self.console.print("\n[bold yellow]Etapa 3: Auto-MI do Alvo (y_lags)[/bold yellow]")
        auto_mi_lag, _ = self._compute_auto_mi_lag(target_ret, lag_max, knn_k)
        self.console.print(f"Embedding dimension do alvo (y_lags): [bold green]{auto_mi_lag}[/bold green]")

        # ── Etapa 4: Geração de Surrogates ────────────────────────────
        self.console.print("\n[bold yellow]Etapa 4: Geração Única de Surrogates Circulares[/bold yellow]")
        surrogates_dict = {}
        for ticker in candidates:
            surrogates_dict[ticker] = generate_circular_surrogates(
                returns[ticker], n_surrogates=self.n_surrogates, seed=42
            )
        self.console.print(f"Gerados {self.n_surrogates} surrogates para {len(candidates)} candidatos.")

        # ── Etapa 5: Matriz de TE Significativa ────────────────────────
        self.console.print("\n[bold yellow]Etapa 5: Matriz de Transfer Entropy Significativa[/bold yellow]")
        te_matrix_data = {}
        p_value_data = {}
        
        for ticker in candidates:
            c_ret = returns[ticker]
            c_surrs = surrogates_dict[ticker]
            te_matrix_data[ticker] = {}
            p_value_data[ticker] = {}
            
            for lag in range(1, lag_max + 1):
                te_obs, p_val = test_te_significance(
                    c_ret, target_ret, c_surrs, lag, y_lags=auto_mi_lag, n_jobs=-1
                )
                p_value_data[ticker][f"lag_{lag}"] = p_val
                # Se não for significativo, vira zero
                if p_val >= alpha:
                    te_matrix_data[ticker][f"lag_{lag}"] = 0.0
                else:
                    te_matrix_data[ticker][f"lag_{lag}"] = te_obs

        score_matrix = pd.DataFrame.from_dict(te_matrix_data, orient="index")
        
        # Plot Heatmap
        try:
            fig_score = go.Figure(data=go.Heatmap(
                z=score_matrix.values,
                x=score_matrix.columns,
                y=score_matrix.index,
                colorscale='Viridis',
                text=np.round(score_matrix.values, 4),
                texttemplate="%{text}",
                colorbar=dict(title="TE Significativa")
            ))
            fig_score.update_layout(title="Etapa 5: Matriz de TE Significativa", template="plotly_white")
            fig_score.show()
        except Exception:
            pass

        # ── Etapa 6: Lag Endógeno ──────────────────────────────────
        self.console.print("\n[bold yellow]Etapa 6: Identificação do Lag Endógeno[/bold yellow]")
        col_sums = score_matrix.sum(axis=0)
        
        if col_sums.max() == 0:
            self.console.print("[bold red]Nenhum candidato apresentou TE significativa em nenhum lag![/bold red]")
            return self._empty_result(window=window, dropped=dropped)
            
        lag_consensus = int(col_sums.idxmax().split("_")[1])
        self.console.print(f"Lag endógeno (maior soma de TE): [bold green]{lag_consensus}[/bold green] com score {col_sums.max():.4f}")

        # ── Etapa 7: Top-K ───────────────────────────────────────────
        self.console.print(f"\n[bold yellow]Etapa 7: Seleção Top-{self.top_k} no Lag {lag_consensus}[/bold yellow]")
        col_name = f"lag_{lag_consensus}"
        scores = score_matrix[col_name].sort_values(ascending=False)
        
        selected: list[SelectedCandidate] = []
        table_all = Table(show_header=True, header_style="bold magenta", title=f"Status de Todos os Candidatos (Lag {lag_consensus})")
        table_all.add_column("Candidato", style="cyan")
        table_all.add_column("TE Observada", justify="right")
        table_all.add_column("TE P-Valor", justify="right")
        table_all.add_column("Status Final", justify="left")
        
        for ticker, score in scores.items():
            ticker_str = str(ticker)
            pval = p_value_data[ticker_str][col_name]
            
            if score <= 0.0 or score < self.score_threshold:
                dropped[ticker_str] = "Não causal (TE não significativa)"
                table_all.add_row(ticker_str, f"{score:.4f}", f"{pval:.4f}", "[red]Dropado (Não Causal / P-val alto)[/red]")
                continue
            if len(selected) >= self.top_k:
                dropped[ticker_str] = f"Fora do Top-{self.top_k}"
                table_all.add_row(ticker_str, f"{score:.4f}", f"{pval:.4f}", "[yellow]Dropado (Rank Baixo)[/yellow]")
                continue
                
            # Compute MI just for compatibility in output
            mi_vals = mutual_information_lags(target_ret, returns[ticker_str], max_lag=lag_consensus, k=knn_k)
            mi_at_lag = next((mi for l, mi in mi_vals if l == lag_consensus), 0.0)
            
            # Compute Granger just for compatibility in output (Feature Engineering)
            g_res = granger_causality(target_ret, [returns[ticker_str]], lag_max=lag_consensus, alpha=alpha)
            g_pval = 1.0
            if g_res:
                dk = next(iter(g_res))
                g_pval = g_res[dk]["all_p_values"].get(lag_consensus, 1.0)
                
            selected.append(SelectedCandidate(
                ticker=ticker_str,
                component="retorno",
                series=returns[ticker_str],
                lag_tau=lag_consensus,
                relation_type="non-linear",
                mi_value=mi_at_lag,
                te_value=score,
                granger_pvalue=g_pval
            ))
            table_all.add_row(ticker_str, f"{score:.4f}", f"{pval:.4f}", "[green]Selecionado[/green]")
            
        self.console.print(table_all)

        # ── Etapa 8: Enriquecimento (STL/Johansen/Granger) ───────────
        self.console.print("\n[bold yellow]Etapa 8: Enriquecimento e Descritivas Pós-Seleção (STL e Cointegração)[/bold yellow]")
        selected_tickers = [s.ticker for s in selected]
        
        stl_components = self._compute_stl([target_ticker] + selected_tickers, smoothed)
        
        # Teste ADF e Johansen Condicional para Tendências
        coint_trend = {}
        coint_resid = {}
        
        table_stl = Table(show_header=True, header_style="bold blue", title="Propriedades STL e Cointegração")
        table_stl.add_column("Componente", style="cyan")
        table_stl.add_column("Série", justify="left")
        table_stl.add_column("ADF p-val", justify="right")
        table_stl.add_column("Estacionário?", justify="center")
        
        trend_series = []
        resid_series = []
        all_trends_nonstat = True
        all_resids_nonstat = True
        
        for t in [target_ticker] + selected_tickers:
            comp = stl_components.get(t, {})
            tr = comp.get("trend")
            re = comp.get("residual")
            
            if tr is not None:
                ap = adf(tr, alpha=alpha)['p_value']
                is_stat = ap < alpha
                if is_stat: all_trends_nonstat = False
                table_stl.add_row("Tendência", t, f"{ap:.4f}", "[green]Sim[/green]" if is_stat else "[red]Não[/red]")
                trend_series.append(tr)
                
            if re is not None:
                ap = adf(re, alpha=alpha)['p_value']
                is_stat = ap < alpha
                if is_stat: all_resids_nonstat = False
                table_stl.add_row("Resíduo", t, f"{ap:.4f}", "[green]Sim[/green]" if is_stat else "[red]Não[/red]")
                resid_series.append(re)
                
        self.console.print(table_stl)
        
        if all_trends_nonstat and len(trend_series) > 1:
            try:
                coint_trend = johansen_cointegration(trend_series, lag_order=1)
                rank = coint_trend.get("rank", 0)
                self.console.print(f"[bold green]=> Tendências Cointegradas![/bold green] Rank: {rank}" if coint_trend.get("is_cointegrated") else "[yellow]=> Tendências NÃO Cointegradas.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Erro Johansen (Tendência): {e}[/red]")
        else:
            self.console.print("[dim]=> Cointegração de Tendências pulada (Séries possuem componentes estacionários)[/dim]")

        if all_resids_nonstat and len(resid_series) > 1:
            try:
                coint_resid = johansen_cointegration(resid_series, lag_order=1)
                rank = coint_resid.get("rank", 0)
                self.console.print(f"[bold green]=> Resíduos Cointegradas![/bold green] Rank: {rank}" if coint_resid.get("is_cointegrated") else "[yellow]=> Resíduos NÃO Cointegradas.[/yellow]")
            except Exception as e:
                self.console.print(f"[red]Erro Johansen (Resíduo): {e}[/red]")
        else:
            self.console.print("[dim]=> Cointegração de Resíduos pulada (Séries possuem componentes estacionários)[/dim]")
        
        trend_te = {}
        residual_te = {}
        target_trend = stl_components.get(target_ticker, {}).get("trend")
        target_residual = stl_components.get(target_ticker, {}).get("residual")
        
        for ticker in selected_tickers:
            pred_trend = stl_components.get(ticker, {}).get("trend")
            pred_residual = stl_components.get(ticker, {}).get("residual")
            
            if pred_trend is not None and target_trend is not None:
                try:
                    trend_te[ticker] = transfer_entropy(pred_trend, target_trend, lag=lag_consensus, y_lags=auto_mi_lag)
                except Exception:
                    trend_te[ticker] = 0.0
                    
            if pred_residual is not None and target_residual is not None:
                try:
                    residual_te[ticker] = transfer_entropy(pred_residual, target_residual, lag=lag_consensus, y_lags=auto_mi_lag)
                except Exception:
                    residual_te[ticker] = 0.0

        if coint_trend:
            rank = coint_trend.get("rank", 0)
            is_coint = coint_trend.get("is_cointegrated", False)
            table_coint = Table(show_header=True, header_style="bold blue", title="Cointegração de Johansen (Tendências)")
            table_coint.add_column("Rank (r)", justify="left")
            table_coint.add_column("Co-integrado?", justify="left")
            table_coint.add_row(str(rank), "[bold green]Sim[/bold green]" if is_coint else "[red]Não[/red]")
            self.console.print(table_coint)

        descriptives = self._compute_descriptives(selected_tickers, returns, target_ret, lag_max, alpha)

        self.result = PipelineResult(
            selected=selected,
            lag_consensus=lag_consensus,
            score_matrix=score_matrix,
            auto_mi_lag=auto_mi_lag,
            smoothing_window=window,
            smoothing_method=self.config.smoothing_method,
            stl_components=stl_components,
            cointegration=coint_trend,
            descriptives=descriptives,
            dropped=dropped,
            trend_transfer_entropy=trend_te,
            residual_transfer_entropy=residual_te,
        )
        return self.result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _empty_result(self, *, window: int, dropped: dict[str, str]) -> PipelineResult:
        return PipelineResult(
            selected=[], lag_consensus=0, score_matrix=pd.DataFrame(),
            auto_mi_lag=1, smoothing_window=window, smoothing_method=self.config.smoothing_method,
            stl_components={}, cointegration={}, descriptives={}, dropped=dropped,
            trend_transfer_entropy={}, residual_transfer_entropy={}
        )

    @staticmethod
    def _compute_auto_mi_lag(target_ret: pd.Series, lag_max: int, knn_k: int) -> tuple[int, list]:
        results = mutual_information_lags(target_ret, target_ret, max_lag=min(lag_max, 10), k=knn_k)
        if not results: return 1, []
        max_lag = max(results, key=lambda pair: pair[1])[0]
        return max_lag, results

    def _compute_stl(self, selected_tickers: list[str], smoothed: dict[str, pd.Series]) -> dict[str, dict[str, pd.Series]]:
        stl_results = {}
        for ticker in selected_tickers:
            try:
                decomp = stl_decomposition(smoothed[ticker], period=self.config.stl_period)
                stl_results[ticker] = {"trend": decomp["trend"], "seasonal": decomp["seasonal"], "residual": decomp["residual"]}
            except Exception:
                stl_results[ticker] = {}
        return stl_results

    @staticmethod
    def _compute_johansen_trends(selected_tickers: list[str], stl_components: dict[str, dict[str, pd.Series]], target_ticker: str) -> dict:
        if not selected_tickers or target_ticker not in stl_components or "trend" not in stl_components[target_ticker]:
            return {}
        series_list = [stl_components[target_ticker]["trend"]]
        for ticker in selected_tickers:
            if ticker in stl_components and "trend" in stl_components[ticker]:
                series_list.append(stl_components[ticker]["trend"])
            else:
                return {}
        try:
            return johansen_cointegration(series_list, lag_order=1)
        except Exception:
            return {}

    @staticmethod
    def _compute_descriptives(selected_tickers: list[str], returns: dict[str, pd.Series], target_ret: pd.Series, lag_max: int, alpha: float) -> dict[str, Any]:
        target_acf = acf_pacf(target_ret, lags=lag_max, alpha=alpha)
        descriptives = {"target_acf_pacf": target_acf}
        for ticker in selected_tickers:
            c_ret = returns[ticker]
            g_res = granger_causality(target_ret, [c_ret], lag_max=lag_max, alpha=alpha)
            descriptives[ticker] = {
                "stats": stats(c_ret),
                "pearson": pearson(c_ret, target_ret),
                "spearman": spearman(c_ret, target_ret),
                "acf_pacf": acf_pacf(c_ret, lags=lag_max, alpha=alpha),
                "granger_causality": g_res
            }
        return descriptives
