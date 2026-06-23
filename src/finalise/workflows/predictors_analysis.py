"""
workflows.predictors_analysis
-----------------------------
Pipeline de seleção de preditores baseado em Transfer Entropy e Surrogates.

Dado um alvo e N séries candidatas (preços de fechamento), executa:

    0. Smoothing nos preços brutos
    1. Log-retornos (k=1)
    2. Preparação de candidatos (filtro de dados insuficientes)
    3. Auto-MI      — embedding dimension do alvo para a TE (y_lags)
    4. Geração de Surrogates — Deslocamento circular para cada candidato
    5. TE Score Matrix — TE testada contra Surrogates (candidato × lag)
    6. Lag Endógeno — argmax da soma das TEs significativas
    7. Top-K        — seleção final no lag endógeno
    8. Enriquecimento — STL + Johansen (pós-seleção) e Descritivas (incluindo Granger)
    9. Feature Engineering — Construção e validação de features derivadas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import plotly.graph_objects as go
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
    generated_features: pd.DataFrame


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        self.console.print(Panel(
            "[bold cyan]Iniciando Pipeline PredictorsAnalysis (TE Surrogates v5)[/bold cyan]",
            border_style="cyan",
        ))

        target_ticker = self.config.target_ticker
        alpha = self.config.alpha
        lag_max = self.config.lag_max
        knn_k = self.config.knn_k
        candidate_tickers = [t for t in self.config.predictor_tickers if t != target_ticker]

        # ── Etapa 0: Smoothing ────────────────────────────────────────────
        smoothed, window = self._step0_smoothing()

        # ── Etapa 1: Log-retornos ─────────────────────────────────────────
        returns, target_ret = self._step1_log_returns(smoothed, target_ticker)

        # ── Etapa 2: Preparação de candidatos ─────────────────────────────
        candidates, dropped = self._step2_prepare_candidates(
            candidate_tickers, returns
        )
        if not candidates:
            return self._empty_result(window=window, dropped=dropped)

        # ── Etapa 3: Auto-MI do alvo ──────────────────────────────────────
        auto_mi_lag = self._step3_auto_mi(target_ret, lag_max, knn_k)

        # ── Etapa 4: Geração de surrogates ────────────────────────────────
        surrogates_dict = self._step4_surrogates(candidates, returns)

        # ── Etapa 5: Matriz de TE significativa ───────────────────────────
        score_matrix, p_value_data, granger_cache = self._step5_te_matrix(
            candidates, returns, target_ret, surrogates_dict,
            auto_mi_lag, lag_max, alpha,
        )

        # ── Etapa 6: Lag endógeno ─────────────────────────────────────────
        lag_consensus = self._step6_lag_consensus(score_matrix)
        if lag_consensus is None:
            return self._empty_result(window=window, dropped=dropped)

        # ── Etapa 7: Top-K ────────────────────────────────────────────────
        selected, dropped = self._step7_top_k(
            score_matrix, p_value_data, granger_cache, returns,
            target_ret, lag_consensus, auto_mi_lag, alpha, knn_k, dropped,
        )

        # ── Etapa 8: Enriquecimento ──────────────────────────────────────
        selected_tickers = [s.ticker for s in selected]
        stl_components, coint_trend, coint_resid, trend_te, residual_te = (
            self._step8_enrichment(
                selected_tickers, target_ticker, smoothed, returns,
                target_ret, lag_consensus, auto_mi_lag, alpha,
            )
        )

        descriptives = self._compute_descriptives(
            selected_tickers, returns, target_ret, lag_max, alpha
        )

        # ── Etapa 9: Features e validação por surrogates ──────────────────
        scores = score_matrix[f"lag_{lag_consensus}"].sort_values(ascending=False)
        generated_features = self._step9_features_and_surrogates(
            coint_trend, coint_resid, scores, returns, window,
            target_ret, lag_consensus, auto_mi_lag, alpha,
        )

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
            generated_features=generated_features,
        )
        return self.result

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def _step0_smoothing(self) -> tuple[dict[str, pd.Series], int]:
        """Etapa 0: aplica smoothing em todas as séries."""
        self.console.print("\n[bold yellow]Etapa 0: Smoothing (Filtro Passa-Baixa)[/bold yellow]")
        policy = _resolve_policy(self.config.smoothing_method)
        window = self.smoothing_window
        target_ticker = self.config.target_ticker

        if window is None:
            window = find_optimal_window(self.prices_dict[target_ticker])
            self.console.print(f"Janela ótima via alvo: [bold green]{window}[/bold green]")
        else:
            self.console.print(f"Janela definida pelo usuário: [bold green]{window}[/bold green]")

        smoothed: dict[str, pd.Series] = {}
        for ticker, prices in self.prices_dict.items():
            smoothed[ticker] = apply_smoothing(prices, window, policy)

        return smoothed, window

    def _step1_log_returns(
        self, smoothed: dict[str, pd.Series], target_ticker: str
    ) -> tuple[dict[str, pd.Series], pd.Series]:
        """Etapa 1: transforma preços suavizados em log-retornos."""
        self.console.print("\n[bold yellow]Etapa 1: Transformação para Log-Retornos[/bold yellow]")
        returns: dict[str, pd.Series] = {}
        for ticker, prices_s in smoothed.items():
            ret = log_returns(prices_s, k=1)
            if not ret.empty:
                returns[ticker] = ret

        if target_ticker not in returns:
            raise ValueError(f"Série alvo '{target_ticker}' vazia após smoothing + retornos.")

        return returns, returns[target_ticker]

    def _step2_prepare_candidates(
        self,
        candidate_tickers: list[str],
        returns: dict[str, pd.Series],
    ) -> tuple[list[str], dict[str, str]]:
        """Etapa 2: filtra candidatos com dados insuficientes."""
        self.console.print("\n[bold yellow]Etapa 2: Preparação dos Candidatos[/bold yellow]")
        dropped: dict[str, str] = {}
        candidates: list[str] = []

        for ticker in candidate_tickers:
            if ticker not in returns or len(returns[ticker]) < 10:
                dropped[ticker] = "Dados insuficientes"
                continue
            candidates.append(ticker)

        self.console.print(f"Candidatos em análise: {len(candidates)}")
        return candidates, dropped

    def _step3_auto_mi(
        self, target_ret: pd.Series, lag_max: int, knn_k: int
    ) -> int:
        """Etapa 3: calcula a embedding dimension via Auto-MI."""
        self.console.print("\n[bold yellow]Etapa 3: Auto-MI do Alvo (y_lags)[/bold yellow]")
        auto_mi_lag, _ = self._compute_auto_mi_lag(target_ret, lag_max, knn_k)
        self.console.print(
            f"Embedding dimension do alvo (y_lags): [bold green]{auto_mi_lag}[/bold green]"
        )
        return auto_mi_lag

    def _step4_surrogates(
        self,
        candidates: list[str],
        returns: dict[str, pd.Series],
    ) -> dict[str, np.ndarray]:
        """Etapa 4: gera surrogates circulares para cada candidato."""
        self.console.print("\n[bold yellow]Etapa 4: Geração Única de Surrogates Circulares[/bold yellow]")
        surrogates_dict = {}
        for i, ticker in enumerate(candidates):
            surrogates_dict[ticker] = generate_circular_surrogates(
                returns[ticker],
                n_surrogates=self.n_surrogates,
                seed=42 + i,  # seed diferente por candidato
            )
        self.console.print(
            f"Gerados {self.n_surrogates} surrogates para {len(candidates)} candidatos."
        )
        return surrogates_dict

    def _step5_te_matrix(
        self,
        candidates: list[str],
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
        surrogates_dict: dict[str, np.ndarray],
        auto_mi_lag: int,
        lag_max: int,
        alpha: float,
    ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Etapa 5: constrói a matriz de TE significativa (candidato × lag).

        Retorna também os p-valores e um cache dos resultados Granger
        para evitar recálculo nas etapas seguintes.
        """
        self.console.print("\n[bold yellow]Etapa 5: Matriz de Transfer Entropy Significativa[/bold yellow]")
        te_matrix_data = {}
        p_value_data = {}
        granger_cache: dict[str, dict] = {}

        for ticker in candidates:
            c_ret = returns[ticker]
            c_surrs = surrogates_dict[ticker]
            te_matrix_data[ticker] = {}
            p_value_data[ticker] = {}

            # Granger (calculado UMA vez por candidato e cacheado)
            g_res = granger_causality(target_ret, [c_ret], lag_max=lag_max, alpha=alpha)
            g_pvals = {}
            if g_res:
                dk = next(iter(g_res))
                g_pvals = g_res[dk].get("all_p_values", {})
            granger_cache[ticker] = {"result": g_res, "p_values": g_pvals}

            for lag in range(1, lag_max + 1):
                te_obs, p_val = test_te_significance(
                    c_ret, target_ret, c_surrs, lag, y_lags=auto_mi_lag, n_jobs=-1
                )
                p_value_data[ticker][f"lag_{lag}"] = p_val

                g_pval_lag = g_pvals.get(lag, 1.0)
                is_granger_causal = g_pval_lag <= alpha

                if p_val >= alpha and not is_granger_causal:
                    te_matrix_data[ticker][f"lag_{lag}"] = 0.0
                else:
                    te_matrix_data[ticker][f"lag_{lag}"] = te_obs

        score_matrix = pd.DataFrame.from_dict(te_matrix_data, orient="index")

        # Heatmap
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
            fig_score.update_layout(
                title="Etapa 5: Matriz de TE Significativa",
                template="plotly_white",
            )
            fig_score.show()
        except Exception:
            pass

        return score_matrix, p_value_data, granger_cache

    def _step6_lag_consensus(self, score_matrix: pd.DataFrame) -> Optional[int]:
        """Etapa 6: identifica o lag endógeno (maior soma de TE)."""
        self.console.print("\n[bold yellow]Etapa 6: Identificação do Lag Endógeno[/bold yellow]")
        col_sums = score_matrix.sum(axis=0)

        if col_sums.max() == 0:
            self.console.print(
                "[bold red]Nenhum candidato apresentou TE significativa em nenhum lag![/bold red]"
            )
            return None

        lag_consensus = int(col_sums.idxmax().split("_")[1])
        self.console.print(
            f"Lag endógeno (maior soma de TE): [bold green]{lag_consensus}[/bold green] "
            f"com score {col_sums.max():.4f}"
        )
        return lag_consensus

    def _step7_top_k(
        self,
        score_matrix: pd.DataFrame,
        p_value_data: dict,
        granger_cache: dict,
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
        lag_consensus: int,
        auto_mi_lag: int,
        alpha: float,
        knn_k: int,
        dropped: dict[str, str],
    ) -> tuple[list[SelectedCandidate], dict[str, str]]:
        """Etapa 7: seleciona os Top-K candidatos no lag endógeno."""
        self.console.print(
            f"\n[bold yellow]Etapa 7: Seleção Top-{self.top_k} no Lag {lag_consensus}[/bold yellow]"
        )
        col_name = f"lag_{lag_consensus}"
        scores = score_matrix[col_name].sort_values(ascending=False)

        selected: list[SelectedCandidate] = []
        table_all = Table(
            show_header=True, header_style="bold magenta",
            title=f"Status de Todos os Candidatos (Lag {lag_consensus})",
        )
        table_all.add_column("Candidato", style="cyan")
        table_all.add_column("TE Observada", justify="right")
        table_all.add_column("TE P-Valor", justify="right")
        table_all.add_column("Status Final", justify="left")

        for ticker, score in scores.items():
            ticker_str = str(ticker)
            pval = p_value_data[ticker_str][col_name]

            if score <= 0.0 or score < self.score_threshold:
                dropped[ticker_str] = "Não causal (TE não significativa)"
                table_all.add_row(
                    ticker_str, f"{score:.4f}", f"{pval:.4f}",
                    "[red]Dropado (Não Causal / P-val alto)[/red]",
                )
                continue
            if len(selected) >= self.top_k:
                dropped[ticker_str] = f"Fora do Top-{self.top_k}"
                table_all.add_row(
                    ticker_str, f"{score:.4f}", f"{pval:.4f}",
                    "[yellow]Dropado (Rank Baixo)[/yellow]",
                )
                continue

            # MI para compatibilidade
            mi_vals = mutual_information_lags(
                target_ret, returns[ticker_str], max_lag=lag_consensus, k=knn_k
            )
            mi_at_lag = next((mi for l, mi in mi_vals if l == lag_consensus), 0.0)

            # Granger do cache (sem recalcular)
            g_cached = granger_cache.get(ticker_str, {})
            g_pvals = g_cached.get("p_values", {})
            g_pval = g_pvals.get(lag_consensus, 1.0)

            selected.append(SelectedCandidate(
                ticker=ticker_str,
                component="retorno",
                series=returns[ticker_str],
                lag_tau=lag_consensus,
                relation_type="non-linear",
                mi_value=mi_at_lag,
                te_value=score,
                granger_pvalue=g_pval,
            ))
            table_all.add_row(
                ticker_str, f"{score:.4f}", f"{pval:.4f}",
                "[green]Selecionado[/green]",
            )

        self.console.print(table_all)
        return selected, dropped

    def _step8_enrichment(
        self,
        selected_tickers: list[str],
        target_ticker: str,
        smoothed: dict[str, pd.Series],
        returns: dict[str, pd.Series],
        target_ret: pd.Series,
        lag_consensus: int,
        auto_mi_lag: int,
        alpha: float,
    ) -> tuple[dict, dict, dict, dict, dict]:
        """Etapa 8: STL, cointegração de Johansen e TE de componentes."""
        self.console.print(
            "\n[bold yellow]Etapa 8: Enriquecimento Pós-Seleção (STL e Cointegração)[/bold yellow]"
        )

        stl_components = self._compute_stl(
            [target_ticker] + selected_tickers, smoothed
        )

        # ADF e Johansen para tendências e resíduos
        coint_trend, coint_resid = self._compute_cointegration(
            target_ticker, selected_tickers, stl_components, alpha
        )

        # TE entre componentes STL
        trend_te, residual_te = self._compute_component_te(
            target_ticker, selected_tickers, stl_components,
            lag_consensus, auto_mi_lag,
        )

        return stl_components, coint_trend, coint_resid, trend_te, residual_te

    def _step9_features_and_surrogates(
        self,
        coint_trend: dict,
        coint_resid: dict,
        scores: pd.Series,
        returns: dict[str, pd.Series],
        window: int,
        target_ret: pd.Series,
        lag_consensus: int,
        auto_mi_lag: int,
        alpha: float,
    ) -> pd.DataFrame:
        """Etapa 9: constrói features derivadas e valida com surrogates."""
        self.console.print(
            "\n[bold yellow]Etapa 9: Teste de Surrogates e Resumo Final das Features[/bold yellow]"
        )

        generated_features = self._compute_features(
            coint_trend, coint_resid, scores, returns, window
        )

        if generated_features.empty:
            return generated_features

        # Validar cada feature via surrogates
        table_feat = Table(
            show_header=True, header_style="bold magenta",
            title=f"Resumo Final das Features (Lag {lag_consensus})",
        )
        table_feat.add_column("Feature", style="cyan")
        table_feat.add_column("TE Observada", justify="right")
        table_feat.add_column("P-Valor (TE)", justify="right")
        table_feat.add_column("Status", justify="left")

        valid_features = []
        for col in generated_features.columns:
            f_series = generated_features[col].dropna()
            if len(f_series) < 10:
                table_feat.add_row(col, "-", "-", "[red]Dropada (Insuficiente)[/red]")
                continue

            f_surrs = generate_circular_surrogates(
                f_series, n_surrogates=self.n_surrogates, seed=42
            )
            te_obs, p_val = test_te_significance(
                f_series, target_ret, f_surrs, lag_consensus,
                y_lags=auto_mi_lag, n_jobs=-1,
            )

            if p_val < alpha:
                table_feat.add_row(
                    col, f"{te_obs:.4f}", f"{p_val:.4f}",
                    "[green]Escolhida[/green]",
                )
                valid_features.append(col)
            else:
                table_feat.add_row(
                    col, f"{te_obs:.4f}", f"{p_val:.4f}",
                    "[red]Dropada[/red]",
                )

        self.console.print(table_feat)
        return generated_features[valid_features]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_cointegration(
        self,
        target_ticker: str,
        selected_tickers: list[str],
        stl_components: dict,
        alpha: float,
    ) -> tuple[dict, dict]:
        """Testa cointegração de Johansen para tendências e resíduos."""
        coint_trend: dict = {}
        coint_resid: dict = {}

        table_stl = Table(
            show_header=True, header_style="bold blue",
            title="Propriedades STL e Cointegração",
        )
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
                if is_stat:
                    all_trends_nonstat = False
                table_stl.add_row(
                    "Tendência", t, f"{ap:.4f}",
                    "[green]Sim[/green]" if is_stat else "[red]Não[/red]",
                )
                trend_series.append(tr)

            if re is not None:
                ap = adf(re, alpha=alpha)['p_value']
                is_stat = ap < alpha
                if is_stat:
                    all_resids_nonstat = False
                table_stl.add_row(
                    "Resíduo", t, f"{ap:.4f}",
                    "[green]Sim[/green]" if is_stat else "[red]Não[/red]",
                )
                resid_series.append(re)

        self.console.print(table_stl)

        if all_trends_nonstat and len(trend_series) > 1:
            try:
                coint_trend = johansen_cointegration(trend_series, lag_order=1)
                is_coint = coint_trend.get("is_cointegrated", False)
                rank = coint_trend.get("rank", 0)
                msg = (
                    f"[bold green]=> Tendências Cointegradas![/bold green] Rank: {rank}"
                    if is_coint
                    else "[yellow]=> Tendências NÃO Cointegradas.[/yellow]"
                )
                self.console.print(msg)
            except Exception as e:
                self.console.print(f"[red]Erro Johansen (Tendência): {e}[/red]")
        else:
            self.console.print(
                "[dim]=> Cointegração de Tendências pulada (componentes estacionários)[/dim]"
            )

        if all_resids_nonstat and len(resid_series) > 1:
            try:
                coint_resid = johansen_cointegration(resid_series, lag_order=1)
                is_coint = coint_resid.get("is_cointegrated", False)
                rank = coint_resid.get("rank", 0)
                msg = (
                    f"[bold green]=> Resíduos Cointegrados![/bold green] Rank: {rank}"
                    if is_coint
                    else "[yellow]=> Resíduos NÃO Cointegrados.[/yellow]"
                )
                self.console.print(msg)
            except Exception as e:
                self.console.print(f"[red]Erro Johansen (Resíduo): {e}[/red]")
        else:
            self.console.print(
                "[dim]=> Cointegração de Resíduos pulada (componentes estacionários)[/dim]"
            )

        return coint_trend, coint_resid

    def _compute_component_te(
        self,
        target_ticker: str,
        selected_tickers: list[str],
        stl_components: dict,
        lag_consensus: int,
        auto_mi_lag: int,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Calcula TE entre componentes STL dos preditores e do alvo."""
        trend_te: dict[str, float] = {}
        residual_te: dict[str, float] = {}

        target_trend = stl_components.get(target_ticker, {}).get("trend")
        target_residual = stl_components.get(target_ticker, {}).get("residual")

        for ticker in selected_tickers:
            pred_trend = stl_components.get(ticker, {}).get("trend")
            pred_residual = stl_components.get(ticker, {}).get("residual")

            if pred_trend is not None and target_trend is not None:
                try:
                    trend_te[ticker] = transfer_entropy(
                        pred_trend, target_trend, lag=lag_consensus, y_lags=auto_mi_lag
                    )
                except Exception:
                    trend_te[ticker] = 0.0

            if pred_residual is not None and target_residual is not None:
                try:
                    residual_te[ticker] = transfer_entropy(
                        pred_residual, target_residual, lag=lag_consensus, y_lags=auto_mi_lag
                    )
                except Exception:
                    residual_te[ticker] = 0.0

        return trend_te, residual_te

    def _compute_features(
        self, coint_trend: dict, coint_resid: dict,
        scores: pd.Series, returns: dict, window: int,
    ) -> pd.DataFrame:
        """Constrói features derivadas (cointegração, rolling stats)."""
        features_list = []

        # 1. Séries de cointegração
        if coint_trend and coint_trend.get("coint_series") is not None:
            coint_df = coint_trend["coint_series"]
            for col in coint_df.columns:
                features_list.append(coint_df[col].rename(f"coint_trend_{col}"))

        if coint_resid and coint_resid.get("coint_series") is not None:
            coint_df_resid = coint_resid["coint_series"]
            for col in coint_df_resid.columns:
                features_list.append(coint_df_resid[col].rename(f"coint_resid_{col}"))

        # 2. Features rolling para candidatos relevantes
        relevant_tickers = [str(t) for t, score in scores.items() if score > 0.0]

        def safe_entropy(x):
            if np.max(x) == np.min(x):
                return 0.0
            counts, _ = np.histogram(x, bins='fd')
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))

        def dist_argmax_argmin(x):
            return abs(np.argmax(x) - np.argmin(x))

        for ticker in relevant_tickers:
            s = returns[ticker]
            features_list.append(s.rename(f"{ticker}_valor"))

            r_max = s.rolling(window=window).max().rename(f"{ticker}_max")
            r_min = s.rolling(window=window).min().rename(f"{ticker}_min")
            r_dist = s.rolling(window=window).apply(
                dist_argmax_argmin, raw=True
            ).rename(f"{ticker}_dist_max_min")
            r_ent = s.rolling(window=window).apply(
                safe_entropy, raw=True
            ).rename(f"{ticker}_entropia")

            features_list.extend([r_max, r_min, r_dist, r_ent])

        if features_list:
            return pd.concat(features_list, axis=1)
        return pd.DataFrame()

    def _empty_result(self, *, window: int, dropped: dict[str, str]) -> PipelineResult:
        return PipelineResult(
            selected=[], lag_consensus=0, score_matrix=pd.DataFrame(),
            auto_mi_lag=1, smoothing_window=window,
            smoothing_method=self.config.smoothing_method,
            stl_components={}, cointegration={}, descriptives={},
            dropped=dropped, trend_transfer_entropy={},
            residual_transfer_entropy={}, generated_features=pd.DataFrame(),
        )

    @staticmethod
    def _compute_auto_mi_lag(
        target_ret: pd.Series, lag_max: int, knn_k: int
    ) -> tuple[int, list]:
        results = mutual_information_lags(
            target_ret, target_ret, max_lag=min(lag_max, 10), k=knn_k
        )
        if not results:
            return 1, []
        max_lag = max(results, key=lambda pair: pair[1])[0]
        return max_lag, results

    def _compute_stl(
        self, selected_tickers: list[str], smoothed: dict[str, pd.Series]
    ) -> dict[str, dict[str, pd.Series]]:
        stl_results = {}
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
    def _compute_descriptives(
        selected_tickers: list[str], returns: dict[str, pd.Series],
        target_ret: pd.Series, lag_max: int, alpha: float,
    ) -> dict[str, Any]:
        target_acf = acf_pacf(target_ret, lags=lag_max, alpha=alpha)
        descriptives = {"target_acf_pacf": target_acf}
        for ticker in selected_tickers:
            c_ret = returns[ticker]
            g_res = granger_causality(
                target_ret, [c_ret], lag_max=lag_max, alpha=alpha
            )
            descriptives[ticker] = {
                "stats": stats(c_ret),
                "pearson": pearson(c_ret, target_ret),
                "spearman": spearman(c_ret, target_ret),
                "acf_pacf": acf_pacf(c_ret, lags=lag_max, alpha=alpha),
                "granger_causality": g_res,
            }
        return descriptives
