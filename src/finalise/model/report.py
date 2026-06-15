"""
report.py
---------
Geração do relatório de treinamento e validação do modelo.

Saídas:
  1. Terminal (rich): painel de hiperparâmetros, acurácias e resultado do
     random walk.
  2. Gráfico plotly: série temporal de retorno acumulado com:
     - Traços cinzas: agentes aleatórios (amostra de até MAX_AGENT_TRACES).
     - Traço verde destacado: árvore de decisão.
     - Faixas de desvio-padrão (μ ± 1σ, ± 2σ, ± 3σ) calculadas ponto a
       ponto ao longo do tempo — as "curvas de nível".
     Essas faixas permitem avaliar visualmente se a árvore é um outlier
     estatístico em relação ao campo de agentes aleatórios.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import Any, Dict, Optional

# Número máximo de traços de agentes individuais no gráfico
# (para não sobrecarregar o navegador)
MAX_AGENT_TRACES = 200

# Opacidades das faixas de σ (da mais clara à mais escura, de fora para dentro)
SIGMA_BANDS = [
    (3, "rgba(100,100,200,0.10)"),
    (2, "rgba(100,100,200,0.18)"),
    (1, "rgba(100,100,200,0.28)"),
]


def _build_sigma_bands(
    agents_cumulative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    """
    Calcula μ(t) e σ(t) dos agentes ao longo do tempo.

    Returns
    -------
    mu_t : (T,) média cumulativa dos agentes em cada passo t
    sigma_t : (T,) desvio-padrão cumulativo dos agentes em cada passo t
    bands : list of (n_sigma, lower, upper) para cada múltiplo de σ
    """
    mu_t = np.mean(agents_cumulative, axis=0)
    sigma_t = np.std(agents_cumulative, axis=0)

    bands = []
    for n_sigma, _ in SIGMA_BANDS:
        lower = mu_t - n_sigma * sigma_t
        upper = mu_t + n_sigma * sigma_t
        bands.append((n_sigma, lower, upper))

    return mu_t, sigma_t, bands


def _build_random_walk_figure(
    model_cumulative: np.ndarray,
    agents_cumulative: np.ndarray,
    model_return: float,
    z_score: float,
    is_outlier: bool,
    time_index: Optional[Any] = None,
) -> go.Figure:
    """
    Constrói o gráfico plotly de série temporal do random walk.
    """
    T = len(model_cumulative)
    x_axis = list(time_index) if time_index is not None else list(range(T))

    fig = go.Figure()

    # ── 1. Faixas de desvio-padrão (curvas de nível) ──────────────────────
    mu_t, sigma_t, bands = _build_sigma_bands(agents_cumulative)

    # Renderiza do mais largo para o mais estreito (3σ → 1σ)
    for n_sigma, color in SIGMA_BANDS:
        _, lower, upper = next(
            (b for b in bands if b[0] == n_sigma), (None, None, None)
        )
        if lower is None:
            continue

        # Faixa superior
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name=f"+{n_sigma}σ",
        ))
        # Faixa inferior (preenchida em relação à superior)
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=color,
            showlegend=(n_sigma == 3),  # Legenda apenas na faixa mais externa
            name=f"±{n_sigma}σ agentes" if n_sigma == 3 else f"±{n_sigma}σ",
            hoverinfo="skip",
        ))

    # ── 2. Média dos agentes ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=mu_t,
        mode="lines",
        line=dict(color="rgba(140,140,160,0.7)", width=1.5, dash="dot"),
        name="μ agentes",
        hovertemplate="μ: %{y:.4f}<extra></extra>",
    ))

    # ── 3. Traços individuais dos agentes aleatórios (amostra) ────────────
    n_to_show = min(MAX_AGENT_TRACES, agents_cumulative.shape[0])
    rng = np.random.default_rng(0)
    idx_sample = rng.choice(agents_cumulative.shape[0], size=n_to_show, replace=False)

    for i, agent_idx in enumerate(idx_sample):
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=agents_cumulative[agent_idx],
            mode="lines",
            line=dict(color="rgba(160,160,170,0.15)", width=0.6),
            showlegend=(i == 0),
            name="Agentes aleatórios",
            hoverinfo="skip",
            legendgroup="agents",
        ))

    # ── 4. Traço da árvore de decisão (destaque verde) ────────────────────
    tree_color = "#00c853" if is_outlier else "#ff5252"
    status_label = "✓ Outlier" if is_outlier else "✗ Não supera agentes"
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=model_cumulative,
        mode="lines",
        line=dict(color=tree_color, width=3.0),
        name=f"Árvore (z={z_score:.2f} | {status_label})",
        hovertemplate="Árvore: %{y:.4f}<extra></extra>",
    ))

    # ── 5. Layout ─────────────────────────────────────────────────────────
    title_text = (
        f"Random Walk Backtest — Retorno Acumulado<br>"
        f"<sup>z-score = {z_score:.2f} σ | "
        f"Retorno do modelo = {model_return:.4f} | "
        f"{'<b>VÁLIDO (outlier)</b>' if is_outlier else 'INVÁLIDO (não supera agentes)'}</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title="Período",
        yaxis_title="Retorno Acumulado (log-retorno × sinal)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        height=560,
    )

    return fig


def generate(
    results: dict,
    target_analysis=None,
    predictor_analysis=None,
    target_series=None,
    config=None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gera o relatório consolidado no terminal (rich) e retorna os gráficos
    plotly prontos para exibição.

    Parameters
    ----------
    results : dict
        Dicionário de resultados produzido por Model.run(). Deve conter:
        - best_params, train_accuracy, test_accuracy
        - model_return, agents_mean, agents_std, z_score, is_outlier
        - feature_importance
        - agent_returns, model_cumulative, agents_cumulative
    target_analysis, predictor_analysis : opcionais
        Objetos das fases anteriores do pipeline, para exibição resumida.
    target_series : pd.Series, opcional
        Série alvo, usada para extrair o índice temporal do gráfico.
    config : Config, opcional
    horizon : int, opcional

    Returns
    -------
    dict
        {"random_walk": go.Figure, "feature_importance": go.Figure}
    """
    console = Console()

    # ── Desempacota resultados ─────────────────────────────────────────────
    best_params: dict      = results.get("best_params", {})
    train_acc: float       = results.get("train_accuracy", 0.0)
    test_acc: float        = results.get("test_accuracy", 0.0)
    model_return: float    = results.get("model_return", 0.0)
    agents_mean: float     = results.get("agents_mean", 0.0)
    agents_std: float      = results.get("agents_std", 0.0)
    z_score: float         = results.get("z_score", 0.0)
    is_outlier: bool       = results.get("is_outlier", False)
    feature_importance: dict = results.get("feature_importance", {})
    agent_returns: np.ndarray     = results.get("agent_returns", np.array([]))
    model_cumulative: np.ndarray  = results.get("model_cumulative", np.array([]))
    agents_cumulative: np.ndarray = results.get("agents_cumulative", np.zeros((1, 1)))

    status_color = "green" if is_outlier else "red"
    status_text = "VÁLIDO (Outlier)" if is_outlier else "INVÁLIDO (Não supera agentes aleatórios)"

    console.print()
    console.print(Panel(
        "[bold white]RELATÓRIO DE TREINAMENTO — MÓDULO MODEL (Classificador)[/bold white]",
        style="bold white on dark_blue",
        expand=False,
    ))

    # ── Fase anterior (Alvo) ──────────────────────────────────────────────
    if target_analysis is not None:
        h = target_analysis.horizon
        target_ticker = target_analysis.config.target_ticker
        best_res = target_analysis.results.get(h, {})
        t_table = Table(show_header=True, header_style="bold green", box=box.SIMPLE)
        t_table.add_column("Métrica", style="cyan")
        t_table.add_column("Valor", justify="right")
        t_table.add_row("Horizonte (k*)", f"{h} dias")
        t_table.add_row("Hurst", f"{best_res.get('hurst', 0.5):.4f}")
        t_table.add_row("Entropia de Shannon", f"{best_res.get('shannon', 0.0):.4f}")
        console.print(Panel(
            t_table,
            title=f"[bold green]Fase 1 — Alvo ({target_ticker})[/bold green]",
            border_style="green",
        ))

    # ── Fase anterior (Preditores) ────────────────────────────────────────
    if predictor_analysis is not None:
        selected = predictor_analysis.selected
        p_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        p_table.add_column("Ticker", style="cyan")
        p_table.add_column("Componente", style="yellow")
        p_table.add_column("Lag (τ*)", justify="center")
        p_table.add_column("Tipo", justify="center")
        for sel in selected:
            val_str = (
                f"MI: {sel.mi_value:.4f}"
                if sel.relation_type == "non-linear"
                else f"Granger p={sel.granger_pvalue:.4f}"
            )
            p_table.add_row(sel.ticker, sel.component, str(sel.lag_tau), sel.relation_type.capitalize())
        console.print(Panel(
            p_table,
            title="[bold blue]Fase 2 — Preditores Selecionados[/bold blue]",
            border_style="blue",
        ))

    # ── GridSearch: Hiperparâmetros ───────────────────────────────────────
    hp_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    hp_table.add_column("Hiperparâmetro", style="cyan")
    hp_table.add_column("Valor Selecionado", justify="right")
    for k, v in best_params.items():
        hp_table.add_row(k, str(v))
    console.print(Panel(
        hp_table,
        title="[bold magenta]GridSearch — Melhores Hiperparâmetros[/bold magenta]",
        border_style="magenta",
    ))

    # ── Acurácias ─────────────────────────────────────────────────────────
    acc_table = Table(show_header=False, box=box.SIMPLE)
    acc_table.add_column("", style="cyan")
    acc_table.add_column("", justify="right")
    acc_table.add_row("Acurácia — Treino", f"{train_acc:.2%}")
    acc_table.add_row("Acurácia — Teste", f"{test_acc:.2%}")
    console.print(Panel(
        acc_table,
        title="[bold yellow]Acurácias (split 80-20)[/bold yellow]",
        border_style="yellow",
    ))

    # ── Resultado do Random Walk ──────────────────────────────────────────
    rw_table = Table(show_header=False, box=box.SIMPLE)
    rw_table.add_column("", style="cyan")
    rw_table.add_column("", justify="right")
    rw_table.add_row(
        "Status",
        f"[bold {status_color}]{status_text}[/bold {status_color}]",
    )
    rw_table.add_row("Retorno Acumulado do Modelo", f"{model_return:.4f}")
    rw_table.add_row("Z-Score vs Agentes", f"{z_score:.2f} σ")
    rw_table.add_row("Média dos Agentes", f"{agents_mean:.4f}")
    rw_table.add_row("Desvio-Padrão dos Agentes", f"{agents_std:.4f}")
    console.print(Panel(
        rw_table,
        title=f"[bold {status_color}]Validação — Random Walk[/bold {status_color}]",
        border_style=status_color,
    ))

    # ── Importância de Features ───────────────────────────────────────────
    figures: Dict[str, Any] = {}

    if feature_importance:
        sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        fi_table = Table(
            title="Importância das Features (Árvore de Decisão)",
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE,
        )
        fi_table.add_column("Feature", style="cyan")
        fi_table.add_column("Importância", justify="right")
        for feat, imp in sorted_fi:
            fi_table.add_row(feat, f"{imp:.4f}")
        console.print(fi_table)

        # Gráfico de barras de importância
        feats = [x[0] for x in sorted_fi]
        imps = [x[1] for x in sorted_fi]
        fig_fi = go.Figure(go.Bar(
            x=imps,
            y=feats,
            orientation="h",
            marker_color="#7c4dff",
        ))
        fig_fi.update_layout(
            title="Importância das Features (Árvore de Decisão)",
            xaxis_title="Importância (Gini / Entropia)",
            yaxis_title="Feature",
            template="plotly_white",
            yaxis=dict(autorange="reversed"),
            height=max(300, 50 + 40 * len(feats)),
        )
        figures["feature_importance"] = fig_fi

    # ── Gráfico principal: série temporal do random walk ──────────────────
    if len(model_cumulative) > 0 and agents_cumulative.shape[1] > 0:
        # Índice temporal para o eixo X
        time_index = None
        if target_series is not None and not target_series.empty:
            # Usa os últimos T índices da série alvo
            T = len(model_cumulative)
            if len(target_series.index) >= T:
                time_index = target_series.index[-T:]

        fig_rw = _build_random_walk_figure(
            model_cumulative=model_cumulative,
            agents_cumulative=agents_cumulative,
            model_return=model_return,
            z_score=z_score,
            is_outlier=is_outlier,
            time_index=time_index,
        )
        figures["random_walk"] = fig_rw

    return figures


__all__ = ["generate"]
