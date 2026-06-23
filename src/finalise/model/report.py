"""
report.py
---------
Geração do relatório de treinamento e validação do modelo.

Saídas:
  1. Terminal (rich): painel com métricas comparativas (Modelo vs Buy&Hold vs Perfeito),
     hiperparâmetros, acurácias, win rates e resultado do random walk.
  2. Gráfico plotly: série temporal de retorno acumulado com:
     - Traços cinzas: agentes aleatórios (amostra).
     - Faixas de desvio-padrão (μ ± 1σ, ± 2σ, ± 3σ).
     - Traço verde: árvore de decisão.
     - Traço dourado: retorno perfeito (perfect foresight).
     - Traço azul pontilhado: buy-and-hold.
  3. Gráfico de importância de features.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from typing import Any, Dict, Optional

# Número máximo de traços de agentes individuais no gráfico
MAX_AGENT_TRACES = 200

# Opacidades das faixas de σ
SIGMA_BANDS = [
    (3, "rgba(100,100,200,0.10)"),
    (2, "rgba(100,100,200,0.18)"),
    (1, "rgba(100,100,200,0.28)"),
]


def _build_sigma_bands(
    agents_cumulative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    """Calcula μ(t) e σ(t) dos agentes ao longo do tempo."""
    mu_t = np.mean(agents_cumulative, axis=0)
    sigma_t = np.std(agents_cumulative, axis=0)

    bands = []
    for n_sigma, _ in SIGMA_BANDS:
        lower = mu_t - n_sigma * sigma_t
        upper = mu_t + n_sigma * sigma_t
        bands.append((n_sigma, lower, upper))

    return mu_t, sigma_t, bands


def _build_random_walk_figure(results: Dict[str, Any]) -> go.Figure:
    """Constrói o gráfico plotly de série temporal do random walk."""
    model_cumulative = results["model_cumulative"]
    perfect_cumulative = results["perfect_cumulative"]
    bnh_cumulative = results["bnh_cumulative"]
    agents_cumulative = results["agents_cumulative"]
    z_score = results["z_score"]
    is_outlier = results["is_outlier"]
    model_return = results["model_return"]
    horizon = results["horizon"]
    trade_dates = results.get("trade_dates", None)

    T = len(model_cumulative)
    if trade_dates is not None and len(trade_dates) == T:
        x_axis = list(trade_dates)
    else:
        x_axis = list(range(T))

    fig = go.Figure()

    # ── 1. Faixas de desvio-padrão ────────────────────────────────────────
    mu_t, sigma_t, bands = _build_sigma_bands(agents_cumulative)

    for n_sigma, color in SIGMA_BANDS:
        _, lower, upper = next(
            (b for b in bands if b[0] == n_sigma), (None, None, None)
        )
        if lower is None:
            continue

        fig.add_trace(go.Scatter(
            x=x_axis, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip", name=f"+{n_sigma}σ",
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=color,
            showlegend=(n_sigma == 3),
            name=f"±{n_sigma}σ agentes" if n_sigma == 3 else f"±{n_sigma}σ",
            hoverinfo="skip",
        ))

    # ── 2. Média dos agentes ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis, y=mu_t, mode="lines",
        line=dict(color="rgba(140,140,160,0.7)", width=1.5, dash="dot"),
        name="μ agentes",
        hovertemplate="μ: %{y:.4f}<extra></extra>",
    ))

    # ── 3. Agentes individuais (amostra) ──────────────────────────────────
    n_to_show = min(MAX_AGENT_TRACES, agents_cumulative.shape[0])
    rng = np.random.default_rng(0)
    idx_sample = rng.choice(agents_cumulative.shape[0], size=n_to_show, replace=False)

    for i, agent_idx in enumerate(idx_sample):
        fig.add_trace(go.Scatter(
            x=x_axis, y=agents_cumulative[agent_idx], mode="lines",
            line=dict(color="rgba(160,160,170,0.15)", width=0.6),
            showlegend=(i == 0), name="Agentes aleatórios",
            hoverinfo="skip", legendgroup="agents",
        ))

    # ── 4. Buy-and-Hold ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis, y=bnh_cumulative, mode="lines",
        line=dict(color="#2196f3", width=2.0, dash="dash"),
        name=f"Buy & Hold ({results['bnh_return']:.4f})",
        hovertemplate="B&H: %{y:.4f}<extra></extra>",
    ))

    # ── 5. Retorno Perfeito ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_axis, y=perfect_cumulative, mode="lines",
        line=dict(color="#ffc107", width=2.5, dash="dashdot"),
        name=f"Perfeito ({results['perfect_return']:.4f})",
        hovertemplate="Perfeito: %{y:.4f}<extra></extra>",
    ))

    # ── 6. Árvore de decisão ──────────────────────────────────────────────
    tree_color = "#00c853" if is_outlier else "#ff5252"
    status_label = "✓ Outlier" if is_outlier else "✗ Não supera agentes"
    fig.add_trace(go.Scatter(
        x=x_axis, y=model_cumulative, mode="lines",
        line=dict(color=tree_color, width=3.0),
        name=f"Árvore (z={z_score:.2f} | {status_label})",
        hovertemplate="Árvore: %{y:.4f}<extra></extra>",
    ))

    # ── 7. Layout ─────────────────────────────────────────────────────────
    title_text = (
        f"Random Walk Backtest — Posições Não-Sobrepostas (horizon={horizon}d)<br>"
        f"<sup>z-score = {z_score:.2f}σ | "
        f"Modelo = {model_return:.4f} | "
        f"{'<b>VÁLIDO (outlier)</b>' if is_outlier else 'INVÁLIDO (não supera agentes)'}</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title="Data do Trade",
        yaxis_title="Retorno Acumulado (log)",
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        hovermode="x unified",
        height=620,
    )

    return fig


def generate(
    results: dict,
    predictor_analysis=None,
    config=None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gera o relatório consolidado no terminal (rich) e retorna gráficos plotly.

    Parameters
    ----------
    results : dict
        Dicionário de resultados produzido por Model.validate().
    predictor_analysis : PipelineResult, opcional
        Resultado da fase de seleção de preditores.
    config : Config, opcional
    horizon : int, opcional

    Returns
    -------
    dict com gráficos plotly: {"random_walk": ..., "feature_importance": ...}
    """
    console = Console()

    # ── Desempacota ────────────────────────────────────────────────────────
    best_params = results.get("best_params", {})
    cv_score = results.get("cv_score", 0.0)
    train_acc = results.get("train_accuracy", 0.0)
    test_acc = results.get("test_accuracy", 0.0)
    model_return = results.get("model_return", 0.0)
    perfect_return = results.get("perfect_return", 0.0)
    bnh_return = results.get("bnh_return", 0.0)
    z_score = results.get("z_score", 0.0)
    is_outlier = results.get("is_outlier", False)
    win_rate = results.get("win_rate", 0.0)
    win_rate_buy = results.get("win_rate_buy", 0.0)
    win_rate_sell = results.get("win_rate_sell", 0.0)
    n_buy = results.get("n_buy", 0)
    n_sell = results.get("n_sell", 0)
    n_trades = results.get("n_trades", 0)
    efficiency = results.get("efficiency", 0.0)
    sharpe = results.get("sharpe", 0.0)
    max_drawdown = results.get("max_drawdown", 0.0)
    annual_model = results.get("annual_model", 0.0)
    annual_bnh = results.get("annual_bnh", 0.0)
    annual_perfect = results.get("annual_perfect", 0.0)
    vol_annual = results.get("vol_annual", 0.0)
    period_start = results.get("period_start", "?")
    period_end = results.get("period_end", "?")
    h = results.get("horizon", horizon or 1)
    feature_importance = results.get("feature_importance", {})

    status_color = "green" if is_outlier else "red"
    status_text = "VÁLIDO (Outlier)" if is_outlier else "INVÁLIDO"

    console.print()
    console.print(Panel(
        "[bold white]RELATÓRIO DE TREINAMENTO — MÓDULO MODEL[/bold white]",
        style="bold white on dark_blue", expand=False,
    ))

    # ── Preditores (da fase anterior) ─────────────────────────────────────
    if predictor_analysis is not None:
        selected = getattr(predictor_analysis, "selected", [])
        p_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        p_table.add_column("Ticker / Feature", style="cyan")
        p_table.add_column("Componente", style="yellow")
        p_table.add_column("Lag (tau*)", justify="center")
        for sel in selected:
            p_table.add_row(sel.ticker, sel.component, str(sel.lag_tau))

        generated = getattr(predictor_analysis, "generated_features", None)
        if generated is not None and not generated.empty:
            lag_c = getattr(predictor_analysis, "lag_consensus", h)
            for col in generated.columns:
                p_table.add_row(col, "Engenharia", str(lag_c))

        console.print(Panel(
            p_table,
            title="[bold blue]Preditores & Features Selecionadas[/bold blue]",
            border_style="blue",
        ))

    # ── Hiperparâmetros ───────────────────────────────────────────────────
    hp_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
    hp_table.add_column("Hiperparâmetro", style="cyan")
    hp_table.add_column("Valor", justify="right")
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
    acc_table.add_row("Acurácia — Teste (out-of-sample)", f"{test_acc:.2%}")
    acc_table.add_row("CV Score (média dos folds)", f"{cv_score:.2%}")
    console.print(Panel(
        acc_table,
        title="[bold yellow]Acurácias[/bold yellow]",
        border_style="yellow",
    ))

    # ── Tabela Comparativa de Performance ─────────────────────────────────
    perf_table = Table(
        show_header=True, header_style="bold white",
        box=box.DOUBLE_EDGE,
        title=f"Comparativo de Performance (Período: {period_start} → {period_end})",
    )
    perf_table.add_column("Métrica", style="cyan", min_width=25)
    perf_table.add_column("Modelo", justify="right", style="bold green" if is_outlier else "bold red")
    perf_table.add_column("Buy & Hold", justify="right", style="blue")
    perf_table.add_column("Perfeito", justify="right", style="yellow")

    capital = 1000.0

    perf_table.add_row(
        "Retorno Total",
        f"{(np.exp(model_return) - 1) * 100:.1f}%",
        f"{(np.exp(bnh_return) - 1) * 100:.1f}%",
        f"{(np.exp(perfect_return) - 1) * 100:.1f}%",
    )
    perf_table.add_row(
        "Retorno Anualizado",
        f"{(np.exp(annual_model) - 1) * 100:.1f}%",
        f"{(np.exp(annual_bnh) - 1) * 100:.1f}%",
        f"{(np.exp(annual_perfect) - 1) * 100:.1f}%",
    )
    perf_table.add_row(
        f"Capital Final (R$ {capital:.0f})",
        f"R$ {capital * np.exp(model_return):.2f}",
        f"R$ {capital * np.exp(bnh_return):.2f}",
        f"R$ {capital * np.exp(perfect_return):.2f}",
    )
    perf_table.add_row(
        "Volatilidade Anualizada",
        f"{vol_annual * 100:.1f}%",
        "—",
        "—",
    )
    perf_table.add_row(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
        "—",
        "—",
    )
    perf_table.add_row(
        "Max Drawdown",
        f"{max_drawdown * 100:.1f}%",
        "—",
        "0.0%",
    )
    perf_table.add_row(
        "Win Rate",
        f"{win_rate:.1%}",
        "—",
        "100.0%",
    )
    perf_table.add_row(
        "Eficiência vs Perfeito",
        f"{efficiency:.1f}%",
        "—",
        "100.0%",
    )
    console.print(Panel(perf_table, border_style="white"))

    # ── Detalhes do Trading ───────────────────────────────────────────────
    trade_table = Table(show_header=False, box=box.SIMPLE)
    trade_table.add_column("", style="cyan")
    trade_table.add_column("", justify="right")
    trade_table.add_row("Horizonte por Trade", f"{h} dias")
    trade_table.add_row("Total de Trades", str(n_trades))
    trade_table.add_row("Sinais de Compra", f"{n_buy} ({n_buy/n_trades*100:.0f}%)" if n_trades > 0 else "0")
    trade_table.add_row("Sinais de Venda", f"{n_sell} ({n_sell/n_trades*100:.0f}%)" if n_trades > 0 else "0")
    trade_table.add_row("Win Rate (Compras)", f"{win_rate_buy:.1%}")
    trade_table.add_row("Win Rate (Vendas)", f"{win_rate_sell:.1%}")
    console.print(Panel(
        trade_table,
        title="[bold cyan]Detalhes do Trading[/bold cyan]",
        border_style="cyan",
    ))

    # ── Resultado do Random Walk ──────────────────────────────────────────
    rw_table = Table(show_header=False, box=box.SIMPLE)
    rw_table.add_column("", style="cyan")
    rw_table.add_column("", justify="right")
    rw_table.add_row(
        "Status",
        f"[bold {status_color}]{status_text}[/bold {status_color}]",
    )
    rw_table.add_row("Z-Score vs Agentes", f"{z_score:.2f}σ")
    rw_table.add_row("Média dos Agentes", f"{results.get('agents_mean', 0):.4f}")
    rw_table.add_row("Desvio-Padrão dos Agentes", f"{results.get('agents_std', 0):.4f}")
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
            title="Importância das Features",
            show_header=True, header_style="bold magenta", box=box.SIMPLE,
        )
        fi_table.add_column("Feature", style="cyan")
        fi_table.add_column("Importância", justify="right")
        for feat, imp in sorted_fi:
            fi_table.add_row(feat, f"{imp:.4f}")
        console.print(fi_table)

        feats = [x[0] for x in sorted_fi]
        imps = [x[1] for x in sorted_fi]
        fig_fi = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h", marker_color="#7c4dff",
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

    # ── Gráfico principal ─────────────────────────────────────────────────
    model_cumulative = results.get("model_cumulative", np.array([]))
    agents_cumulative = results.get("agents_cumulative", np.zeros((1, 1)))

    if len(model_cumulative) > 0 and agents_cumulative.shape[1] > 0:
        fig_rw = _build_random_walk_figure(results)
        figures["random_walk"] = fig_rw

    return figures


__all__ = ["generate"]
