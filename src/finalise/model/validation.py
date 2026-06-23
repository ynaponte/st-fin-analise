"""
validation.py
-------------
Validação do classificador via rebalanceamento diário.

O backtest simula um bot de trading com rebalanceamento diário:
  - A cada dia, o modelo avalia as features e decide comprar (+1) ou vender (-1).
  - O P&L desse dia é: sinal(t) × retorno_diário(t+1).
  - Como o modelo foi treinado para prever uma tendência de `horizon` dias, 
    seu sinal tende a ser persistente por esse período.

Para comparação justa, gera-se:
  - n_agents agentes aleatórios. Cada agente muda de opinião a cada `horizon` dias,
    mantendo a mesma frequência teórica de trades da árvore.
  - Retorno perfeito (perfect foresight diário): sempre acerta a direção de t+1.
  - Buy-and-hold: compra e mantém.

Métricas calculadas:
  - Z-score do modelo vs distribuição dos agentes
  - Win rate (total e por classe)
  - Sharpe ratio anualizado
  - Max drawdown
  - Eficiência vs retorno perfeito
  - Retorno anualizado
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


def random_walk_backtest(
    model: Any,
    X: pd.DataFrame,
    y_next_day_returns: pd.Series,
    horizon: int = 1,
    n_agents: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executa o backtest com rebalanceamento diário.

    Parameters
    ----------
    model : Pipeline
        Modelo treinado que emite sinais +1/-1.
    X : pd.DataFrame
        Matriz de features correntes no instante t (dados de TESTE).
    y_next_day_returns : pd.Series
        Retorno do dia seguinte r(t+1), alinhado com X(t).
    horizon : int
        Horizonte de previsão para o qual o modelo foi treinado.
        Usado para simular o tempo de holding dos agentes aleatórios.
    n_agents : int
        Número de agentes aleatórios para comparação.
    seed : int
        Semente para reprodutibilidade.

    Returns
    -------
    dict com todas as métricas e séries para o relatório.
    """
    rng = np.random.default_rng(seed)

    # ── Alinhamento ───────────────────────────────────────────────────────
    common_idx = X.index.intersection(y_next_day_returns.index)
    if len(common_idx) == 0:
        return _empty_results(n_agents)

    X_aligned = X.loc[common_idx]
    fwd_returns = y_next_day_returns.loc[common_idx]

    valid_mask = fwd_returns.notna() & X_aligned.notna().all(axis=1)
    X_valid = X_aligned[valid_mask]
    fwd_valid = fwd_returns[valid_mask].values.astype(float)
    dates_valid = X_valid.index

    n_trades = len(fwd_valid)
    if n_trades == 0:
        return _empty_results(n_agents)

    # ── Sinais do modelo (decisão tomada em t) ────────────────────────────
    model_signals = model.predict(X_valid.values).astype(float)

    # ── P&L do modelo (recompensa obtida em t+1) ──────────────────────────
    model_pnl = model_signals * fwd_valid
    model_cumulative = np.cumsum(model_pnl)
    model_return = float(model_cumulative[-1])

    # ── Retorno perfeito diário (perfect foresight) ───────────────────────
    perfect_signals = np.sign(fwd_valid).astype(float)
    perfect_signals[perfect_signals == 0] = 1.0
    perfect_pnl = perfect_signals * fwd_valid
    perfect_cumulative = np.cumsum(perfect_pnl)
    perfect_return = float(perfect_cumulative[-1])

    # ── Buy-and-hold ──────────────────────────────────────────────────────
    bnh_pnl = fwd_valid.copy()
    bnh_cumulative = np.cumsum(bnh_pnl)
    bnh_return = float(bnh_cumulative[-1])

    # ── Agentes aleatórios ────────────────────────────────────────────────
    # Para ser justo, o agente aleatório não flipa todo dia (o que destrói sinal).
    # Ele toma uma decisão e a mantém por `horizon` dias.
    agent_signals = np.zeros((n_agents, n_trades))
    for i in range(0, n_trades, horizon):
        chunk_size = min(horizon, n_trades - i)
        agent_signals[:, i:i+chunk_size] = rng.choice([-1.0, 1.0], size=(n_agents, 1))

    agents_pnl = agent_signals * fwd_valid[np.newaxis, :]
    agents_cumulative = np.cumsum(agents_pnl, axis=1)
    agent_returns = agents_cumulative[:, -1]

    # ── Métricas ──────────────────────────────────────────────────────────
    mu_agents = float(np.mean(agent_returns))
    sigma_agents = float(np.std(agent_returns))
    z_score = float((model_return - mu_agents) / sigma_agents) if sigma_agents > 0 else 0.0
    is_outlier = bool(z_score > 2.5)

    correct = (model_signals == np.sign(fwd_valid)).astype(float)
    nonzero_mask = fwd_valid != 0
    win_rate = float(np.mean(correct[nonzero_mask])) if nonzero_mask.any() else 0.0

    buy_mask = model_signals > 0
    sell_mask = model_signals < 0
    win_rate_buy = float(np.mean(correct[buy_mask & nonzero_mask])) if (buy_mask & nonzero_mask).any() else 0.0
    win_rate_sell = float(np.mean(correct[sell_mask & nonzero_mask])) if (sell_mask & nonzero_mask).any() else 0.0

    n_buy = int(buy_mask.sum())
    n_sell = int(sell_mask.sum())

    efficiency = float(model_return / perfect_return * 100) if perfect_return > 0 else 0.0

    # Sharpe ratio anualizado (~252 trades por ano já que é rebalanceamento diário)
    trades_per_year = 252.0
    pnl_mean = float(np.mean(model_pnl))
    pnl_std = float(np.std(model_pnl))
    sharpe = float(pnl_mean / pnl_std * np.sqrt(trades_per_year)) if pnl_std > 0 else 0.0

    running_max = np.maximum.accumulate(model_cumulative)
    drawdowns = model_cumulative - running_max
    max_drawdown = float(np.min(drawdowns))

    n_years = n_trades / 252.0
    annual_model = float(model_return / n_years) if n_years > 0 else 0.0
    annual_bnh = float(bnh_return / n_years) if n_years > 0 else 0.0
    annual_perfect = float(perfect_return / n_years) if n_years > 0 else 0.0

    vol_annual = float(pnl_std * np.sqrt(trades_per_year))

    period_start = str(dates_valid[0].date()) if hasattr(dates_valid[0], 'date') else str(dates_valid[0])
    period_end = str(dates_valid[-1].date()) if hasattr(dates_valid[-1], 'date') else str(dates_valid[-1])

    return {
        "model_cumulative": model_cumulative,
        "perfect_cumulative": perfect_cumulative,
        "bnh_cumulative": bnh_cumulative,
        "agents_cumulative": agents_cumulative,
        "trade_dates": dates_valid,
        "model_return": model_return,
        "perfect_return": perfect_return,
        "bnh_return": bnh_return,
        "agent_returns": agent_returns,
        "agents_mean": mu_agents,
        "agents_std": sigma_agents,
        "z_score": z_score,
        "is_outlier": is_outlier,
        "win_rate": win_rate,
        "win_rate_buy": win_rate_buy,
        "win_rate_sell": win_rate_sell,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_trades": n_trades,
        "efficiency": efficiency,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annual_model": annual_model,
        "annual_bnh": annual_bnh,
        "annual_perfect": annual_perfect,
        "vol_annual": vol_annual,
        "period_start": period_start,
        "period_end": period_end,
        "horizon": horizon,
        "model_pnl": model_pnl,
        "model_signals": model_signals,
        "forward_returns": fwd_valid,
    }


def _empty_results(n_agents: int) -> Dict[str, Any]:
    return {
        "model_cumulative": np.array([0.0]),
        "perfect_cumulative": np.array([0.0]),
        "bnh_cumulative": np.array([0.0]),
        "agents_cumulative": np.zeros((n_agents, 1)),
        "trade_dates": pd.DatetimeIndex([]),
        "model_return": 0.0,
        "perfect_return": 0.0,
        "bnh_return": 0.0,
        "agent_returns": np.zeros(n_agents),
        "agents_mean": 0.0,
        "agents_std": 0.0,
        "z_score": 0.0,
        "is_outlier": False,
        "win_rate": 0.0,
        "win_rate_buy": 0.0,
        "win_rate_sell": 0.0,
        "n_buy": 0,
        "n_sell": 0,
        "n_trades": 0,
        "efficiency": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "annual_model": 0.0,
        "annual_bnh": 0.0,
        "annual_perfect": 0.0,
        "vol_annual": 0.0,
        "period_start": "",
        "period_end": "",
        "horizon": 1,
        "model_pnl": np.array([]),
        "model_signals": np.array([]),
        "forward_returns": np.array([]),
    }


__all__ = ["random_walk_backtest"]
