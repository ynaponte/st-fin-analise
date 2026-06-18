"""
validation.py
-------------
Validação do classificador via random walk.

O random walk aqui é uma simulação de trading:
  - O modelo emite sinais +1 (compra) ou -1 (venda) para cada período t.
  - O P&L de um período é:  sinal(t) × retorno_real(t).
  - O retorno acumulado é a soma cumulativa desses P&Ls ao longo do tempo.

Para cada um dos n_agents agentes aleatórios, o processo é idêntico, mas
os sinais são amostrados uniformemente de {-1, +1}.

A validação usa os dados em **ordem temporal original** (sem embaralhamento),
pois o objetivo é simular uma estratégia de trading real no tempo.

Retorna tanto os valores finais (para z-score) quanto as séries temporais
completas (para o gráfico de série temporal com curvas de nível).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def random_walk_backtest(
    model: Any,
    X: pd.DataFrame,
    y_returns: pd.Series,
    n_agents: int = 1000,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa o backtest comparando a árvore com n_agents aleatórios.

    Os dados são usados em ordem temporal (sem embaralhamento) para que
    o retorno acumulado seja uma série temporal interpretável.

    Parameters
    ----------
    model : DecisionTreeClassifier
        Modelo treinado que emite sinais +1/-1.
    X : pd.DataFrame
        Matriz de features, em ordem temporal.
    y_returns : pd.Series
        Log-retornos reais do ativo alvo, em ordem temporal (não rótulos!).
        Usado para calcular o P&L efetivo de cada sinal.
    n_agents : int
        Número de agentes aleatórios para comparação.
    seed : int
        Semente para reprodutibilidade dos agentes aleatórios.

    Returns
    -------
    model_return : float
        Retorno acumulado total do modelo ao final do período.
    agent_returns : np.ndarray, shape (n_agents,)
        Retorno acumulado final de cada agente aleatório.
    model_cumulative : np.ndarray, shape (T,)
        Série temporal de retorno acumulado do modelo (P&L cumulativo).
    agents_cumulative : np.ndarray, shape (n_agents, T)
        Séries temporais de retorno acumulado de cada agente aleatório.

    Raises
    ------
    ValueError
        Se X e y_returns não tiverem sobreposição de índices.
    """
    rng = np.random.default_rng(seed)

    # Alinhamento temporal: X e y_returns devem ter os mesmos índices
    common_idx = X.index.intersection(y_returns.index)
    if len(common_idx) == 0:
        return 0.0, np.zeros(n_agents), np.zeros(1), np.zeros((n_agents, 1))

    X_aligned = X.loc[common_idx].values  # numpy array: evita warning de feature names
    ret_aligned = y_returns.loc[common_idx].values.astype(float)

    # Sinais do modelo: +1 ou -1
    model_signals = model.predict(X_aligned).astype(float)

    # P&L por período e retorno cumulativo do modelo
    model_pnl = model_signals * ret_aligned
    model_cumulative = np.cumsum(model_pnl)
    model_return = float(model_cumulative[-1])

    # Agentes aleatórios: n_agents × T sinais em {-1, +1}
    agent_signals = rng.choice([-1.0, 1.0], size=(n_agents, len(ret_aligned)))

    # P&L por período e retorno cumulativo por agente
    # agents_pnl: (n_agents, T)
    agents_pnl = agent_signals * ret_aligned[np.newaxis, :]
    agents_cumulative = np.cumsum(agents_pnl, axis=1)

    # Retorno final de cada agente
    agent_returns = agents_cumulative[:, -1]

    return model_return, agent_returns, model_cumulative, agents_cumulative


def metrics(
    agent_returns: np.ndarray,
    model_return: float,
    z_threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Calcula métricas de comparação entre o modelo e os agentes aleatórios.

    Parameters
    ----------
    agent_returns : np.ndarray
        Retornos finais dos agentes (usados para estimar μ e σ da distribuição).
    model_return : float
        Retorno final do modelo.
    z_threshold : float
        Limiar de z-score para considerar o modelo como outlier positivo.

    Returns
    -------
    dict com:
        model_return, agents_mean, agents_std, z_score, is_outlier
    """
    mu = float(np.mean(agent_returns))
    sigma = float(np.std(agent_returns))

    z_score = float((model_return - mu) / sigma) if sigma > 0 else 0.0
    is_outlier = bool(z_score > z_threshold)

    return {
        "model_return": model_return,
        "agents_mean": mu,
        "agents_std": sigma,
        "z_score": z_score,
        "is_outlier": is_outlier,
    }


__all__ = ["random_walk_backtest", "metrics"]
