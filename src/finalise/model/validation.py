import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def random_walk_backtest(model, X: pd.DataFrame, y: pd.Series, n_agents: int = 1000, seed: int = 42) -> Tuple[float, np.ndarray]:
    """
    Executa um backtest comparando as predições do modelo com 'n_agents' agentes aleatórios.
    O retorno do modelo é a soma do log-retorno real quando o modelo prevê corretamente a direção,
    simplificado pela multiplicação do sinal da predição com o log-retorno real.
    Agentes aleatórios tomam decisões aleatórias de compra/venda (+1/-1).
    """
    rng = np.random.default_rng(seed)
    
    common_idx = X.index.intersection(y.index)
    data = pd.concat([X.loc[common_idx], y.loc[common_idx]], axis=1).dropna()
    
    if data.empty:
        return 0.0, np.zeros(n_agents)
        
    X_clean = data[X.columns]
    y_clean = data.iloc[:, -1]
    
    y_pred = model.predict(X_clean)
    
    # Posições de trade do modelo (sinal da predição)
    # Predições de 0 não resultam em posições (posição 0)
    model_positions = np.sign(y_pred)
    model_return = float(np.sum(model_positions * y_clean.values))
    
    # Agentes aleatórios
    # Gera N matrizes de posições [-1, 1]
    agent_positions = rng.choice([-1, 1], size=(n_agents, len(y_clean)))
    
    # Cada linha de agent_positions é um agente.
    # Multiplica ponto a ponto e soma por eixo 1 para obter o retorno acumulado por agente
    agent_returns = np.sum(agent_positions * y_clean.values[np.newaxis, :], axis=1)
    
    return model_return, agent_returns

def metrics(y_true: pd.Series, y_pred: np.ndarray, agent_returns: np.ndarray, model_return: float) -> Dict[str, Any]:
    """
    Calcula as métricas de validação, incluindo o z-score e se o modelo é outlier
    (supera agentes aleatórios por mais de 2.5 desvios-padrão).
    """
    mu = float(np.mean(agent_returns))
    sigma = float(np.std(agent_returns))
    
    if sigma == 0:
        z_score = 0.0
    else:
        z_score = float((model_return - mu) / sigma)
        
    is_outlier = bool(z_score > 2.5)
    
    return {
        "model_return": model_return,
        "agents_mean": mu,
        "agents_std": sigma,
        "z_score": z_score,
        "is_outlier": is_outlier
    }
