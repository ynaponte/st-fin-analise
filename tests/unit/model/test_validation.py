import numpy as np
import pandas as pd
from finalise.model import validation


class MockModel:
    def __init__(self, pred):
        self.pred = pred
    def predict(self, X):
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        return self.pred[:n]


def test_ut_val_01():
    """Backtest produz resultados com n_agents retornos."""
    X = pd.DataFrame(np.random.randn(20, 2))
    y = pd.Series(np.random.randn(20))
    model = MockModel(np.ones(20))

    results = validation.random_walk_backtest(model, X, y, horizon=1, n_agents=50, seed=42)
    assert len(results["agent_returns"]) == 50
    assert isinstance(results["model_return"], float)


def test_ut_val_02():
    """Agentes aleatórios têm distribuição centrada em 0 para retornos zero-mean."""
    X = pd.DataFrame(np.random.randn(200, 2))
    y = pd.Series(np.random.randn(200))
    model = MockModel(np.ones(200))
    
    results = validation.random_walk_backtest(model, X, y, horizon=1, n_agents=1000, seed=42)
    assert np.abs(np.mean(results["agent_returns"])) < 2.0


def test_ut_val_perfect_model():
    """Modelo perfeito é identificado como outlier."""
    np.random.seed(42)
    y = pd.Series(np.random.randn(200))
    perfect_pred = np.sign(y.values).astype(float)
    perfect_pred[perfect_pred == 0] = 1.0
    model = MockModel(perfect_pred)
    X = pd.DataFrame(np.zeros((200, 2)))

    results = validation.random_walk_backtest(model, X, y, horizon=1, n_agents=1000, seed=42)
    
    assert results["model_return"] > 0
    assert results["is_outlier"] is True
    assert results["z_score"] > 2.5


def test_ut_val_daily_trades():
    """Rebalanceamento diário: n_trades = len(data)."""
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randn(100))
    model = MockModel(np.ones(100))

    results = validation.random_walk_backtest(model, X, y, horizon=5, n_agents=10, seed=42)
    
    # 100 pontos = 100 trades de 1 dia
    assert results["n_trades"] == 100


def test_ut_val_metrics_present():
    """Verifica que todas as métricas esperadas estão presentes."""
    X = pd.DataFrame(np.random.randn(50, 2))
    y = pd.Series(np.random.randn(50))
    model = MockModel(np.ones(50))

    results = validation.random_walk_backtest(model, X, y, horizon=1, n_agents=10, seed=42)
    
    expected_keys = [
        "model_return", "perfect_return", "bnh_return",
        "win_rate", "sharpe", "max_drawdown", "efficiency",
        "z_score", "is_outlier", "n_trades",
        "model_cumulative", "perfect_cumulative", "bnh_cumulative",
    ]
    for key in expected_keys:
        assert key in results, f"Chave '{key}' ausente nos resultados"


def test_ut_val_perfect_return():
    """Retorno perfeito captura o valor absoluto de todos os retornos diários."""
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    X = pd.DataFrame(np.zeros((5, 1)))
    y = pd.Series(returns)
    model = MockModel(np.ones(5))

    results = validation.random_walk_backtest(model, X, y, horizon=1, n_agents=10, seed=42)
    
    expected_perfect = np.sum(np.abs(returns))
    assert np.isclose(results["perfect_return"], expected_perfect, atol=1e-10)
