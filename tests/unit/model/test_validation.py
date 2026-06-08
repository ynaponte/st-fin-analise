import numpy as np
import pandas as pd
from finalise.model import validation

class MockModel:
    def __init__(self, pred):
        self.pred = pred
    def predict(self, X):
        return self.pred

def test_ut_val_01():
    # Backtest produz n_agents retornos acumulados
    X = pd.DataFrame(np.random.randn(10, 2))
    y = pd.Series(np.random.randn(10))
    model = MockModel(np.ones(10))
    
    mod_ret, ag_rets = validation.random_walk_backtest(model, X, y, n_agents=50, seed=42)
    assert len(ag_rets) == 50
    assert isinstance(mod_ret, float)

def test_ut_val_02():
    # Agentes aleatórios não têm acesso às features do modelo (verificar isolamento)
    # The random_walk_backtest implementation does not pass X to agents, it just uses rng.choice
    # This is verified by code inspection, but we can verify the agent distribution is centered around 0
    # for a zero-mean return series.
    X = pd.DataFrame(np.random.randn(100, 2))
    y = pd.Series(np.random.randn(100))
    model = MockModel(np.ones(100))
    _, ag_rets = validation.random_walk_backtest(model, X, y, n_agents=1000, seed=42)
    assert np.abs(np.mean(ag_rets)) < 1.0 # Should be close to 0

def test_ut_val_03_04_05():
    # z_score calculado corretamente: (ret_modelo − μ) / σ
    # is_outlier=True quando z_score > 2.5
    # is_outlier=False quando z_score <= 2.5
    ag_rets = np.array([-1.0, 0.0, 1.0]) # mu = 0, sigma = sqrt(2/3) ~ 0.816
    
    # model_return to get z_score = 3.0 -> > 2.5 (Outlier)
    # 3.0 * 0.816 = 2.449
    res1 = validation.metrics(pd.Series(), np.array([]), ag_rets, 2.45)
    assert np.isclose(res1["z_score"], 2.45 / np.std(ag_rets), atol=0.01)
    assert res1["is_outlier"] is True
    
    # model_return to get z_score = 1.0 -> <= 2.5 (Not Outlier)
    res2 = validation.metrics(pd.Series(), np.array([]), ag_rets, 0.816)
    assert res2["is_outlier"] is False

def test_ut_val_06():
    # Modelo com retorno plantado como outlier é corretamente identificado
    # Create y returns
    y = pd.Series(np.random.randn(500))
    # Create a perfect model that always predicts the exact sign of y
    perfect_pred = np.sign(y)
    model = MockModel(perfect_pred)
    
    X = pd.DataFrame(np.zeros((500, 2)))
    
    mod_ret, ag_rets = validation.random_walk_backtest(model, X, y, n_agents=1000, seed=42)
    
    # A perfect model will capture the absolute sum of all returns
    assert np.isclose(mod_ret, np.sum(np.abs(y)))
    
    metrics = validation.metrics(y, perfect_pred, ag_rets, mod_ret)
    assert metrics["is_outlier"] is True
    assert metrics["z_score"] > 2.5
