import pandas as pd
import numpy as np
from finalise.model import Model
from finalise.predictors.selector import SelectedCandidate
from finalise.config import Config

def test_it_model_pipeline_01(prices_synthetic):
    # Model.run() completo com candidatos sintéticos selecionados
    # target
    y = np.log(prices_synthetic / prices_synthetic.shift(1)).dropna()
    
    # features
    cand1 = SelectedCandidate(
        ticker="TEST1", component="bruto",
        series=y.copy(), lag_tau=1, relation_type="linear",
        mi_value=0.5, te_value=None, granger_pvalue=0.01
    )
    
    config = Config("TARGET", ["TEST1"])
    
    m = Model([cand1], y, config)
    m.run()
    
    assert m.model_obj is not None
    assert len(m.agent_returns) == 1000
    assert "TEST1_bruto_lag1" in m.feature_importance
    
    figs = m.report()
    assert "agent_distribution" in figs
    assert "feature_importance" in figs

def test_it_model_pipeline_02_03(prices_synthetic):
    # IT-M-02 Modelo com retorno acumulado plantado acima de 2.5σ retorna is_outlier=True
    # IT-M-03 Modelo com retorno aleatório retorna is_outlier=False na maioria das seeds
    y = np.log(prices_synthetic / prices_synthetic.shift(1)).dropna()
    
    # We create a feature that is highly predictive of y
    # To do this, let's create a candidate whose lag_tau=1 perfectly matches y
    # so series[t] at lag 1 matches y[t]. Thus series = y.shift(-1)
    perfect_feat = y.shift(-1).fillna(0)
    
    cand_perfect = SelectedCandidate(
        ticker="PERFECT", component="bruto",
        series=perfect_feat, lag_tau=1, relation_type="linear",
        mi_value=1.0, te_value=None, granger_pvalue=0.0
    )
    
    m_perfect = Model([cand_perfect], y, Config("TARGET", ["PERFECT"]))
    m_perfect.run()
    assert m_perfect.is_outlier is True
    assert m_perfect.z_score > 2.5
    
    # Aleatório
    rng = np.random.default_rng(42)
    random_feat = pd.Series(rng.standard_normal(len(y)), index=y.index)
    cand_random = SelectedCandidate(
        ticker="RANDOM", component="bruto",
        series=random_feat, lag_tau=1, relation_type="linear",
        mi_value=0.1, te_value=None, granger_pvalue=0.5
    )
    
    m_random = Model([cand_random], y, Config("TARGET", ["RANDOM"]))
    m_random.run()
    assert m_random.is_outlier is False
