import pandas as pd
from typing import List, Dict, Any
from . import features
from . import tree
from . import validation
from . import report
from finalise.predictors.selector import SelectedCandidate

class Model:
    def __init__(self, selected: List[SelectedCandidate], target_series: pd.Series, config: Any):
        self.selected = selected
        self.target_series = target_series
        self.config = config
        
        # Results
        self.is_outlier = False
        self.z_score = 0.0
        self.model_return = 0.0
        self.agent_returns = []
        self.feature_importance = {}
        self.results = {}
        self.model_obj = None

    def run(self):
        # 1. Build features
        X = features.build(self.selected)
        y = self.target_series
        
        if X.empty:
            raise ValueError("Matriz de features vazia. Não é possível treinar o modelo.")
            
        # Alinha as features e o target, removendo NaNs
        common_idx = X.index.intersection(y.index)
        data = pd.concat([X.loc[common_idx], y.loc[common_idx]], axis=1).dropna()
        if data.empty:
            raise ValueError("Sem observações válidas após o alinhamento de X e y.")
            
        X_clean = data[X.columns]
        y_clean = data.iloc[:, -1]
            
        # 2. Treinar modelo
        # Padrão: usar apenas max_depth=3 para evitar overfitting severo como base
        self.model_obj, self.feature_importance = tree.fit(X_clean, y_clean, max_depth=5)
        
        # 3. Validação (Random Walk Backtest)
        n_agents = 1000
        mod_ret, ag_rets = validation.random_walk_backtest(
            self.model_obj, X_clean, y_clean, n_agents=n_agents, seed=42
        )
        
        # Predições para métricas (apesar de termos simplificado no validation)
        y_pred = self.model_obj.predict(X_clean)
        
        val_metrics = validation.metrics(y_clean, y_pred, ag_rets, mod_ret)
        
        # Populando as propriedades públicas conforme o design doc
        self.model_return = val_metrics["model_return"]
        self.agent_returns = ag_rets
        self.z_score = val_metrics["z_score"]
        self.is_outlier = val_metrics["is_outlier"]
        
        self.results = {
            "model_return": self.model_return,
            "agents_mean": val_metrics["agents_mean"],
            "agents_std": val_metrics["agents_std"],
            "z_score": self.z_score,
            "is_outlier": self.is_outlier,
            "feature_importance": self.feature_importance,
            "agent_returns": self.agent_returns
        }

    def report(self) -> dict:
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")
        return report.generate(self.results)

__all__ = ["Model"]
