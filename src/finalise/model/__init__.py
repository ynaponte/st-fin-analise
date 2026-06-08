import pandas as pd
from typing import List, Dict, Any
from . import features
from . import tree
from . import validation
from . import report
from finalise.predictors.selector import SelectedCandidate

class Model:
    def __init__(self, selected: List[SelectedCandidate], target_series: pd.Series, config: Any, horizon: int = None):
        self.selected = selected
        self.target_series = target_series
        self.config = config
        self.horizon = horizon
        
        # Results
        self.is_outlier = False
        self.z_score = 0.0
        self.model_return = 0.0
        self.agent_returns = []
        self.feature_importance = {}
        self.results = {}
        self.model_obj = None

    def build_data(self) -> tuple[pd.DataFrame, pd.Series]:
        X = features.build(self.selected)
        y = self.target_series
        
        if X.empty:
            raise ValueError("Matriz de features vazia.")
            
        common_idx = X.index.intersection(y.index)
        data = pd.concat([X.loc[common_idx], y.loc[common_idx]], axis=1).dropna()
        if data.empty:
            raise ValueError("Sem observações válidas após o alinhamento de X e y.")
            
        return data[X.columns], data.iloc[:, -1]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_obj, self.feature_importance = tree.fit(X, y, max_depth=5)

    def predict(self, X: pd.DataFrame):
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado ou carregado.")
        return self.model_obj.predict(X)

    def validate(self, X: pd.DataFrame, y: pd.Series, n_agents: int = 1000):
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado ou carregado para validação.")
            
        mod_ret, ag_rets = validation.random_walk_backtest(
            self.model_obj, X, y, n_agents=n_agents, seed=42
        )
        
        y_pred = self.predict(X)
        val_metrics = validation.metrics(y, y_pred, ag_rets, mod_ret)
        
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

    def run(self):
        """Pipeline monolítica original que agrupa as chamadas para conveniência."""
        X_clean, y_clean = self.build_data()
        self.fit(X_clean, y_clean)
        self.validate(X_clean, y_clean)
        
    def load(self, filepath: str):
        """Carrega um modelo pré-treinado do disco."""
        import joblib
        self.model_obj = joblib.load(filepath)

    def report(self, target_analysis=None, predictor_analysis=None) -> dict:
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")
        return report.generate(
            self.results,
            target_analysis=target_analysis,
            predictor_analysis=predictor_analysis,
            target_series=self.target_series,
            config=self.config,
            horizon=self.horizon
        )
        
    def save(self, filepath: str):
        """
        Salva o modelo treinado (DecisionTreeRegressor) em um arquivo usando joblib.
        """
        import joblib
        if self.model_obj is None:
            raise ValueError("O modelo ainda não foi treinado. Execute run() primeiro.")
        joblib.dump(self.model_obj, filepath)

__all__ = ["Model"]
