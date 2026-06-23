"""
model/__init__.py
-----------------
API pública do módulo model.

Expõe a classe Model, que orquestra:
  1. build_data()  → constrói X, y_labels e y_next_day_returns com split temporal
  2. fit()         → treina DecisionTreeClassifier com TimeSeriesSplit (no treino)
  3. validate()    → random walk backtest diário nos dados de teste (nunca vistos)
  4. run()         → executa o pipeline completo
  5. report()      → exibe relatório rich + retorna gráficos plotly
  6. save()/load() → persistência via joblib

Alinhamento temporal correto:
  - Features em t = valores correntes dos preditores (sem shift)
  - Labels em t = sign(retorno acumulado do alvo nos próximos `horizon` dias)
  - P&L Diário = sinal(t) × retorno_diário(t+1)
  - Split temporal: primeiros 80% → treino, últimos 20% → teste
  - Gap de purge entre treino e teste = `horizon` observações
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from typing import Any, Dict, List, Optional, Tuple

from finalise.workflows.predictors_analysis import SelectedCandidate
from . import features, labeler, tree, validation, report as _report


class Model:
    def __init__(
        self,
        selected: List[SelectedCandidate],
        target_series: pd.Series,
        config: Any,
        horizon: int = 1,
        generated_features: Optional[pd.DataFrame] = None,
    ):
        self.selected = selected
        self.target_series = target_series
        self.config = config
        self.horizon = horizon
        self.generated_features = generated_features

        self._X_train: Optional[pd.DataFrame] = None
        self._X_test: Optional[pd.DataFrame] = None
        self._y_train_labels: Optional[pd.Series] = None
        self._y_test_labels: Optional[pd.Series] = None
        self._y_test_next_returns: Optional[pd.Series] = None

        self.model_obj = None
        self.best_params: Dict[str, Any] = {}
        self.cv_score: float = 0.0
        self.train_accuracy: float = 0.0
        self.test_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}
        self.results: Dict[str, Any] = {}

    def build_data(
        self,
        train_ratio: float = 0.80,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Constrói features, labels e retornos diários, com split temporal.
        """
        X = features.build(self.selected)

        if self.generated_features is not None and not self.generated_features.empty:
            X = pd.concat([X, self.generated_features], axis=1)

        if X.empty:
            raise ValueError("Matriz de features vazia. Verifique os preditores selecionados.")

        y_labels = labeler.label(self.target_series, horizon=self.horizon)
        y_next = labeler.next_day_return(self.target_series)

        common_idx = (
            X.index
            .intersection(y_labels.index)
            .intersection(y_next.index)
        )

        if len(common_idx) == 0:
            raise ValueError("Sem índices em comum.")

        X = X.loc[common_idx]
        y_labels = y_labels.loc[common_idx]
        y_next = y_next.loc[common_idx]

        valid_mask = (
            X.notna().all(axis=1)
            & y_labels.notna()
            & y_next.notna()
        )
        X = X[valid_mask]
        y_labels = y_labels[valid_mask].astype(int)
        y_next = y_next[valid_mask]

        if X.empty:
            raise ValueError("Sem observações válidas após alinhamento.")

        n = len(X)
        train_end = int(n * train_ratio)
        test_start = min(train_end + self.horizon, n)

        if test_start >= n:
            raise ValueError("Dados insuficientes para split temporal.")

        self._X_train = X.iloc[:train_end]
        self._X_test = X.iloc[test_start:]
        self._y_train_labels = y_labels.iloc[:train_end]
        self._y_test_labels = y_labels.iloc[test_start:]
        self._y_test_next_returns = y_next.iloc[test_start:]

        return (
            self._X_train,
            self._X_test,
            self._y_train_labels,
            self._y_test_labels,
            self._y_test_next_returns,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Treina o classificador."""
        (
            self.model_obj,
            self.best_params,
            self.cv_score,
            self.feature_importance,
        ) = tree.fit(X_train, y_train, horizon=self.horizon)

        train_pred = self.model_obj.predict(X_train.values)
        self.train_accuracy = float(np.mean(train_pred == y_train.values))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Emite sinais +1/-1 para cada observação de X."""
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado.")
        return self.model_obj.predict(X.values)

    def validate(
        self,
        X_test: pd.DataFrame,
        y_test_next_returns: pd.Series,
        n_agents: int = 1000,
    ) -> None:
        """Executa o random walk backtest nos dados de TESTE."""
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado.")

        self.results = validation.random_walk_backtest(
            self.model_obj,
            X_test,
            y_test_next_returns,
            horizon=self.horizon,
            n_agents=n_agents,
            seed=42,
        )

        if self._y_test_labels is not None:
            test_pred = self.model_obj.predict(X_test.values)
            self.test_accuracy = float(np.mean(test_pred == self._y_test_labels.values))

        self.results["best_params"] = self.best_params
        self.results["cv_score"] = self.cv_score
        self.results["train_accuracy"] = self.train_accuracy
        self.results["test_accuracy"] = self.test_accuracy
        self.results["feature_importance"] = self.feature_importance

    def run(self, n_agents: int = 1000) -> None:
        """Executa o pipeline completo."""
        X_train, X_test, y_train, y_test_labels, y_test_next = self.build_data()
        self.fit(X_train, y_train)
        self.validate(X_test, y_test_next, n_agents=n_agents)

    def report(
        self,
        predictor_analysis=None,
    ) -> Dict[str, Any]:
        """Gera o relatório no terminal e retorna gráficos plotly."""
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")

        return _report.generate(
            self.results,
            predictor_analysis=predictor_analysis,
            config=self.config,
            horizon=self.horizon,
        )

    def save(self, filepath: str) -> None:
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado.")
        joblib.dump(self.model_obj, filepath)

    def load(self, filepath: str) -> None:
        self.model_obj = joblib.load(filepath)

__all__ = ["Model"]
