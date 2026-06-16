"""
model/__init__.py
-----------------
API pública do módulo model.

Expõe a classe Model, que orquestra:
  1. build_data()  → constrói X, y_labels e y_returns
  2. fit()         → treina DecisionTreeClassifier com GridSearch (embaralhado)
  3. validate()    → random walk em ordem temporal
  4. run()         → executa o pipeline completo
  5. report()      → exibe relatório rich + retorna gráficos plotly
  6. save()/load() → persistência via joblib
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from typing import Any, Dict, List, Optional, Tuple

from finalise.workflows.predictors_analysis import SelectedCandidate
from . import features, labeler, tree, validation, report as _report


class Model:
    """
    Classificador de direção de mercado baseado em árvore de decisão.

    O modelo recebe as séries preditoras (via SelectedCandidate) e a série
    alvo (log-retornos). Internamente:
      - O rotulador converte os log-retornos em classes +1 (compra) / -1 (venda).
      - Os dados são embaralhados e divididos 80-20 para treinamento.
      - Um GridSearchCV busca os melhores hiperparâmetros da DecisionTreeClassifier.
      - A validação ocorre via random walk em ordem temporal, sem embaralhamento.
      - Um relatório rico é gerado no terminal e gráficos plotly são retornados.

    Parameters
    ----------
    selected : list of SelectedCandidate
        Preditores selecionados pela fase de PredictorAnalysis.
    target_series : pd.Series
        Log-retornos do ativo alvo.
    config : Config
        Configuração global do pipeline.
    horizon : int, opcional
        Horizonte selecionado (apenas informativo para o relatório).
    """

    def __init__(
        self,
        selected: List[SelectedCandidate],
        target_series: pd.Series,
        config: Any,
        horizon: Optional[int] = None,
    ):
        self.selected = selected
        self.target_series = target_series
        self.config = config
        self.horizon = horizon

        # Armazenados após fit()
        self.model_obj = None
        self.best_params: Dict[str, Any] = {}
        self.train_accuracy: float = 0.0
        self.test_accuracy: float = 0.0
        self.feature_importance: Dict[str, float] = {}

        # Armazenados após validate()
        self.model_return: float = 0.0
        self.agent_returns: np.ndarray = np.array([])
        self.model_cumulative: np.ndarray = np.array([])
        self.agents_cumulative: np.ndarray = np.zeros((1, 1))
        self.z_score: float = 0.0
        self.is_outlier: bool = False
        self.results: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # 1. Construção dos dados
    # ──────────────────────────────────────────────────────────────────────

    def build_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Constrói a matriz de features e os vetores de rótulos e retornos.

        Returns
        -------
        X_clean : pd.DataFrame
            Features alinhadas, sem NaNs.
        y_labels : pd.Series
            Rótulos +1/-1 gerados pelo labeler (para treinamento).
        y_returns : pd.Series
            Log-retornos reais alinhados com X (para cálculo do P&L na validação).

        Raises
        ------
        ValueError
            Se os dados resultantes estiverem vazios.
        """
        # Constrói a matriz de features com shifts de lag
        X = features.build(self.selected)
        if X.empty:
            raise ValueError("Matriz de features vazia. Verifique os preditores selecionados.")

        # Rotula a série alvo
        y_labels = labeler.label(self.target_series)

        # Alinha X, y_labels e y_returns (retornos reais)
        # X, y_labels e target_series podem ter índices ligeiramente diferentes
        common_idx = X.index.intersection(y_labels.index).intersection(
            self.target_series.index
        )

        if len(common_idx) == 0:
            raise ValueError(
                "Sem índices em comum entre features, rótulos e série alvo. "
                "Verifique os lags e o horizonte."
            )

        X_clean = X.loc[common_idx]
        y_labels_clean = y_labels.loc[common_idx]
        y_returns_clean = self.target_series.loc[common_idx]

        # Remove quaisquer NaNs residuais
        valid_mask = (
            X_clean.notna().all(axis=1)
            & y_labels_clean.notna()
            & y_returns_clean.notna()
        )
        X_clean = X_clean[valid_mask]
        y_labels_clean = y_labels_clean[valid_mask]
        y_returns_clean = y_returns_clean[valid_mask]

        if X_clean.empty:
            raise ValueError(
                "Sem observações válidas após o alinhamento e remoção de NaNs."
            )

        return X_clean, y_labels_clean, y_returns_clean

    # ──────────────────────────────────────────────────────────────────────
    # 2. Treinamento
    # ──────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y_labels: pd.Series) -> None:
        """
        Treina o classificador com embaralhamento + GridSearchCV.

        Dados embaralhados antes do split para que a árvore aprenda
        a relação features→classe sem causalidade temporal.

        Parameters
        ----------
        X : pd.DataFrame
            Features (output de build_data).
        y_labels : pd.Series
            Rótulos +1/-1 (output de build_data).
        """
        (
            self.model_obj,
            self.best_params,
            self.train_accuracy,
            self.test_accuracy,
            self.feature_importance,
        ) = tree.fit(X, y_labels)

    # ──────────────────────────────────────────────────────────────────────
    # 3. Predição
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Emite sinais +1/-1 para cada observação de X.

        Parameters
        ----------
        X : pd.DataFrame
            Features para predição.

        Returns
        -------
        np.ndarray de int (+1 ou -1) com comprimento len(X).
        """
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado. Execute fit() ou run() primeiro.")
        return self.model_obj.predict(X)

    # ──────────────────────────────────────────────────────────────────────
    # 4. Validação (random walk)
    # ──────────────────────────────────────────────────────────────────────

    def validate(
        self,
        X: pd.DataFrame,
        y_returns: pd.Series,
        n_agents: int = 1000,
    ) -> None:
        """
        Executa o random walk em ordem temporal e calcula métricas.

        Parameters
        ----------
        X : pd.DataFrame
            Features em ordem temporal.
        y_returns : pd.Series
            Log-retornos reais em ordem temporal (NÃO embaralhados).
        n_agents : int
            Número de agentes aleatórios para comparação.
        """
        if self.model_obj is None:
            raise ValueError("O modelo não foi treinado. Execute fit() ou run() primeiro.")

        (
            self.model_return,
            self.agent_returns,
            self.model_cumulative,
            self.agents_cumulative,
        ) = validation.random_walk_backtest(
            self.model_obj, X, y_returns, n_agents=n_agents, seed=42
        )

        val_metrics = validation.metrics(self.agent_returns, self.model_return)

        self.z_score = val_metrics["z_score"]
        self.is_outlier = val_metrics["is_outlier"]

        self.results = {
            # GridSearch / Acurácias
            "best_params": self.best_params,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            # Validação
            "model_return": self.model_return,
            "agents_mean": val_metrics["agents_mean"],
            "agents_std": val_metrics["agents_std"],
            "z_score": self.z_score,
            "is_outlier": self.is_outlier,
            # Séries temporais para o gráfico
            "model_cumulative": self.model_cumulative,
            "agents_cumulative": self.agents_cumulative,
            "agent_returns": self.agent_returns,
            # Features
            "feature_importance": self.feature_importance,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 5. Pipeline completo
    # ──────────────────────────────────────────────────────────────────────

    def run(self, n_agents: int = 1000) -> None:
        """
        Executa o pipeline completo: build_data → fit → validate.

        Parameters
        ----------
        n_agents : int
            Número de agentes aleatórios para o random walk.
        """
        X_clean, y_labels, y_returns = self.build_data()
        self.fit(X_clean, y_labels)
        self.validate(X_clean, y_returns, n_agents=n_agents)

    # ──────────────────────────────────────────────────────────────────────
    # 6. Relatório
    # ──────────────────────────────────────────────────────────────────────

    def report(
        self,
        target_analysis=None,
        predictor_analysis=None,
    ) -> Dict[str, Any]:
        """
        Gera e exibe o relatório de treinamento no terminal (rich) e
        retorna os gráficos plotly.

        Parameters
        ----------
        target_analysis : TargetAnalysis, opcional
            Resultado da Fase 1 (exibido no resumo).
        predictor_analysis : PredictorAnalysis, opcional
            Resultado da Fase 2 (exibido no resumo).

        Returns
        -------
        dict
            {"random_walk": go.Figure, "feature_importance": go.Figure}

        Raises
        ------
        ValueError
            Se run() ainda não foi executado.
        """
        if not self.results:
            raise ValueError("Nenhum resultado disponível. Execute run() primeiro.")

        return _report.generate(
            self.results,
            target_analysis=target_analysis,
            predictor_analysis=predictor_analysis,
            target_series=self.target_series,
            config=self.config,
            horizon=self.horizon,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 7. Persistência
    # ──────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """
        Salva o modelo treinado em disco via joblib.

        Parameters
        ----------
        filepath : str
            Caminho do arquivo de saída (ex.: "modelo_finalise.joblib").
        """
        if self.model_obj is None:
            raise ValueError("O modelo ainda não foi treinado. Execute run() primeiro.")
        joblib.dump(self.model_obj, filepath)

    def load(self, filepath: str) -> None:
        """
        Carrega um modelo previamente salvo do disco.

        Parameters
        ----------
        filepath : str
            Caminho do arquivo .joblib a ser carregado.
        """
        self.model_obj = joblib.load(filepath)


__all__ = ["Model"]
