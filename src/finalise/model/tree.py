"""
tree.py
-------
Treinamento da árvore de decisão classificatória.

Fluxo:
  1. Alinhamento e limpeza de X (features) e y (rótulos +1/-1).
  2. Embaralhamento (shuffle) dos dados — a árvore aprende a relação
     features→classe sem nenhuma noção de causalidade temporal.
  3. Split 80-20 (treino / teste), mantendo a proporção de classes.
  4. GridSearchCV (5-fold estratificado) sobre hiperparâmetros relevantes
     da DecisionTreeClassifier.
  5. Retorna o melhor estimador, os hiperparâmetros, as acurácias e as
     importâncias de features.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle as sk_shuffle
from typing import Dict, Tuple, Any


# Grade de hiperparâmetros para o GridSearchCV
PARAM_GRID: Dict[str, Any] = {
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__class_weight": [None, "balanced"],
}


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    cv_folds: int = 5,
) -> Tuple[Pipeline, Dict[str, Any], float, float, Dict[str, float]]:
    """
    Treina um DecisionTreeClassifier com busca de hiperparâmetros via GridSearchCV.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de features já construída (sem NaNs).
    y : pd.Series
        Rótulos +1/-1 gerados pelo labeler (mesmo índice de X).
    random_state : int
        Semente para reprodutibilidade do shuffle e do split.
    test_size : float
        Fração reservada para teste (padrão 0.20 → split 80-20).
    cv_folds : int
        Número de folds para a validação cruzada estratificada.

    Returns
    -------
    best_model : DecisionTreeClassifier
        Modelo treinado com os melhores hiperparâmetros.
    best_params : dict
        Hiperparâmetros selecionados pelo GridSearchCV.
    train_accuracy : float
        Acurácia no conjunto de treino.
    test_accuracy : float
        Acurácia no conjunto de teste.
    feature_importances : dict
        Importância de cada feature (baseada em impureza de Gini/Entropia).

    Raises
    ------
    ValueError
        Se X ou y estiverem vazios ou sem sobreposição de índices.
    """
    if X.empty or y.empty:
        raise ValueError("X ou y não podem ser vazios para o treinamento.")

    # ── 1. Alinhamento e remoção de NaNs ──────────────────────────────────
    common_idx = X.index.intersection(y.index)
    if len(common_idx) == 0:
        raise ValueError(
            "X e y não possuem índices em comum. Verifique o alinhamento temporal."
        )

    X_aligned = X.loc[common_idx].copy()
    y_aligned = y.loc[common_idx].copy()

    data = pd.concat([X_aligned, y_aligned], axis=1).dropna()
    if data.empty:
        raise ValueError(
            "Sem observações válidas após remover NaNs. "
            "Verifique os lags e o alinhamento entre features e rótulos."
        )

    X_clean = data[X.columns].values
    y_clean = data["label"].values
    feature_names = list(X.columns)

    # ── 2. Embaralhamento ─────────────────────────────────────────────────
    # Quebra qualquer estrutura temporal: a árvore aprende
    # padrão features→classe, não dependência temporal.
    X_shuffled, y_shuffled = sk_shuffle(X_clean, y_clean, random_state=random_state)

    # ── 3. Split 80-20 estratificado ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled,
        y_shuffled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_shuffled,
    )

    # ── 4. GridSearchCV ───────────────────────────────────────────────────
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=random_state))
    ])
    cv_strategy = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,         # Re-treina o melhor modelo em todo o conjunto de treino
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_model: Pipeline = grid_search.best_estimator_
    best_params: Dict[str, Any] = grid_search.best_params_

    # ── 5. Métricas de acurácia ───────────────────────────────────────────
    train_accuracy = float(best_model.score(X_train, y_train))
    test_accuracy = float(best_model.score(X_test, y_test))

    # ── 6. Importância de features ────────────────────────────────────────
    dt_model = best_model.named_steps["classifier"]
    feature_importances: Dict[str, float] = dict(
        zip(feature_names, dt_model.feature_importances_)
    )

    return best_model, best_params, train_accuracy, test_accuracy, feature_importances


__all__ = ["fit", "PARAM_GRID"]
