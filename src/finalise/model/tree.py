"""
tree.py
-------
Treinamento da árvore de decisão classificatória.

Fluxo:
  1. Recebe X_train e y_train já separados temporalmente pelo chamador.
  2. GridSearchCV com TimeSeriesSplit para evitar vazamento de dados.
  3. Retorna o melhor estimador, os hiperparâmetros, os scores CV e as
     importâncias de features.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Dict, Tuple, Any


# Grade de hiperparâmetros simplificada para evitar overfitting
PARAM_GRID: Dict[str, Any] = {
    "scaler": [StandardScaler(), MinMaxScaler(), "passthrough"],
    "classifier__max_depth": [2, 3, 4, 5, 6, 7, 8],
    "classifier__min_samples_split": [5, 10, 20],
    "classifier__min_samples_leaf": [5, 10, 20],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__class_weight": [None, "balanced"],
}


def fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = 42,
    cv_folds: int = 5,
    horizon: int = 1,
) -> Tuple[Pipeline, Dict[str, Any], float, Dict[str, float]]:
    """
    Treina um DecisionTreeClassifier com TimeSeriesSplit.

    Usa TimeSeriesSplit com gap=horizon para garantir que a validação
    no GridSearchCV nunca esbarre nos rótulos de treino do mesmo período.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("X_train ou y_train não podem ser vazios para o treinamento.")

    feature_names = list(X_train.columns)
    X_vals = X_train.values
    y_vals = y_train.values

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(random_state=random_state))
    ])
    
    # ── TimeSeriesSplit para evitar Data Leakage ──────────────────────────
    # gap=horizon garante que a janela de validação não pegue retornos futuros
    # que estavam sendo usados no fold de treino.
    cv_strategy = TimeSeriesSplit(
        n_splits=cv_folds, 
        gap=horizon
    )

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=PARAM_GRID,
        scoring="accuracy",
        cv=cv_strategy,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_vals, y_vals)

    best_model: Pipeline = grid_search.best_estimator_
    best_params: Dict[str, Any] = grid_search.best_params_
    cv_score = float(grid_search.best_score_)

    dt_model = best_model.named_steps["classifier"]
    feature_importances: Dict[str, float] = dict(
        zip(feature_names, dt_model.feature_importances_)
    )

    return best_model, best_params, cv_score, feature_importances


__all__ = ["fit", "PARAM_GRID"]
