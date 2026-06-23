import numpy as np
import pandas as pd
import pytest
from finalise.model import features
from finalise.workflows.predictors_analysis import SelectedCandidate


def test_ut_feat_01():
    """Features representam valores correntes (sem shift)."""
    s1 = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
    cand = SelectedCandidate(
        ticker="TICKER1",
        component="retorno",
        series=s1,
        lag_tau=2,
        relation_type="non-linear",
        mi_value=0.5,
        te_value=None,
        granger_pvalue=0.01
    )
    X = features.build([cand])

    # Features usam valores correntes (sem shift)
    assert "TICKER1_retorno" in X.columns
    # Valor em t=0 deve ser s1[0] (sem deslocamento)
    assert X.iloc[0, 0] == 1.0
    assert X.iloc[1, 0] == 2.0
    assert X.iloc[2, 0] == 3.0


def test_ut_feat_02():
    """Nenhuma feature em t usa dados de t+1 ou posterior (sem look-ahead)."""
    s1 = pd.Series([10, 20, 30, 40], index=[0, 1, 2, 3])
    cand = SelectedCandidate(
        ticker="A", component="retorno", series=s1, lag_tau=1,
        relation_type="non-linear", mi_value=0.1, te_value=None, granger_pvalue=0.01
    )
    X = features.build([cand])
    # Feature em t=0 é o valor corrente s1[0]
    assert X.iloc[0, 0] == 10
    # Sem shift, os valores são diretos
    assert X.iloc[3, 0] == 40


def test_ut_feat_03():
    """Matriz X tem shape (T, N_candidatos)."""
    s1 = pd.Series(np.arange(10))
    s2 = pd.Series(np.arange(10))
    cand1 = SelectedCandidate("A", "retorno", s1, 2, "non-linear", 0.1, None, 0.0)
    cand2 = SelectedCandidate("B", "retorno", s2, 3, "non-linear", 0.1, None, 0.0)
    X = features.build([cand1, cand2])
    assert X.shape == (10, 2)


def test_ut_feat_04():
    """Features de múltiplos candidatos são alinhadas por índice."""
    dates = pd.date_range("2020-01-01", periods=4)
    s1 = pd.Series([1, 2, 3, 4], index=dates)
    s2 = pd.Series([10, 20, 30, 40], index=dates)
    cand1 = SelectedCandidate("A", "retorno", s1, 1, "non-linear", 0.1, None, 0.0)
    cand2 = SelectedCandidate("B", "retorno", s2, 2, "non-linear", 0.1, None, 0.0)
    X = features.build([cand1, cand2])

    # Valores correntes, sem shift
    assert X.iloc[2, 0] == 3.0   # A em t=2
    assert X.iloc[2, 1] == 30.0  # B em t=2


def test_ut_feat_empty():
    assert features.build([]).empty
