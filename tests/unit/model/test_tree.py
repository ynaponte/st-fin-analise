import numpy as np
import pandas as pd
from finalise.model import tree
from sklearn.pipeline import Pipeline


def test_ut_tree_01():
    """tree.fit treina sem erro para X e y válidos (classificação)."""
    np.random.seed(42)
    # TimeSeriesSplit precisa de um bom número de amostras
    n = 150
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([1, -1], size=n), name="label")

    model, best_params, cv_score, importance = tree.fit(X, y, horizon=5)
    assert isinstance(model, Pipeline)
    assert len(importance) == 2
    assert cv_score > 0


def test_ut_tree_02_03():
    """feature_importance soma 1.0 e tem comprimento correto."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.where(X["f1"] > 0, 1, -1), name="label")

    model, _, _, importance = tree.fit(X, y)

    assert len(importance) == 3
    assert "f1" in importance
    assert np.isclose(sum(importance.values()), 1.0)


def test_ut_tree_returns_pipeline():
    """O modelo retornado é um Pipeline com scaler + classifier."""
    X = pd.DataFrame(np.random.randn(150, 2), columns=["f1", "f2"])
    y = pd.Series(np.random.choice([1, -1], size=150), name="label")

    model, _, _, _ = tree.fit(X, y)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps
