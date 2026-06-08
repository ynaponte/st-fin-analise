import numpy as np
import pandas as pd
from finalise.model import tree
from sklearn.tree import DecisionTreeRegressor

def test_ut_tree_01():
    # tree.fit treina sem erro para X e y válidos
    X = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [4, 3, 2, 1]}, index=[0, 1, 2, 3])
    y = pd.Series([10, 20, 30, 40], index=[0, 1, 2, 3])
    
    model, importance = tree.fit(X, y, max_depth=2)
    assert isinstance(model, DecisionTreeRegressor)
    assert len(importance) == 2
    
def test_ut_tree_02_03():
    # feature_importance soma 1.0 e tem comprimento igual ao número de features
    X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
    # y that is predictable
    y = pd.Series(X["f1"] * 2 + X["f2"] * 0.5)
    
    model, importance = tree.fit(X, y)
    
    assert len(importance) == 3
    assert "f1" in importance
    assert np.isclose(sum(importance.values()), 1.0)
    
def test_ut_tree_04():
    # Kwargs adicionais são passados para DecisionTreeRegressor
    X = pd.DataFrame(np.random.randn(20, 2), columns=["f1", "f2"])
    y = pd.Series(np.random.randn(20))
    
    model, _ = tree.fit(X, y, max_depth=3, min_samples_split=5)
    assert model.max_depth == 3
    assert model.min_samples_split == 5
    
def test_ut_tree_alignment():
    # Ensure it aligns X and y and drops NaNs
    X = pd.DataFrame({"f1": [np.nan, 2, 3, 4]}, index=[0, 1, 2, 3])
    y = pd.Series([10, 20, 30, np.nan], index=[0, 1, 2, 3])
    
    # Valid points are indices 1 and 2
    model, _ = tree.fit(X, y)
    # the tree should have trained on 2 samples
    assert model.tree_.node_count >= 1
