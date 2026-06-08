import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Tuple

def fit(X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[DecisionTreeRegressor, Dict[str, float]]:
    """
    Treina o modelo DecisionTreeRegressor utilizando a matriz X e o vetor alvo y.
    Ignora kwargs se não fornecidos (repasse para o modelo).
    As features e y devem estar alinhadas em termos de índices de tempo.
    """
    if X.empty or y.empty:
        raise ValueError("X ou y não podem ser vazios para o treinamento.")
        
    # Alinhando os índices e removendo NaNs devido a defasagens
    common_idx = X.index.intersection(y.index)
    data = pd.concat([X.loc[common_idx], y.loc[common_idx]], axis=1).dropna()
    
    if data.empty:
        raise ValueError("Sem observações válidas após o alinhamento de X e y (verifique NaNs causados por defasagens excessivas).")
        
    X_clean = data[X.columns]
    y_clean = data.iloc[:, -1]
    
    model = DecisionTreeRegressor(random_state=kwargs.pop("random_state", 42), **kwargs)
    model.fit(X_clean, y_clean)
    
    # Feature importances baseadas na impureza de Gini/MSE
    importance_dict = dict(zip(X_clean.columns, model.feature_importances_))
    
    return model, importance_dict
