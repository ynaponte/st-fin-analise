import pandas as pd
from typing import List, Tuple
from finalise.workflows.predictors_analysis import SelectedCandidate

def build(selected: List[SelectedCandidate]) -> pd.DataFrame:
    """
    Constrói a matriz de features X a partir dos candidatos selecionados.
    Para cada candidato, a feature correspondente é a sua série temporal 
    deslocada pelo seu atraso ótimo (lag_tau).
    Garante que não há look-ahead bias (violação causal), pois o valor em t 
    utiliza apenas informações de t - lag_tau.
    """
    if not selected:
        # Retorna DataFrame vazio se não houver candidatos
        return pd.DataFrame()
        
    features = {}
    for cand in selected:
        col_name = f"{cand.ticker}_{cand.component}_lag{cand.lag_tau}"
        # Shift data forward by lag_tau, so value at t is series[t - lag_tau]
        features[col_name] = cand.series.shift(cand.lag_tau)
        
    # Concat along columns
    X = pd.DataFrame(features)
    
    return X
