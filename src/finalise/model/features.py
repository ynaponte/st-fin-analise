"""
features.py
-----------
Construção da matriz de features X a partir dos candidatos selecionados.

As features representam os valores CORRENTES dos preditores em cada instante t.
Não há deslocamento temporal (shift) aplicado aqui — o alinhamento causal é
garantido pelo labeler, que rotula com base no retorno FUTURO do alvo
(forward-looking labels com horizon = lag_consensus).

Esquema temporal:
    Features em t  = X(t)  → valores observáveis no instante t
    Label em t     = sign(retorno do alvo de t+1 a t+horizon)  → futuro

Não há look-ahead bias porque:
    - As features usam apenas informação presente/passada
    - O alvo (label) é futuro, exatamente o que queremos prever
"""

import pandas as pd
from typing import List
from finalise.workflows.predictors_analysis import SelectedCandidate


def build(selected: List[SelectedCandidate]) -> pd.DataFrame:
    """
    Constrói a matriz de features X a partir dos candidatos selecionados.

    Para cada candidato, a feature é a sua série temporal no instante t
    (sem deslocamento). O lag_tau permanece armazenado no SelectedCandidate
    para fins informativos (horizonte de previsão), mas não é usado para
    shift nas features.

    Parameters
    ----------
    selected : list of SelectedCandidate
        Preditores selecionados pela PredictorsAnalysis.

    Returns
    -------
    pd.DataFrame
        Matriz de features, indexada por data.
    """
    if not selected:
        return pd.DataFrame()

    features = {}
    for cand in selected:
        col_name = f"{cand.ticker}_{cand.component}"
        features[col_name] = cand.series

    X = pd.DataFrame(features)
    return X


__all__ = ["build"]
