"""
labeler.py
----------
Rotulador de direção da série alvo para o classificador.

Como as features em X já são deslocadas temporalmente no passado (shift(lag_tau)),
a observação X no instante t já representa informações estritamente anteriores a t.
Portanto, o rótulo y no instante t deve ser simplesmente a direção do ativo alvo 
no próprio instante t.

Se o log-retorno(t) > 0  →  classe +1 (comprar, o preço subiu)
Se o log-retorno(t) <= 0 →  classe -1 (vender, o preço caiu ou ficou igual)

Essa lógica alinha perfeitamente o objetivo da árvore de decisão com o 
P&L do random walk em validation.py (que recompensa sinal(t) * retorno(t)).
"""

import pandas as pd
import numpy as np


def label(series: pd.Series) -> pd.Series:
    """
    Rotula a série de log-retornos como +1 (alta) ou -1 (baixa).

    Parameters
    ----------
    series : pd.Series
        Série de log-retornos do ativo alvo, indexada por data.

    Returns
    -------
    pd.Series
        Série de rótulos inteiros (+1 ou -1), com o mesmo índice da entrada.
    """
    if series.empty:
        raise ValueError("A série alvo não pode estar vazia.")

    # Rótulo: +1 se o retorno é positivo, -1 caso contrário
    labels = pd.Series(
        np.where(series > 0, 1, -1),
        index=series.index,
        name="label",
        dtype=int,
    )

    return labels


__all__ = ["label"]
