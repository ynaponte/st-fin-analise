"""
labeler.py
----------
Rotulador de direção da série alvo para o classificador.

A lógica é simples e deliberadamente sem look-ahead:
  - Para cada instante t, analisamos a diferença FUTURA:
      diff(t) = retorno(t+1) - retorno(t)
  - Se diff(t) > 0  →  classe +1  (comprar: o próximo retorno será maior)
  - Se diff(t) <= 0 →  classe -1  (vender: o próximo retorno será menor/igual)

O último elemento da série não recebe rótulo (não há t+1 disponível) e é
descartado. Portanto, o índice resultante tem comprimento N-1.

A série alvo deve conter **log-retornos** (valores contínuos), não preços.
"""

import pandas as pd
import numpy as np


def label(series: pd.Series) -> pd.Series:
    """
    Rotula a série de log-retornos como +1 (compra) ou -1 (venda).

    Parameters
    ----------
    series : pd.Series
        Série de log-retornos do ativo alvo, indexada por data.

    Returns
    -------
    pd.Series
        Série de rótulos inteiros (+1 ou -1), com o mesmo índice da entrada
        exceto o último elemento (sem rótulo futuro disponível).

    Raises
    ------
    ValueError
        Se a série estiver vazia ou contiver menos de 2 elementos.
    """
    if series.empty or len(series) < 2:
        raise ValueError(
            "A série alvo deve ter pelo menos 2 elementos para gerar rótulos."
        )

    # Diferença para frente: quanto o retorno cresce no próximo período
    diff = series.shift(-1) - series

    # Descarta o último elemento (NaN gerado pelo shift(-1))
    diff = diff.iloc[:-1]

    # Rótulo: +1 se a diferença é positiva, -1 caso contrário
    labels = pd.Series(
        np.where(diff > 0, 1, -1),
        index=diff.index,
        name="label",
        dtype=int,
    )

    return labels


__all__ = ["label"]
