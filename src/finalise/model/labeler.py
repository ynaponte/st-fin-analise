"""
labeler.py
----------
Rotulador de direção da série alvo para o classificador.

O rótulo indica a *direção futura* do ativo ao longo do horizonte de previsão
(`horizon`, tipicamente = lag_consensus da análise de TE).

Para horizon = 1:
    label(t) = sign(r(t+1))

Para horizon > 1:
    forward_return(t) = r(t+1) + r(t+2) + ... + r(t+horizon)
                      = log(P(t+horizon) / P(t))
    label(t) = sign(forward_return(t))

Isso garante que o modelo é treinado para prever exatamente o que o
bot de trading precisa saber: "devo comprar ou vender AGORA, para fechar
a posição em `horizon` dias?"

NOTA: os últimos `horizon` elementos terão rótulo NaN (futuro indisponível)
e devem ser descartados antes do treino.
"""

import pandas as pd
import numpy as np


def label(series: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Rotula a série de log-retornos como +1 (compra) ou -1 (venda)
    baseando-se na direção do retorno acumulado nos próximos `horizon` dias.
    """
    if series.empty:
        raise ValueError("A série alvo não pode estar vazia.")
    if horizon < 1:
        raise ValueError(f"Horizonte deve ser >= 1, recebido: {horizon}")

    if horizon == 1:
        forward_return = series.shift(-1)
    else:
        cumret = series.cumsum()
        forward_return = cumret.shift(-horizon) - cumret

    labels = pd.Series(
        np.where(forward_return > 0, 1, -1),
        index=series.index,
        name="label",
        dtype=float,
    )
    labels.iloc[-horizon:] = np.nan
    return labels


def forward_returns(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Calcula o retorno acumulado forward para o horizonte."""
    if horizon == 1:
        fwd = series.shift(-1)
    else:
        cumret = series.cumsum()
        fwd = cumret.shift(-horizon) - cumret

    fwd.iloc[-horizon:] = np.nan
    fwd.name = "forward_return"
    return fwd


def next_day_return(series: pd.Series) -> pd.Series:
    """
    Retorna o retorno logarítmico do dia seguinte: r(t+1).
    Usado para o backtest de rebalanceamento diário.
    """
    nxt = series.shift(-1)
    nxt.name = "next_day_return"
    return nxt


__all__ = ["label", "forward_returns", "next_day_return"]
