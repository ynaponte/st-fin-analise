"""
workflows.te_surrogates
-----------------------
Geração de surrogates por deslocamento circular e teste de significância
para a Transfer Entropy (TE).

O deslocamento circular (circular shift) quebra a sincronia temporal entre X e Y,
mas preserva perfeitamente a autocorrelação e a distribuição interna de X.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from joblib import Parallel, delayed

from finalise.analysis.entropy import transfer_entropy


def generate_circular_surrogates(
    x: pd.Series, n_surrogates: int = 100, seed: int = 42
) -> np.ndarray:
    """
    Gera matriz de surrogates deslocando circularmente a série `x`.

    Parameters
    ----------
    x : pd.Series
        Série temporal original.
    n_surrogates : int, opcional
        Quantidade de surrogates a gerar.
    seed : int, opcional
        Semente aleatória.

    Returns
    -------
    np.ndarray
        Matriz de shape (n_surrogates, n_samples) com os surrogates.
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    if n < 2:
        raise ValueError("Série muito curta para gerar surrogates.")

    # Sorteia deslocamentos únicos (ou com repetição se n_surrogates > n-1)
    # entre 1 e n-1 para não permitir shift = 0.
    replace = n_surrogates >= n
    shifts = rng.choice(np.arange(1, n), size=n_surrogates, replace=replace)

    arr = x.values
    surrogates = np.empty((n_surrogates, n))

    for i, shift in enumerate(shifts):
        surrogates[i, :] = np.roll(arr, shift)

    return surrogates


def test_te_significance(
    x: pd.Series,
    y: pd.Series,
    surrogates: np.ndarray,
    lag: int,
    y_lags: int = 1,
    n_jobs: int = -1,
) -> Tuple[float, float]:
    """
    Computa a TE observada e o p-valor com base na distribuição dos surrogates.

    Parameters
    ----------
    x : pd.Series
        Série previsora (original).
    y : pd.Series
        Série alvo.
    surrogates : np.ndarray
        Matriz de surrogates (n_surrogates, n_samples) gerada previamente.
    lag : int
        Lag para testar (X_t-lag -> Y_t).
    y_lags : int, opcional
        Embedding dimension do alvo (y_lags).
    n_jobs : int, opcional
        Número de jobs para processamento paralelo.

    Returns
    -------
    Tuple[float, float]
        (te_observada, p_valor).
    """
    # 1. TE Observada
    te_obs = transfer_entropy(x, y, lag=lag, y_lags=y_lags)

    # 2. Função auxiliar para a TE do surrogate
    def _calc_te(surr_vals: np.ndarray) -> float:
        surr_series = pd.Series(surr_vals, index=x.index)
        try:
            return transfer_entropy(surr_series, y, lag=lag, y_lags=y_lags)
        except Exception:
            return 0.0

    # 3. TE dos Surrogates
    n_surr = surrogates.shape[0]
    te_surr = Parallel(n_jobs=n_jobs)(
        delayed(_calc_te)(surrogates[i]) for i in range(n_surr)
    )

    # 4. Cálculo do P-valor (one-sided: qual proporção de surrogates superou o TE original?)
    te_surr_arr = np.array(te_surr)
    p_val = np.sum(te_surr_arr >= te_obs) / n_surr

    return float(te_obs), float(p_val)
