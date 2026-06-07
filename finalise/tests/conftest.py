import pytest
import numpy as np
import pandas as pd
from pathlib import Path

@pytest.fixture
def white_noise():
    # ruído branco puro — H ≈ 0.5, MI ≈ 0, sem causalidade
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(1200))

@pytest.fixture
def persistent_series():
    # Fractional Gaussian noise with H=0.75 (persistent, stationary I(0)).
    # Generated via Hosking's method: Cholesky decomposition of the fGn
    # autocovariance matrix  γ(k) = 0.5*(|k-1|^(2H) - 2|k|^(2H) + |k+1|^(2H)).
    rng = np.random.default_rng(42)
    n = 1200
    H = 0.75
    gamma = np.zeros(n)
    for k in range(n):
        gamma[k] = 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))
    C = np.linalg.cholesky(
        np.array([[gamma[abs(i - j)] for j in range(n)] for i in range(n)])
    )
    return pd.Series(C @ rng.standard_normal(n))

@pytest.fixture
def antipersistent_series():
    # série revertendo à média — H < 0.5
    rng = np.random.default_rng(42)
    s = rng.standard_normal(1200)
    return pd.Series(s - 0.5 * np.roll(s, 1))

@pytest.fixture
def random_walk():
    # Passeio aleatório (I(1), não-estacionário) — para testes de raiz unitária.
    rng = np.random.default_rng(42)
    return pd.Series(np.cumsum(rng.standard_normal(1200)))

@pytest.fixture
def prices_synthetic():
    # preços sintéticos a partir de passeio aleatório geométrico
    rng = np.random.default_rng(42)
    log_ret = rng.normal(0.0002, 0.01, 1200)
    prices = 100 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=pd.bdate_range("2019-01-01", periods=1200))

@pytest.fixture(scope="session")
def prices_real():
    # fixture real cacheada — baixa uma vez por sessão
    # Usando a raiz da pasta tests do projeto
    path = Path(__file__).parent / "fixtures" / "prices.parquet"
    if path.exists():
        return pd.read_parquet(path)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    import yfinance as yf
    tickers = ["PETR4.SA", "GC=F", "DX-Y.NYB", "XOM", "CL=F"]
    df = yf.download(tickers, start="2019-01-01", end="2024-01-01")["Close"]
    df.to_parquet(path)
    return df
