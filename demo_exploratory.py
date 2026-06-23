# %% [markdown]
# # Demonstração Exploratória - finalise
# Ativo alvo: Petrobras (PETR4)
# Ativos preditores candidatos: Ouro, Petróleo, Exxon Mobil e Dólar (Dxy)
#
# Este script roda apenas a Análise Exploratória, que serve para identificar
# de forma factual propriedades estatísticas e correlações (lineares e não-lineares)
# iniciais das séries financeiras, servindo como uma base fundamental para análise empírica,
# sem exercer papel de filtro no pipeline preditivo.
# 
# A análise avalia as seguintes métricas fundamentais:
# - Normalidade: Estatística de Anderson-Darling, avaliando a aderência à distribuição $N(\mu, \sigma^2)$.
# - Estacionariedade: Teste de Dickey-Fuller Aumentado (ADF).
# - Persistência/Memória de Longo Prazo: Expoente de Hurst ($H$), onde $H \approx 0.5$ denota passeio aleatório (Random Walk).
# - Divergência da Distribuição de Informação: Jensen-Shannon Divergence ($JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$).
# - Informação Mútua (MI): $MI(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right)$, estimada via o estimador Kraskov-Stögbauer-Grassberger (KSG).
# - Causalidade Linear de Granger: Teste de hipótese para a predição VAR(p), quantificando precedência temporal linear.

# %%
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from finalise.config import Config
from finalise.workflows.exploratory_analysis import ExploratoryAnalysis

# Definindo a configuração inicial
target_ticker = "PETR4.SA"
predictor_tickers = [
    "GC=F",       # Ouro
    "BZ=F",       # Petróleo Brent
    "XOM",        # Exxon Mobil
    "DX-Y.NYB",   # Dólar Index (DXY)
]

config = Config(
    target_ticker=target_ticker,
    predictor_tickers=predictor_tickers,
    smoothing_method="TMA",  # Não utilizado para retornos na fase exploratória, apenas como config
    lag_max=30
)

print(f"Configuração carregada para {target_ticker}")

# Buscando os dados brutos
prices_dict = config.fetch(
    tickers=[target_ticker] + predictor_tickers,
    start="2018-01-01",
    end="2026-01-01"
)

for ticker, series in prices_dict.items():
    print(f"{ticker}: {len(series)} registros, de {series.index.min().date()} a {series.index.max().date()}")

# %% [markdown]
# ## Fase 0: Relatório Exploratório (Informativo)
# Analisa Normalidade, Estacionariedade, Hurst, Correlações e métricas de Informação (MI, JSD, Granger).

print("\n--- Iniciando Análise Exploratória Factual ---")
ea = ExploratoryAnalysis(prices_dict, config)
ea.run()

print("\n[INFO] Análise Exploratória finalizada. Execute 'demo_pipeline.py' para a seleção e modelagem.")
