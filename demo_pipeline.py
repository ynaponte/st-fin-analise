# %% [markdown]
# # Demonstração do Pipeline - finalise
# Ativo alvo: Petrobras (PETR4)
# Ativos preditores candidatos: Ouro, Petróleo, Exxon Mobil e Dólar (Dxy)

# %%
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from finalise.config import Config
from finalise.workflows.predictors_analysis import PredictorsAnalysis
from finalise.analysis.transformation import log_returns
from finalise.model import Model

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
    smoothing_method="TMA",
    lag_max=30
)

print(f"Configuração carregada para {target_ticker}")

# Buscando os dados
prices_dict = config.fetch(
    tickers=[target_ticker] + predictor_tickers,
    start="2018-01-01",
    end="2026-01-01"
)

for ticker, series in prices_dict.items():
    print(f"{ticker}: {len(series)} registros, de {series.index.min().date()} a {series.index.max().date()}")

# %% [markdown]
# ## Fase 1: Análise Causal Multicritério e Seleção de Preditores
# A nova pipeline v4 unifica smoothing, log-retornos, hard gates, métricas lineares (Granger) e não-lineares (MI, TE), otimização do Lag Consensual e extração STL.

print("\n--- Iniciando Pipeline de Análise de Preditores ---")
pa = PredictorsAnalysis(prices_dict, config)
pa_result = pa.run()

# O fluxo de pa.run() exibe relatórios ricos no terminal e gera os gráficos no navegador.
# O resultado está consolidado em pa_result.

# %% [markdown]
# ## Fase 2: Modelagem Preditiva e Backtest (Model)
# Vamos treinar o modelo e fazer validação usando janela deslizante (sliding window) ou split padrão, para verificar se o modelo treinado bate uma estratégia aleatória (agentes de mercado).

if len(pa_result.selected) == 0:
    print("\n[Aviso] Nenhum preditor foi selecionado na análise causal. O modelo não pode ser executado.")
else:
    print("\n--- Iniciando Modelagem e Validação (Backtest) ---")
    
    # Extraímos a série alvo em log-retornos (k=1)
    target_series = log_returns(prices_dict[config.target_ticker], k=1)
    
    # Inicializa o classificador com os preditores que sobreviveram aos gates
    m = Model(pa_result.selected, target_series, config, horizon=1)
    
    print(" - Preparando matriz de features (X, y)...")
    X_clean, y_clean, y_returns = m.build_data()
    
    print(" - Treinando a Árvore de Decisão...")
    m.fit(X_clean, y_clean)
    
    print(" - Executando Random Walk Backtest e Calculando Métricas...")
    m.validate(X_clean, y_returns)
    
    # Exibir o relatório do modelo, incluindo a distribuição do random walk
    # Passamos pa_result para satisfazer a exibição de preditores selecionados.
    m_figs = m.report(predictor_analysis=pa_result)
    for name, fig in m_figs.items():
        fig.show()
        
    print("\n--- Salvando o Modelo ---")
    m.save("modelo_finalise.joblib")
    print("Modelo salvo em disco: 'modelo_finalise.joblib'")
    
    print("\n[INFO] Pipeline finalizada com sucesso.")
