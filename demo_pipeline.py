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
# ## Fase 1: Análise Causal e Seleção de Preditores (TE + Surrogates)
# A pipeline seleciona preditores por Transfer Entropy testada contra Surrogates circulares.
# O lag_consensus identifica o horizonte temporal de máxima transferência de informação.

print("\n--- Iniciando Pipeline de Seleção (TE Surrogates v5) ---")
pa = PredictorsAnalysis(prices_dict, config)
pa_result = pa.run()

# %% [markdown]
# ## Fase 2: Modelagem Preditiva e Backtest (Model)
# O modelo é um bot de trading: prevê a direção dos próximos `lag_consensus` dias.
# A validação usa dados out-of-sample com posições não-sobrepostas.

if len(pa_result.selected) == 0:
    print("\n[Aviso] Nenhum preditor foi selecionado na análise causal. O modelo não pode ser executado.")
else:
    print("\n--- Iniciando Modelagem e Validação (Backtest) ---")
    
    # Série alvo em log-retornos diários (k=1)
    target_series = log_returns(prices_dict[config.target_ticker], k=1)
    
    # O horizonte de previsão é o lag_consensus da análise de TE
    horizon = pa_result.lag_consensus
    print(f" - Horizonte de previsão: {horizon} dias (lag_consensus da análise de TE)")
    
    # Inicializa o classificador
    m = Model(
        pa_result.selected, 
        target_series, 
        config, 
        horizon=horizon,
        generated_features=pa_result.generated_features,
    )
    
    print(" - Preparando dados (features + labels forward-looking + split temporal)...")
    X_train, X_test, y_train, y_test_labels, y_test_fwd = m.build_data()
    print(f"   Treino: {len(X_train)} obs | Teste: {len(X_test)} obs (out-of-sample)")
    
    print(" - Treinando a Árvore de Decisão (GridSearchCV)...")
    m.fit(X_train, y_train)
    
    print(" - Executando Random Walk Backtest (posições não-sobrepostas)...")
    m.validate(X_test, y_test_fwd)
    
    # Relatório completo
    m_figs = m.report(predictor_analysis=pa_result)
    for name, fig in m_figs.items():
        fig.show()
        
    print("\n--- Salvando o Modelo ---")
    m.save("modelo_finalise.joblib")
    print("Modelo salvo em disco: 'modelo_finalise.joblib'")
    
    print("\n[INFO] Pipeline finalizada com sucesso.")
