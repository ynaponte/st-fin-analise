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
# A pipeline metodológica implementa um arcabouço rigoroso para detecção de direcionalidade da informação:
# 1. **Transformação e Smoothing**: Aplicação de Filtros Passa-Baixa otimizados para atenuação do ruído estocástico e obtenção de log-retornos ($R_t = \ln(P_t/P_{t-1})$).
# 2. **Auto-MI**: Estimação da Informação Mútua Autoregressiva para determinação empírica da dimensão de embutimento (embedding dimension, $y_{lags}$) da série alvo.
# 3. **Surrogates Circulares**: Geração de permutações circulares (Surrogates) mantendo a Função Densidade de Probabilidade (PDF) univariada, para teste de robustez da hipótese nula $H_0$ (ausência de causalidade direcional).
# 4. **Transfer Entropy (TE)**: Medição assimétrica de causalidade de Shannon ($T_{X \to Y}(\tau) = H(Y_t \mid Y_{t-1:t-y_{lags}}) - H(Y_t \mid Y_{t-1:t-y_{lags}}, X_{t-\tau})$). A Transfer Entropy observada é testada contra a distribuição dos Surrogates e filtrada estatisticamente ($p$-valor < $\alpha$), paralelamente com Causalidade Linear de Granger.
# 5. **Lag Endógeno e Seleção**: Maximização transversal da soma de TE informacional para determinar o $\tau$ endógeno (horizonte reativo do sistema como um todo), resultando no subconjunto de top-K preditores ótimo no sistema complexo.
# 6. **Enriquecimento Estrutural**: Decomposição em Seasonal and Trend decomposition using Loess (STL) avaliando cointegração de Johansen e causalidade inter-séries em componentes lentas (tendência) e rápidas (ruído). Geração das derivadas para retroalimentação na modelagem.

print("\n--- Iniciando Pipeline de Seleção (TE Surrogates v5) ---")
pa = PredictorsAnalysis(prices_dict, config)
pa_result = pa.run()

# %% [markdown]
# ## Fase 2: Modelagem Preditiva e Backtest (Model)
# O paradigma preditivo assenta-se na previsão do direcionamento estocástico (sinal) dos ativos.
# Para manter rigor sobre data snooping e look-ahead bias:
# - Avaliação "Walk-Forward" adotando backtest *out-of-sample* com partições temporais estritas.
# - Operações não-sobrepostas: A predição atua no horizonte temporal ótimo de $t \to t+\tau$, onde $\tau$ é o `lag_consensus` global.
# - Target binário ($y_t \in \{-1, 1\}$) sobre a expectativa direcional do log-retorno do ativo alvo no horizonte avaliado.
# - Aplica-se otimização do classificador (via `GridSearchCV` sobre Decision Tree) parametrizado pelas métricas dinâmicas oriundas do TE-pipeline.

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
