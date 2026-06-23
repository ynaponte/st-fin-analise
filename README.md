# Finalise

## Descrição do Módulo
O módulo **Finalise** é uma pipeline analítica e preditiva causal desenvolvida em Python para séries temporais financeiras. O sistema orquestra a seleção rigorosa de preditores baseada em Transfer Entropy (TE) com teste de significância por surrogates circulares, seguida de modelagem preditiva via árvore de decisão configurada como bot de trading. O fluxo garante a ausência de viés de antecipação (*look-ahead bias*) e validação out-of-sample com split temporal.

## Instalação
O projeto utiliza o gerenciador de pacotes `uv` e requer Python >= 3.12. 
As dependências essenciais incluem `numpy`, `pandas`, `plotly`, `scipy`, `statsmodels`, `scikit-learn`, `rich`, entre outras.

Para instalar e sincronizar o ambiente, execute na raiz do projeto:
```bash
uv sync
```
Alternativamente, usando `pip` para instalar o pacote no seu ambiente Python atual de forma editável:
```bash
pip install -e .
```

## Lógica Geral de Análise (Pipeline Finalise)

A pipeline é executada a partir de `demo_pipeline.py`, que encadeia a execução sequencial das duas fases principais. O processo é parametrizado por um objeto `Config` compartilhado.

```mermaid
flowchart TD
    Config[Configuração Inicial] --> PA[Fase 1: Seleção de Preditores]
    Prices[(Séries de Preços)] --> PA
    PA --> |Preditores + lag_consensus| Model[Fase 2: Treinamento e Validação]
    PA -.-> |Features derivadas| Model
    Model --> Report[Relatório com Métricas Comparativas]
```

## Submódulo: Predictors (`src/finalise/workflows`)

Pipeline v5 de seleção de preditores baseada em Transfer Entropy com teste de significância por surrogates circulares.

```mermaid
flowchart TD
    A["Entrada: preços brutos<br>(alvo + N candidatos)"] --> B["Etapa 0: Smoothing<br>(preços brutos)"]
    B --> C["Etapa 1: Log-retornos (k=1)<br>dos preços suavizados"]
    
    C --> D["Etapa 2: Preparação<br>(filtra dados insuficientes)"]
    D --> E["Etapa 3: Auto-MI do alvo<br>→ y_lags (embedding dim)"]
    
    E --> F["Etapa 4: Surrogates Circulares<br>(seed única por candidato)"]
    
    F --> G["Etapa 5: Matriz de TE Significativa<br>(candidato × lag)<br>TE retida se p < α OU Granger causal"]
    
    G --> H["Etapa 6: Lag Endógeno<br>ℓ* = argmax_l Σ_c TE(c,l)"]
    
    H --> I["Etapa 7: Top-K no lag ℓ*"]
    
    I --> J["Etapa 8: Enriquecimento<br>STL + Johansen + TE componentes"]
    
    J --> K["Etapa 9: Feature Engineering<br>+ Validação por Surrogates"]
    
    K --> OUT["Saída: PipelineResult<br>(selected, lag_consensus, features)"]
```

**Análises Realizadas (Pipeline v5)**:
- **Smoothing e Log-retornos**: Aplica filtros passa-baixa antes do cálculo de log-retornos diários.
- **Transfer Entropy com Surrogates**: Para cada candidato e cada lag, a TE observada é comparada contra uma distribuição nula gerada por deslocamento circular. O p-valor determina significância. Granger causality serve como critério complementar.
- **Lag Consensual**: O lag que maximiza a soma total de TE significativa entre todos os candidatos.
- **Enriquecimento Pós-Seleção**: Decomposição STL, cointegração de Johansen, e features derivadas (rolling stats) validadas individualmente por surrogates.

## Submódulo: Model (`src/finalise/model`)

O submódulo `model` implementa um **bot de trading** baseado em árvore de decisão. Dado que a análise de TE identificou o `lag_consensus` como horizonte de previsibilidade, o modelo decide: *"dado o estado atual dos preditores, devo comprar ou vender, para fechar a posição em `lag_consensus` dias?"*

```mermaid
flowchart TD
    Cands(Candidatos Selecionados) --> Build[Construção de Features]
    Target("Série Alvo (log-retornos k=1)") --> Build
    
    subgraph "Alinhamento Causal"
        Build --> Feat["Features X(t) = valores correntes<br>(sem shift)"]
        Build --> Label["Labels y(t) = sign(retorno forward<br>nos próximos lag_consensus dias)"]
        Build --> FwdRet["y_returns(t) = retorno acumulado<br>forward de lag_consensus dias"]
    end
    
    subgraph "Split Temporal"
        Feat --> Split["80% Treino | Gap de Purge | 20% Teste"]
        Label --> Split
        FwdRet --> Split
    end
    
    Split --> Fit["Treino: GridSearchCV<br>DecisionTreeClassifier<br>(shuffle nos folds internos)"]
    
    Split --> Val["Validação: Backtest Out-of-Sample<br>(posições não-sobrepostas)"]
    
    Fit --> Val
    
    subgraph "Backtest e Métricas"
        Val --> Sig["Sinal a cada horizon dias"]
        Sig --> PnL["P&L = sinal × retorno forward"]
        PnL --> Comp["Comparação:<br>• Modelo vs 1000 Agentes Aleatórios<br>• vs Buy & Hold<br>• vs Retorno Perfeito"]
        Comp --> Metrics["Win Rate, Sharpe, Drawdown,<br>Eficiência, Z-Score"]
    end
    
    Metrics --> Report["Relatório Rich + Plotly"]
```

**Destaques**:
- **Sem Look-Ahead Bias**: Features usam valores correntes dos preditores; labels são forward-looking. A causalidade é garantida pelo design.
- **Split Temporal com Purge**: Primeiros 80% para treino, gap de `horizon` observações, últimos 20% para teste. O modelo nunca vê dados de teste.
- **Posições Não-Sobrepostas**: No backtest, uma decisão é tomada a cada `horizon` dias, evitando correlação entre trades consecutivos.
- **Benchmarks Completos**: O modelo é comparado contra agentes aleatórios (z-score), buy-and-hold, e retorno perfeito (perfect foresight).
- **Métricas de Trading**: Win rate (total e por classe), Sharpe ratio anualizado, max drawdown, eficiência vs perfeito, retorno anualizado.
