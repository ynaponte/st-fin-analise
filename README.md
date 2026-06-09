# Finalise

## Descrição do Módulo
O módulo **Finalise** é uma pipeline analítica e preditiva causal desenvolvida em Python para séries temporais financeiras. O sistema orquestra a análise de um ativo alvo, a seleção rigorosa de preditores baseada em causalidade (linear e não-linear) e a modelagem preditiva, garantindo a ausência de viés de antecipação (*look-ahead bias*). O fluxo é gerido pelo orquestrador principal e dividido em três submódulos fundamentais: `target`, `predictors` e `model`.

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

A pipeline é executada a partir de `analyze.py`, que encadeia a execução sequencial das três fases principais. O processo é parametrizado por um objeto `Config` compartilhado.

```mermaid
flowchart TD
    Config[Configuração Inicial] --> TA[Fase 1: Target Analysis]
    Prices[(Séries de Preços)] --> TA
    TA --> |Horizonte Ótimo k*| PA[Fase 2: Predictor Analysis]
    Prices --> PA
    PA --> |Preditores Selecionados| Model[Fase 3: Treinamento e Validação]
    TA -.-> |Série Alvo no Horizonte k*| PA
    PA -.-> |Matriz de Features X Deslocada| Model
    Model --> Report[Resumo Executivo e Gráficos]
```

## Submódulo: Target (`src/finalise/target`)

O submódulo `target` foca em analisar a série temporal do ativo alvo em diferentes horizontes temporais (escalas) e eleger a escala `k` que apresenta as melhores propriedades preditivas.

```mermaid
flowchart TD
    Start[Série de Preços do Alvo] --> Horizontes(Iteração por Horizontes k)
    
    subgraph Iteração por Horizonte
        Horizontes --> R[Retornos Logarítmicos Não-Sobrepostos]
        R --> Desc[Estatísticas Descritivas e Normalidade AD]
        Desc --> ADF{Estacionário? ADF}
        ADF -- Sim --> H[Hurst Exponent via DFA]
        ADF -- Não --> Discard[Descartar Horizonte]
        H --> Ent[Entropia de Shannon via Bins F-D]
        Ent --> AutoMI[Auto-Informação Mútua KSG]
        AutoMI --> Perm[Teste de Permutação Phipson-Smyth]
        Perm --> ACF[ACF/PACF]
    end
    
    ACF --> Selector[Seletor de Horizonte]
    Selector --> |Rank: |H-0.5|, Max Auto-MI, Shannon| Final(Horizonte Ótimo k*)
```

**Análises Realizadas**:
- **Retornos**: Calculados com espaçamento de tamanho `k` (sem sobreposição).
- **Estatísticas e Normalidade**: Teste de Anderson-Darling; define o `alpha` dinamicamente (0.05 se normal, 0.02 caso contrário).
- **Estacionariedade (ADF)**: Séries I(1) são descartadas.
- **Expoente de Hurst (DFA)**: Mensura previsibilidade (desvio de 0.5 indica persistência ou reversão à média).
- **Entropia e Auto-MI**: A regra de Freedman-Diaconis estima os bins. O estimador KSG k-NN calcula a Auto-Informação Mútua. Um teste de permutação avalia a significância estatística.
- **Seleção**: Classifica horizontes por distância do Hurst em relação a 0.5, valor de Auto-MI máximo e entropia de Shannon.

## Submódulo: Predictors (`src/finalise/predictors`)

O submódulo `predictors` realiza a extração de características (features) causais de cada ativo preditor candidato e emprega uma árvore de decisão para aceitá-los baseando-se em previsibilidade linear ou não-linear.

```mermaid
flowchart TD
    Prices[(Preços Brutos)] --> Coint[Teste Engle-Granger Cointegração]
    Prices --> CandGen(Geração de Candidatos)
    
    subgraph Geração de Candidatos
        CandGen --> Raw[Retorno Diário k=1]
        CandGen --> Smooth[Suavização Causal SMA/EMA/DEMA/Gauss]
        CandGen --> STL[Decomposição STL Causal Trend/Seas/Resid]
    end
    
    Raw --> Granger{Teste de Granger\nc/ correção FDR}
    Smooth --> Granger
    
    Granger -- Significativo --> SelLin[Selecionado como Linear]
    Granger -- Falha --> CMI{Cross-MI k-NN}
    STL --> CMI
    
    CMI -- Significativo --> TE{Transfer Entropy\nPermutação}
    CMI -- Falha --> Desc[Descartado]
    
    TE -- Significativo --> SelNL[Selecionado como Não-Linear]
    TE -- Falha --> Desc
    
    SelLin --> Dedup[Remoção de Duplicatas intra-ticker]
    SelNL --> Dedup
    Dedup --> Final(Candidatos Selecionados com lag_tau)
```

**Análises Realizadas**:
- **Geração Causal**: Retornos brutos (k=1), filtragem passa-baixa sem viés de antecipação e decomposição `STL` reajustada iterativamente (*step-by-step*) com *forward-fill*.
- **Causalidade Linear (Granger)**: Aplicado em dados diários estacionários. Corrige os p-valores ao longo de múltiplos lags via FDR de Benjamini-Hochberg.
- **Informação Mútua Cruzada (MI) e Transfer Entropy (TE)**: Representa o fluxo não-linear. Utiliza teste de permutação estatística máxima no Cross-MI. A Transfer Entropy é avaliada condicionando o ativo ao histórico ótimo do alvo (lag de auto-MI da Fase 1). O preditor é eleito não-linear em caso de significância estatística robusta na TE.

## Submódulo: Model (`src/finalise/model`)

O submódulo `model` alinha estritamente as características preditivas de forma atrasada e providencia o aprendizado de máquina e validação simulada.

```mermaid
flowchart TD
    Cands(Candidatos Selecionados) --> Build[Alinhamento de Dados]
    Target(Série Alvo k*) --> Build
    
    subgraph Engenharia de Features
        Build --> Shift[Shift t-lag_tau para cada feature]
        Shift --> DropNA[Remoção de NaNs e Interseção de Índices]
    end
    
    DropNA --> Fit[Treinamento DecisionTreeRegressor]
    Fit --> FeatImp[Importância de Features Gini/MSE]
    Fit --> Val[Validação Random Walk Backtest]
    
    subgraph Validação
        Val --> Pred[Sinal do Modelo: sign]
        Pred --> Ret[Multiplicação pelo Retorno Real k*]
        Ret --> Sim[Simulação de 1000 Agentes Aleatórios]
        Sim --> ZScore{Z-Score > 2.5?}
        ZScore -- Sim --> Outlier[Modelo Válido: Outlier Estatístico]
        ZScore -- Não --> Inv[Modelo Inválido / Ruído]
    end
```

**Análises Realizadas**:
- **Engenharia (`features.py`)**: Utiliza os lags ótimos `lag_tau` extraídos na fase de preditores aplicando estrito descolamento (`shift`) causal, impedindo fuga de informações futuras.
- **Machine Learning (`tree.py`)**: Ajusta um `DecisionTreeRegressor` e mensura os pesos de cada variável.
- **Validação Tipo Random Walk (`validation.py`)**: Gera empiricamente os retornos que seriam obtidos por 1000 agentes operando a esmo. Ao calcular a pontuação `z-score` e estipular um nível crítico (`> 2.5`), comprova a genuinidade da taxa de acerto do modelo preditivo.
