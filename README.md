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
    Selector --> |"Rank: |H-0.5|, Max Auto-MI, Shannon"| Final(Horizonte Ótimo k*)
```

**Análises Realizadas**:
- **Retornos**: Calculados com espaçamento de tamanho `k` (sem sobreposição).
- **Estatísticas e Normalidade**: Teste de Anderson-Darling; define o `alpha` dinamicamente (0.05 se normal, 0.02 caso contrário).
- **Estacionariedade (ADF)**: Séries I(1) são descartadas.
- **Expoente de Hurst (DFA)**: Mensura previsibilidade (desvio de 0.5 indica persistência ou reversão à média).
- **Entropia e Auto-MI**: A regra de Freedman-Diaconis estima os bins. O estimador KSG k-NN calcula a Auto-Informação Mútua. Um teste de permutação avalia a significância estatística.
- **Seleção**: Classifica horizontes por distância do Hurst em relação a 0.5, valor de Auto-MI máximo e entropia de Shannon.

## Submódulo: Predictors (`src/finalise/workflows`)

O submódulo orquestra a pipeline v4 de avaliação multicritério de causalidade, selecionando candidatos baseando-se em um score composto de métricas lineares e não lineares, além de aplicar *Hard Gates* para descartar ruído precocemente.

```mermaid
flowchart TD
    A["Entrada: preços brutos<br>(alvo + N candidatos)"] --> B["Etapa 0: Smoothing<br>(preços brutos)"]
    B --> C["Etapa 1: Log-retornos (k=1)<br>dos preços suavizados"]
    
    C --> D["Etapa 2: Hard Gates<br>ADF (log-retornos)<br>Hurst (preços suavizados, ref=0.5 fixo)"]
    D --> E{"Passa?"}
    E -- Não --> DROP["❌ Descartar"]
    E -- Sim --> F

    F["Etapa 3: JSD<br>log-retornos candidato vs alvo<br>→ peso global (1 - JSD)"] --> G

    G["Etapa 4: Auto-MI do alvo<br>→ y_lags para TE<br>+ ACF/PACF do alvo"] --> H

    H["Etapa 5: Score Matrix (candidato × lag)"] --> H1
    H1["5a: Granger F-stat por lag<br>(zerado se p ≥ α)"] --> NORM
    H --> H2["5b: MI (KSG) por lag"] --> NORM
    H --> H3["5c: TE em bits<br>(só lags com MI > média)"] --> NORM

    NORM["Etapa 6: Normalizar (min-max)"] --> SCORE

    SCORE["Etapa 7: Score composto<br>S(c,l) = JSD_w(c) × [0.50·G + 0.25·MI + 0.25·TE]"]

    SCORE --> LAG["Etapa 8: Lag Consensual<br>ℓ* = argmax_l Σ_c S(c,l)"]

    LAG --> SEL["Etapa 9: Top-K no lag ℓ*"]

    SEL --> ENRICH["Etapa 10: Enriquecimento pós-seleção"]
    ENRICH --> STL["10a: STL nos preços suavizados<br>dos selecionados<br>→ features adicionais"]
    ENRICH --> JOH["10b: Johansen<br>(preços suavizados, informativo)"]
    
    STL --> DESC["Etapa 11: Descritivas<br>(relatório)"]
    JOH --> DESC
    DESC --> OUT["Saída: PipelineResult"]
```

**Análises Realizadas (Pipeline v4)**:
- **Smoothing e Log-retornos**: Aplica filtros passa-baixa antes do cálculo de log-retornos.
- **Hard Gates**: Rejeita ativamente séries não-estacionárias (via ADF) e *random walks* genuínos (onde o expoente de Hurst se concentra rigidamente em $0.5 \pm tolerância$).
- **Score Matrix e Lag Consensual**: Extrai estatísticas Granger (linear), Mutual Information e Transfer Entropy (não-linear). Multiplica pela Similaridade de Distribuição Global (1 - Divergência de Jensen-Shannon) e pontua simultaneamente múltiplos lags para descobrir o lag ótimo (Top-K) que unifica as decisões do mercado.
- **Enriquecimento Pós-Seleção**: Realiza a decomposição causal STL e verifica Spreads cointegrados via Johansen apenas para o Top-K que sobreviveu à filtragem pesada.

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
