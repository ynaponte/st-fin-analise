# Software Architecture Document — `finalise`

**Versão**: v1.0.0

---

## 1. Estrutura do Pacote

```mermaid
graph TD
    ROOT["finalise/"]

    ROOT --> ANALYZE["analyze.py\norquestrador opcional"]
    ROOT --> CONFIG["config.py"]
    ROOT --> TARGET["target/"]
    ROOT --> PREDICTORS["predictors/"]
    ROOT --> MODEL["model/"]

    TARGET --> T_INIT["__init__.py\nTargetAnalysis"]
    TARGET --> T_RET["returns.py"]
    TARGET --> T_DESC["descriptive.py"]
    TARGET --> T_STAT["stationarity.py"]
    TARGET --> T_HURST["hurst.py"]
    TARGET --> T_ENT["entropy.py"]
    TARGET --> T_ACF["autocorrelation.py"]
    TARGET --> T_SEL["selector.py"]
    TARGET --> T_REP["report.py"]

    PREDICTORS --> P_INIT["__init__.py\nPredictorAnalysis"]
    PREDICTORS --> P_RET["returns.py"]
    PREDICTORS --> P_DEC["decomposition.py"]
    PREDICTORS --> P_SMO["smoothing.py"]
    PREDICTORS --> P_MI["mi.py"]
    PREDICTORS --> P_GRA["granger.py"]
    PREDICTORS --> P_TE["transfer_entropy.py"]
    PREDICTORS --> P_COI["cointegration.py"]
    PREDICTORS --> P_SEL["selector.py"]
    PREDICTORS --> P_REP["report.py"]

    MODEL --> M_INIT["__init__.py\nModel"]
    MODEL --> M_FEAT["features.py"]
    MODEL --> M_TREE["tree.py"]
    MODEL --> M_VAL["validation.py"]
    MODEL --> M_REP["report.py"]
```

---

## 2. Arquitetura de Módulos e Dependências

```mermaid
graph LR
    CONFIG["config.py\nConfiguração global\nImutável após init"]

    subgraph TARGET["finalise.target"]
        direction TB
        TA_API["TargetAnalysis\nAPI Pública"]
        TA_INT["returns · descriptive\nstationarity · hurst\nentropy · autocorrelation\nselector · report"]
        TA_API --> TA_INT
    end

    subgraph PREDICTORS["finalise.predictors"]
        direction TB
        PA_API["PredictorAnalysis\nAPI Pública"]
        PA_INT["returns · decomposition · smoothing\nmi · granger · transfer_entropy\ncointegration · selector · report"]
        PA_API --> PA_INT
    end

    subgraph MODEL["finalise.model"]
        direction TB
        M_API["Model\nAPI Pública"]
        M_INT["features · tree\nvalidation · report"]
        M_API --> M_INT
    end

    ANALYZE["analyze()\norquestrador"]

    CONFIG -->|injeção| TARGET
    CONFIG -->|injeção| PREDICTORS
    CONFIG -->|injeção| MODEL

    TARGET -->|"ta.horizon\nta.alpha\nta.results"| PREDICTORS
    PREDICTORS -->|"pa.selected\n(candidatos + τ*)"| MODEL

    ANALYZE --> TARGET
    ANALYZE --> PREDICTORS
    ANALYZE --> MODEL

    YF["yfinance"]:::ext
    RICH["rich"]:::ext
    PLOTLY["plotly"]:::ext
    SKL["sklearn"]:::ext

    YF --> CONFIG
    RICH --> TARGET
    RICH --> PREDICTORS
    RICH --> MODEL
    PLOTLY --> TARGET
    PLOTLY --> PREDICTORS
    PLOTLY --> MODEL
    SKL --> MODEL

    classDef ext fill:none,stroke-dasharray:4
```

---

## 3. Fluxo de Dados Global

```mermaid
flowchart TD
    RAW["Preços brutos\nP(t) — via yfinance"]

    RAW --> CONFIG_FETCH["config.fetch(tickers, start, end)"]
    CONFIG_FETCH --> PRICES["prices_dict\n{ticker: Series}"]

    PRICES --> TARGET_RUN["TargetAnalysis.run()"]
    PRICES --> PA_RUN["PredictorAnalysis.run()"]

    TARGET_RUN --> HORIZON["ta.horizon: k ∈ {1,5,21}\nta.alpha: 0.05 | 0.02"]
    HORIZON --> PA_RUN

    PA_RUN --> SELECTED["pa.selected\n[{série, τ*, tipo, componente}]"]
    SELECTED --> MODEL_RUN["Model.run()"]

    PRICES -->|"preços brutos\npara cointegração"| PA_RUN

    MODEL_RUN --> RESULT["result\n{is_outlier, z_score,\nfeature_importance}"]

    TARGET_RUN --> REP_T["target.report()"]
    PA_RUN --> REP_P["predictors.report()"]
    MODEL_RUN --> REP_M["model.report()"]
```

---

## 4. Diagrama de Execução — `finalise.target`

```mermaid
flowchart TD
    START(["TargetAnalysis.run()"])

    START --> FETCH["returns.compute(prices, k)\npara k ∈ {1, 5, 21}\nnon-overlapping"]

    FETCH --> DESC["descriptive.stats(series)\nmédia · dp · skewness · curtose\nAnderson-Darling"]

    DESC --> NORM{{"is_normal?"}}
    NORM -->|"sim"| ALPHA5["α = 0.05"]
    NORM -->|"não"| ALPHA2["α = 0.02\n⚠ rich warning"]

    ALPHA5 --> ADF
    ALPHA2 --> ADF

    ADF["stationarity.adf(series)\npor escala"]
    ADF --> STATCHECK{{"estacionária?"}}
    STATCHECK -->|"não"| EXCLUDE["excluir escala\n⚠ rich warning"]
    STATCHECK -->|"sim"| HURST

    HURST["hurst.dfa(series)\nH e |H − 0.5| por escala"]
    HURST --> SHANNON["entropy.shannon(series)\nbins: Freedman-Diaconis"]
    SHANNON --> AUTOMI["entropy.auto_mi(series, lag_max)\nKraskov k-NN\nperfil MI(lag)"]
    AUTOMI --> ACFPACF["autocorrelation.acf_pacf(series)\nr_k e r_k²"]

    ACFPACF --> SEL["selector.choose_horizon(results)\nranking: |H−0.5| · auto-MI · entropia"]

    SEL --> GATE{{"sinal detectado?\n|H−0.5| > 0.05\nou auto-MI sig."}}
    GATE -->|"não"| REPORT_STOP["report.generate()\n⚠ sem sinal — pipeline encerra\nou continua forçado"]
    GATE -->|"sim"| HORIZON["ta.horizon = k*\nta.alpha = α vigente"]
    HORIZON --> REPORT_OK["report.generate()\nresumo + gráficos plotly"]
```

---

## 5. Diagrama de Execução — `finalise.predictors`

```mermaid
flowchart TD
    START(["PredictorAnalysis.run()"])

    START --> COINTE["cointegration.engle_granger(\n  price_alvo, price_preditor\n)\nsobre preços brutos I(1)"]

    COINTE --> RETS["returns.compute(prices, k*)\nnon-overlapping\nk* herdado de ta.horizon"]

    RETS --> EXPAND["Expansão de candidatos por preditor X"]

    EXPAND --> BRUTO["X_bruto\nr_k(t)"]
    EXPAND --> SUAV["X_suavizado\nsmoothing.apply(X, window, method)"]
    EXPAND --> STL["decomposition.stl(X, period)\nSTL sobre r_k(t)"]

    STL --> TREND["X_tendência"]
    STL --> SEAS["X_sazonalidade"]
    STL --> RESID["X_resíduo"]

    BRUTO & SUAV & TREND & SEAS & RESID --> PIPELINE

    subgraph PIPELINE["Pipeline de seleção — por candidato c"]
        direction TB
        MI["mi.cross_mi_lags(c, alvo, lag_max)\nKraskov k-NN\nτ* = argmax MI(τ)\nBonferroni: threshold = α / lag_max"]

        MI --> MISIG{{"MI(τ*) significativa?"}}
        MISIG -->|"não"| DISCARD1["descartar c"]
        MISIG -->|"sim"| GRANGER

        GRANGER["granger.test(c, alvo, lag=τ*)\np-valor ao nível α"]
        GRANGER --> GSIG{{"rejeita H0?"}}
        GSIG -->|"sim"| INCLUDE_L["incluir c\ntipo: linear"]
        GSIG -->|"não"| TE

        TE["transfer_entropy.compute(c, alvo, lag=τ*)\nbins: Freedman-Diaconis\nteste de permutação N=500"]
        TE --> TESIG{{"TE significativa?\n> percentil (1−α) da nula"}}
        TESIG -->|"sim"| INCLUDE_NL["incluir c\ntipo: não-linear"]
        TESIG -->|"não"| DISCARD2["descartar c"]
    end

    INCLUDE_L & INCLUDE_NL --> SELECTED["pa.selected\n[{série, τ*, tipo, componente}]"]
    SELECTED --> REPORT["report.generate()\nresumo + gráficos plotly"]
```

---

## 6. Diagrama de Execução — `finalise.model`

```mermaid
flowchart TD
    START(["Model.run()"])

    START --> BUILD["features.build(pa.selected)\nmatriz X: {candidato_i(t − τ*_i)}\ngarantia causal: sem look-ahead"]

    BUILD --> TARGET_VEC["target: r_k*(t+h)\nlog-retorno bruto do alvo\nno horizonte selecionado"]

    TARGET_VEC --> TRAIN["tree.fit(X, y)\nsklearn.DecisionTreeRegressor"]

    TRAIN --> BACKTEST["validation.random_walk_backtest(\n  model, X, y,\n  n_agents, seed\n)"]

    BACKTEST --> AGENTS["N agentes com estratégias\naleatórias de trading\nsem acesso às séries explicativas"]

    AGENTS --> DIST["distribuição de retornos\nacumulados dos agentes\nμ_agents · σ_agents"]

    DIST --> ZSCORE["z_score = (ret_modelo − μ) / σ"]

    ZSCORE --> GATE{{"z_score > 2.5?"}}
    GATE -->|"sim"| OUTLIER["is_outlier = True\nmodelo validado"]
    GATE -->|"não"| NOT_OUTLIER["is_outlier = False\nmodelo não supera agentes"]

    OUTLIER & NOT_OUTLIER --> IMPORTANCE["feature_importance\nimpurity-based"]

    IMPORTANCE --> REPORT["report.generate()\ndistribuição agentes + modelo\ncurva retorno acumulado\nimportância de features\nrich + plotly"]
```

---

## 7. Diagrama de Execução — Orquestrador `analyze()`

```mermaid
flowchart TD
    START(["analyze(prices_dict, config)"])

    START --> PHASE1["TargetAnalysis(prices, config).run()"]

    PHASE1 --> GATE1{{"ta.horizon identificado?"}}
    GATE1 -->|"não\nconfig.force=False"| ABORT["encerra\nreport diagnóstico via rich"]
    GATE1 -->|"não\nconfig.force=True"| WARN["⚠ warning via rich\ncontinua com melhor escala disponível"]
    GATE1 -->|"sim"| PHASE2

    WARN --> PHASE2
    PHASE2["PredictorAnalysis(\n  prices_dict,\n  ta.results[ta.horizon],\n  config, alpha=ta.alpha\n).run()"]

    PHASE2 --> GATE2{{"pa.selected não vazio?"}}
    GATE2 -->|"não"| ABORT2["encerra\nreport: nenhum preditor selecionado"]
    GATE2 -->|"sim"| PHASE3

    PHASE3["Model(\n  pa.selected,\n  target_series,\n  config\n).run()"]

    PHASE3 --> RESULT["AnalysisResult\n.target → TargetAnalysis\n.predictors → PredictorAnalysis\n.model → Model"]
```

---

## 8. Contrato de Dados entre Módulos

```mermaid
classDiagram
    class Config {
        +str target_ticker
        +list~str~ predictor_tickers
        +list~int~ horizons
        +int smoothing_window
        +str smoothing_method
        +int stl_period
        +float alpha
        +int n_permutations
        +int knn_k
        +int lag_max
        +bool force_continue
        +fetch(tickers, start, end) prices_dict
    }

    class TargetResult {
        +int horizon
        +float alpha
        +dict results
        +run()
        +report()
    }

    class SelectedCandidate {
        +str ticker
        +str component
        +Series series
        +int lag_tau
        +str relation_type
        +float mi_value
        +float te_value
        +float granger_pvalue
    }

    class PredictorResult {
        +list~SelectedCandidate~ selected
        +dict cointegration
        +dict results
        +run()
        +report()
    }

    class ModelResult {
        +bool is_outlier
        +float z_score
        +float model_return
        +dict agent_returns
        +dict feature_importance
        +run()
        +report()
    }

    Config --> TargetResult : injeção
    Config --> PredictorResult : injeção
    Config --> ModelResult : injeção
    TargetResult --> PredictorResult : horizon · alpha
    PredictorResult --> ModelResult : selected~SelectedCandidate~
```
