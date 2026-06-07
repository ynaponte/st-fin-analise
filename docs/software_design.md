# Software Design Document (SDD) — `finalise`

**Versão**: v1.0.1

## 1. Escopo

Biblioteca Python para análise estatística de séries temporais financeiras, com foco em identificação de ativos preditores e modelagem preditiva por árvore de decisão para trading automatizado. A biblioteca é uma caixa de ferramentas que compila soluções existentes em uma interface coesa, usando teoria da informação e métodos estatísticos. A arquitetura é **monolito modular**: três submódulos independentes (`target`, `predictors`, `model`), cada um expondo uma API de alto nível com a sequência recomendada encapsulada, sem impedir o uso direto dos métodos internos.

---

## 2. Estrutura do Pacote

```
finalise/
├── __init__.py
├── config.py
├── target/
│   ├── __init__.py          ← API pública do submódulo
│   ├── returns.py
│   ├── stationarity.py
│   ├── descriptive.py
│   ├── hurst.py
│   ├── entropy.py
│   ├── autocorrelation.py
│   └── selector.py
├── predictors/
│   ├── __init__.py          ← API pública do submódulo
│   ├── returns.py
│   ├── decomposition.py
│   ├── mi.py
│   ├── granger.py
│   ├── transfer_entropy.py
│   ├── cointegration.py
│   ├── smoothing.py
│   └── selector.py
├── model/
│   ├── __init__.py          ← API pública do submódulo
│   ├── features.py
│   ├── tree.py
│   └── validation.py
└── analyze.py               ← orquestrador opcional
```

---

## 3. Restrições Globais

- **R-G1** — Toda análise é estritamente causal: nenhum cálculo em t utiliza dados posteriores a t.
- **R-G2** — A série alvo é configurada em `config`. Não é intercambiável em runtime.
- **R-G3** — O submódulo `predictors` opera com duas representações dos dados em paralelo: **preços brutos (I(1))** para o teste de cointegração de Engle-Granger e **log-retornos (I(0))** para todos os demais testes (MI cruzada, Granger, TE, decomposição STL, suavização). `PredictorAnalysis` recebe `prices_dict` (preços brutos) e calcula log-retornos internamente. O submódulo `target` opera exclusivamente sobre log-retornos.
- **R-G4** — Suavização é aplicada exclusivamente às séries preditoras, nunca à série alvo.
- **R-G5** — Shift de séries suavizadas para compensar atraso de fase é proibido: introduz look-ahead bias.
- **R-G6** — Bins de histogramas estimados por Freedman-Diaconis em todo o pacote.
- **R-G7** — `finalise.predictors` decide quais features entram. `finalise.model` constrói e usa as features selecionadas. Nenhuma lógica de seleção reside em `model`.
- **R-G8** — Cada componente STL de um preditor (tendência, sazonalidade, resíduo) é tratado como candidato independente no pipeline de seleção. Para N preditores candidatos, até 5N candidatos entram no pipeline (série bruta + série suavizada + 3 componentes STL).
- **R-G9** — Log-retornos semanais (k=5) e mensais (k=21) são calculados sem sobreposição de janelas (non-overlapping). Consequência: ~240 observações semanais e ~57 mensais para 1200 dias úteis de dados. Quando a escala mensal for selecionada, um aviso explícito é emitido sobre potência estatística reduzida dos testes.
- **R-G10** — Nível de significância α padrão é 5%. Em caso de não-normalidade detectada pelo teste de Anderson-Darling (RF04), α é reduzido automaticamente para 2% em todos os testes subsequentes daquela série.
- **R-G11** — Output no terminal via `rich`. Gráficos via `plotly`. Obtenção de dados via `yfinance`.

---

## 4. Configuração (`config.py`)

- **R-C1** — Parâmetros obrigatórios: ticker do ativo alvo, lista de tickers preditores candidatos, horizontes k (padrão: {1, 5, 21} dias), janela de suavização (padrão: 30 dias úteis), método de MA (padrão: DEMA), período STL (padrão: 21 dias úteis).
- **R-C2** — Parâmetros opcionais: nível de significância α (padrão: 0.05), número de permutações para TE (padrão: 500), número de vizinhos k-NN para Kraskov (padrão: 5), lag máximo para varredura de MI (padrão: 21).
- **R-C3** — Configuração acessível como objeto imutável após inicialização. Submódulos recebem config por injeção, não por variável global.
- **R-C4** — `yfinance` é a fonte de dados. O módulo de configuração expõe `fetch(tickers, start, end)` que retorna preços de fechamento ajustados para todos os tickers.

---

## 5. Submódulo `finalise.target`

### 5.1 Métodos internos

**`returns.compute(prices, k, overlapping=False)`**
Log-retornos diários: `r_1(t) = log(P(t) / P(t-1))`.
Log-retornos semanais e mensais por soma de log-retornos diários consecutivos sem sobreposição:
`r_k = sum(r_1(t-i) for i in 0..k-1)` amostrado a cada k dias.
Descarta NaN sem imputação.

**`descriptive.stats(series)`**
Calcula média, desvio padrão, skewness e curtose. Executa teste de Anderson-Darling. Retorna dict com todas as estatísticas e flag `is_normal`. Se `is_normal=False`, ajusta α de 0.05 para 0.02 no objeto de configuração da série. Output formatado via `rich`.

**`stationarity.adf(series)`**
Teste ADF. Retorna estatística, p-valor e decisão binária ao nível α vigente (5% ou 2%). Séries não estacionárias emitem aviso via `rich` e são excluídas das análises subsequentes.

**`hurst.dfa(series)`**
Expoente de Hurst via DFA. Retorna H e |H − 0.5|.

**`entropy.shannon(series)`**
Entropia de Shannon. Bins por Freedman-Diaconis. Retorna valor em nats.

**`entropy.auto_mi(series, lag_max)`**
MI de `r_k(t)` com `r_k(t−lag)` para lag = 1..lag_max. Estimador Kraskov k-NN. Retorna perfil MI(lag) e lag ótimo interno.

**`autocorrelation.acf_pacf(series, lags)`**
ACF e PACF sobre `r_k` e sobre `r_k²`. Retorna coeficientes e intervalos de confiança.

**`selector.choose_horizon(results)`**
Ranqueia horizontes pelo vetor (|H − 0.5|, auto-MI máxima, entropia). Critério primário: |H − 0.5|. Emite aviso via `rich` se nenhum horizonte apresentar sinal (|H − 0.5| < 0.05 e auto-MI não significativa). Seleção é vinculante para o pipeline automatizado.

### 5.2 Relatórios

**`report.generate(results)`**
Resumo executivo via `rich` com: estatísticas descritivas, resultado de normalidade, Hurst por escala, perfil de auto-MI, ACF/PACF, horizonte selecionado e justificativa. Gráficos `plotly`: distribuição com histograma (Freedman-Diaconis) + KDE, perfil MI(lag), ACF/PACF, comparação de Hurst entre escalas.

### 5.3 API pública (`target/__init__.py`)

**`TargetAnalysis(prices, config)`**

```python
ta = TargetAnalysis(prices, config)
ta.run()       # executa toda a fase 1
ta.horizon     # horizonte selecionado (k em dias)
ta.alpha       # α vigente após teste de normalidade
ta.results     # dict com todas as métricas por escala
ta.report()    # resumo executivo + gráficos
```

---

## 6. Submódulo `finalise.predictors`

> **Nota sobre dados de entrada:** Este submódulo recebe `prices_dict` (preços brutos) e a `target_series` (log-retornos do alvo, produzida pela Fase 1). Internamente, mantém dois fluxos de dados:
> - **Preços brutos I(1)** — utilizados exclusivamente no teste de Engle-Granger (Seção 6.2, `cointegration`).
> - **Log-retornos I(0)** — calculados via `returns.compute` e utilizados em todo o pipeline de seleção (MI, Granger, TE) e na geração de candidatos (STL, suavização).

### 6.1 Geração de candidatos

**`returns.compute(prices, k, overlapping=False)`**
Reutiliza `target.returns.compute`. Converte os preços brutos de cada preditor em log-retornos no horizonte k* herdado da Fase 1. Estes log-retornos alimentam a expansão de candidatos e o pipeline de seleção. Os preços brutos originais são preservados separadamente para cointegração.

**`decomposition.stl(series, period)`**
Decomposição STL da série de log-retornos. Retorna dict `{trend, seasonal, residual}`. Período configurável via `config`. Cada componente é uma série indexada por data, estritamente causal.

**`smoothing.apply(series, window, method)`**
Aplica MA sem shift. Métodos suportados: SMA, EMA, DEMA, TMA, Gaussiana.

**`smoothing.group_delay(window, method)`**
Retorna atraso em amostras. Uso exclusivo para documentação e visualização.

Expansão de candidatos por preditor X (todos derivados de log-retornos):
```
{X_bruto, X_suavizado, X_tendência, X_sazonalidade, X_resíduo}
```

### 6.2 Pipeline de seleção

> Os testes abaixo (MI, Granger, TE) operam sobre **log-retornos I(0)** — séries estacionárias, conforme exigido pelos pressupostos estatísticos. O teste de cointegração, por sua vez, opera sobre **preços brutos I(1)** e é executado separadamente, antes da expansão de candidatos.

**`cointegration.engle_granger(price_x, price_y)`**
Teste de Engle-Granger sobre **preços brutos (I(1))**. Executado para cada par (preditor, alvo) antes da geração de candidatos. Retorna estatística, p-valor, spread e coeficientes da regressão. Levanta `ValueError` se receber séries com valores negativos (indicativo de log-retornos). Resultados armazenados em `pa.cointegration` — independentes do pipeline de log-retornos.

**`mi.cross_mi_lags(x, y, lag_max)`**
MI entre `x(t−τ)` e `y(t)` para τ = 1..lag_max. Entradas: log-retornos I(0). Estimador Kraskov k-NN. Retorna perfil MI(τ) e τ* = argmax. Significância por Bonferroni: threshold por lag = α / lag_max.

**`granger.test(x, y, lag)`**
Teste de Granger de x → y no lag τ*. Entradas: log-retornos I(0) (requisito de estacionaridade). Retorna p-valor e decisão ao nível α vigente.

**`transfer_entropy.compute(x, y, lag)`**
TE de x → y no lag τ*. Entradas: log-retornos I(0). Bins por Freedman-Diaconis. Significância por teste de permutação (N configurável, padrão 500). TE é significativa se superar o percentil (1 − α) da distribuição nula.

**`selector.select(candidates, target, config)`**
Pipeline de decisão aplicado a cada candidato individualmente (todos candidatos são derivados de log-retornos):

```
para cada candidato c em {X_bruto, X_suavizado, X_tendência, X_sazonalidade, X_resíduo}:
  1. MI cruzada com lags → τ*(c) e significância (Bonferroni)
  2. Se MI não significativa → descartar c
  3. Se MI significativa → Granger no lag τ*(c)
     - Rejeita H0 → incluir c (relação linear)
     - Não rejeita → TE no lag τ*(c)
       - TE significativa → incluir c (relação não-linear)
       - TE não significativa → descartar c
```

Retorna lista de candidatos selecionados com: série, τ*, tipo de relação (linear / não-linear), componente de origem. Seleção é vinculante para o pipeline automatizado.

### 6.3 Relatórios

**`report.generate(results)`**
Resumo executivo via `rich` com: candidatos avaliados vs. selecionados por preditor, tipo de relação identificada, resultados de cointegração, lag ótimo por candidato selecionado. Gráficos `plotly`: heatmap de MI cruzada por lag, p-valores de Granger, valores de TE com intervalo de permutação.

### 6.4 API pública (`predictors/__init__.py`)

**`PredictorAnalysis(prices_dict, target_series, config, alpha=None)`**

- `prices_dict`: `dict[str, pd.Series]` — preços brutos de todos os ativos (alvo + preditores). Usado para cointegração (I(1)) e para cálculo interno de log-retornos (I(0)).
- `target_series`: `pd.Series` — log-retornos do alvo no horizonte k* selecionado pela Fase 1.
- `alpha`: nível de significância herdado de `ta.alpha` (0.05 ou 0.02).

```python
pa = PredictorAnalysis(prices_dict, target_series, config, alpha=ta.alpha)
pa.run()            # cointegração sobre preços brutos + geração de candidatos (log-retornos) + pipeline de seleção
pa.selected         # lista de candidatos selecionados com metadados
pa.cointegration    # resultados de cointegração por par de preços brutos
pa.results          # dict completo com MI, Granger, TE por candidato (log-retornos)
pa.report()         # resumo executivo + gráficos
```

---

## 7. Submódulo `finalise.model`

### 7.1 Métodos internos

**`features.build(selected, t)`**
Constrói matriz de features `{candidato_i(t − τ*_i)}` para cada candidato em `pa.selected`. Garantia causal: nenhuma feature usa dados posteriores a t.

**`tree.fit(X, y, **kwargs)`**
Wrapper sobre `sklearn.DecisionTreeRegressor`. Expõe importância de features (impurity-based).

**`validation.random_walk_backtest(model, X, y, n_agents, seed)`**
Backtest por random walk: o modelo é colocado contra `n_agents` agentes com estratégias aleatórias de trading (sem acesso às séries explicativas). Retorno acumulado de cada agente é calculado no mesmo período out-of-sample. O modelo é considerado bom se seu retorno acumulado superar média + 2.5 desvios-padrão da distribuição dos agentes aleatórios.

**`validation.metrics(y_true, y_pred, agent_returns, model_return)`**
Retorna: retorno acumulado do modelo, média e desvio-padrão dos agentes, z-score do modelo, flag `is_outlier` (z > 2.5).

### 7.2 Relatórios

**`report.generate(results)`**
Resumo executivo via `rich` com: retorno acumulado do modelo, z-score vs. agentes aleatórios, importância de features. Gráficos `plotly`: distribuição de retornos dos agentes com modelo marcado, curva de retorno acumulado ao longo do tempo, importância de features.

### 7.3 API pública (`model/__init__.py`)

**`Model(selected, target_series, config)`**

```python
m = Model(pa.selected, target_series, config)
m.run()               # constrói features, treina, valida
m.is_outlier          # bool — modelo supera 2.5σ dos agentes aleatórios
m.z_score             # z-score do retorno acumulado
m.feature_importance  # importância por feature selecionada
m.report()            # resumo executivo + gráficos
```

---

## 8. Orquestrador (`analyze.py`)

- **R-PIP1** — Opcional. Conecta os três submódulos em sequência.
- **R-PIP2** — Toda decisão de seleção passa pelos métodos `selector` de cada submódulo. O orquestrador não contém lógica de decisão.
- **R-PIP3** — Logging estruturado em cada etapa via `rich`: entrada, saída, decisão e justificativa.
- **R-PIP4** — Gate explícito após Fase 1: se `TargetAnalysis` não identificar sinal, o pipeline encerra com relatório diagnóstico. Comportamento configurável: exceção (padrão) ou warning com continuação forçada.

```python
from finalise import analyze

result = analyze(prices_dict, config)
result.target      # TargetAnalysis
result.predictors  # PredictorAnalysis
result.model       # Model
```

---

## 9. Fora do Escopo (versão inicial)

- Modelos além de árvore de decisão
- Dados não financeiros
- Interface gráfica ou API HTTP
- Execução em tempo real / streaming
