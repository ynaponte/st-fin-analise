# Plano de Testes — `finalise`

**Versão**: v1.0.1

---

## 1. Estratégia Geral

| Nível | Escopo | Dados | Framework |
|---|---|---|---|
| Unit | Função isolada, sem dependências externas | Sintéticos (numpy/hypothesis) | pytest + hypothesis |
| Integration | Interação entre métodos de um mesmo submódulo | Sintéticos controlados | pytest |
| End-to-End | Pipeline completo `analyze()` | Fixtures reais via yfinance | pytest |

**Princípios transversais:**
- Nenhum teste de unit depende de rede ou de outros módulos
- Dados sintéticos são gerados com seed fixo para reprodutibilidade
- Fixtures reais são baixadas uma vez e cacheadas em `tests/fixtures/`
- Violações de causalidade (R-G1) são verificadas em todos os testes de features
- Testes de propriedade (hypothesis) cobrem invariantes matemáticos, não comportamento específico

---

## 2. Estrutura de Arquivos

```
tests/
├── conftest.py                  # fixtures globais e helpers
├── fixtures/
│   └── prices.parquet           # dados reais pré-baixados (yfinance)
├── unit/
│   ├── target/
│   │   ├── test_returns.py
│   │   ├── test_descriptive.py
│   │   ├── test_stationarity.py
│   │   ├── test_hurst.py
│   │   ├── test_entropy.py
│   │   ├── test_autocorrelation.py
│   │   └── test_selector.py
│   ├── predictors/
│   │   ├── test_returns.py
│   │   ├── test_decomposition.py
│   │   ├── test_smoothing.py
│   │   ├── test_mi.py
│   │   ├── test_granger.py
│   │   ├── test_transfer_entropy.py
│   │   ├── test_cointegration.py
│   │   └── test_selector.py
│   └── model/
│       ├── test_features.py
│       ├── test_tree.py
│       └── test_validation.py
├── integration/
│   ├── test_target_pipeline.py
│   ├── test_predictors_pipeline.py
│   └── test_model_pipeline.py
└── e2e/
    └── test_analyze.py
```

---

## 3. Fixtures Globais (`conftest.py`)

```python
# Séries sintéticas reutilizadas nos testes de unit

@pytest.fixture
def white_noise():
    # ruído branco puro — H ≈ 0.5, MI ≈ 0, sem causalidade
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(1200))

@pytest.fixture
def persistent_series():
    # passeio aleatório com drift — H > 0.5
    rng = np.random.default_rng(42)
    return pd.Series(np.cumsum(rng.standard_normal(1200)) + 0.001)

@pytest.fixture
def antipersistent_series():
    # série revertendo à média — H < 0.5
    rng = np.random.default_rng(42)
    s = rng.standard_normal(1200)
    return pd.Series(s - 0.5 * np.roll(s, 1))

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
    path = Path("tests/fixtures/prices.parquet")
    if path.exists():
        return pd.read_parquet(path)
    import yfinance as yf
    tickers = ["PETR4.SA", "GC=F", "DX-Y.NYB", "XOM", "CL=F"]
    df = yf.download(tickers, start="2019-01-01", end="2024-01-01")["Close"]
    df.to_parquet(path)
    return df
```

---

## 4. Testes de Unit — `finalise.target`

### `test_returns.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-RET-01 | `r_1(t) = log(P(t)/P(t-1))` produz resultado correto para array conhecido | Determinístico | Sintético |
| UT-RET-02 | `r_k` non-overlapping tem comprimento `floor(N/k)` para k=5 e k=21 | Determinístico | Sintético |
| UT-RET-03 | Sem sobreposição: índices consecutivos de `r_5` distam exatamente 5 dias úteis | Determinístico | Sintético |
| UT-RET-04 | NaN no input resulta em descarte sem imputação | Determinístico | Sintético |
| UT-RET-05 | `r_5` por soma de diários é igual a `log(P(t)/P(t-5))` para mesmo período | Determinístico | Sintético |
| UT-RET-P01 | Para qualquer série de preços positivos, log-retornos têm média próxima de zero | Property-based | hypothesis |

### `test_descriptive.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-DESC-01 | Média, dp, skewness e curtose corretos para série normal conhecida | Determinístico | Sintético |
| UT-DESC-02 | Anderson-Darling retorna `is_normal=True` para série normal (N=1000, seed fixo) | Determinístico | Sintético |
| UT-DESC-03 | Anderson-Darling retorna `is_normal=False` para distribuição t-Student com df=3 | Determinístico | Sintético |
| UT-DESC-04 | `is_normal=False` dispara ajuste de α de 0.05 para 0.02 no objeto retornado | Determinístico | Sintético |
| UT-DESC-P01 | Curtose de qualquer distribuição normal sintética converge para 3 com N grande | Property-based | hypothesis |

### `test_stationarity.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-STAT-01 | ADF rejeita H0 (estacionária) para ruído branco com α=0.05 | Determinístico | Sintético |
| UT-STAT-02 | ADF não rejeita H0 para passeio aleatório puro | Determinístico | Sintético |
| UT-STAT-03 | Série não estacionária emite warning (capturado via `pytest.warns`) | Determinístico | Sintético |
| UT-STAT-04 | Decisão usa α=0.02 quando passado como argumento | Determinístico | Sintético |

### `test_hurst.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-HURST-01 | DFA retorna H ∈ (0, 1) para qualquer série válida | Determinístico | Sintético |
| UT-HURST-02 | Ruído branco retorna H no intervalo (0.45, 0.55) | Determinístico | `white_noise` |
| UT-HURST-03 | Passeio aleatório retorna H > 0.55 | Determinístico | `persistent_series` |
| UT-HURST-04 | Série antipersistente retorna H < 0.45 | Determinístico | `antipersistent_series` |
| UT-HURST-P01 | H está sempre em (0, 1) para qualquer série de floats não-constante | Property-based | hypothesis |

### `test_entropy.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-ENT-01 | Shannon de distribuição uniforme é maior que de distribuição concentrada | Determinístico | Sintético |
| UT-ENT-02 | Bins calculados por Freedman-Diaconis crescem com IQR maior | Determinístico | Sintético |
| UT-ENT-03 | auto-MI de ruído branco retorna valores próximos de zero para todos os lags | Determinístico | `white_noise` |
| UT-ENT-04 | auto-MI de série AR(1) tem pico em lag=1 | Determinístico | Sintético |
| UT-ENT-05 | Perfil MI(lag) tem comprimento igual a `lag_max` | Determinístico | Sintético |
| UT-ENT-P01 | Entropia de Shannon é sempre ≥ 0 para qualquer distribuição | Property-based | hypothesis |

### `test_autocorrelation.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-ACF-01 | ACF de ruído branco tem todos os coeficientes dentro do intervalo de confiança | Determinístico | `white_noise` |
| UT-ACF-02 | ACF de AR(1) com φ=0.5 tem coeficiente em lag=1 próximo de 0.5 | Determinístico | Sintético |
| UT-ACF-03 | ACF de `r²` de série GARCH tem coeficientes significativos | Determinístico | Sintético |
| UT-ACF-04 | PACF corta após lag=1 para AR(1) verdadeiro | Determinístico | Sintético |

### `test_selector.py` (target)

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-SEL-T-01 | Horizonte com maior |H−0.5| é sempre selecionado quando único critério diverge | Determinístico | Sintético |
| UT-SEL-T-02 | Gate emite warning quando |H−0.5| < 0.05 e auto-MI não significativa | Determinístico | Sintético |
| UT-SEL-T-03 | Escalas com ADF não estacionário são excluídas antes do ranking | Determinístico | Sintético |

---

## 5. Testes de Unit — `finalise.predictors`

### `test_decomposition.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-DEC-01 | Componentes STL somam à série original: `trend + seasonal + residual = series` | Determinístico | Sintético |
| UT-DEC-02 | Cada componente tem o mesmo índice temporal da série original | Determinístico | Sintético |
| UT-DEC-03 | Componente sazonal tem período correto (detectável via ACF) | Determinístico | Sintético |
| UT-DEC-04 | Nenhum componente usa dados futuros (verificar que série t depende só de t-1..t-n) | Determinístico | Sintético |

### `test_smoothing.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-SMO-01 | SMA(n) de série constante retorna a mesma constante | Determinístico | Sintético |
| UT-SMO-02 | `apply()` não desloca a série (índice de saída igual ao de entrada) | Determinístico | Sintético |
| UT-SMO-03 | `group_delay(n, 'sma')` retorna (n-1)/2 | Determinístico | Sintético |
| UT-SMO-04 | Todos os métodos suportados executam sem erro para janela válida | Determinístico | Sintético |
| UT-SMO-P01 | Para qualquer janela ≥ 1, saída tem comprimento ≤ comprimento da entrada | Property-based | hypothesis |

### `test_mi.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-MI-01 | MI cruzada de séries independentes retorna valores próximos de zero | Determinístico | Sintético |
| UT-MI-02 | MI cruzada com lag correto é maior que com lag errado para série com causalidade conhecida | Determinístico | Sintético |
| UT-MI-03 | τ* retornado é o argmax do perfil MI(τ) | Determinístico | Sintético |
| UT-MI-04 | Correção de Bonferroni: threshold = α / lag_max | Determinístico | Sintético |
| UT-MI-P01 | MI é sempre ≥ 0 para qualquer par de séries | Property-based | hypothesis |

### `test_granger.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-GRA-01 | Granger rejeita H0 para série com causalidade linear plantada (VAR sintético) | Determinístico | Sintético |
| UT-GRA-02 | Granger não rejeita H0 para séries independentes | Determinístico | Sintético |
| UT-GRA-03 | Teste usa exatamente o lag τ* passado como argumento | Determinístico | Sintético |
| UT-GRA-04 | Decisão usa α vigente (0.02 vs 0.05) corretamente | Determinístico | Sintético |
| UT-GRA-05 | Entradas devem ser séries I(0) estacionárias — resultado com série I(1) não é confiável (documental, não validado por asserção) | Determinístico | Sintético |

### `test_transfer_entropy.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-TE-01 | TE de séries independentes não é significativa pelo teste de permutação | Determinístico | Sintético |
| UT-TE-02 | TE de série com dependência não-linear plantada é significativa | Determinístico | Sintético |
| UT-TE-03 | Distribuição nula tem comprimento igual a N permutações configurado | Determinístico | Sintético |
| UT-TE-04 | Bins calculados por Freedman-Diaconis, não valor fixo | Determinístico | Sintético |
| UT-TE-05 | TE não é simétrica: TE(X→Y) ≠ TE(Y→X) para série com direção conhecida | Determinístico | Sintético |

### `test_cointegration.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-COI-01 | Engle-Granger detecta cointegração para par cointegrado sintético | Determinístico | Sintético |
| UT-COI-02 | Engle-Granger não detecta cointegração para passeios aleatórios independentes | Determinístico | Sintético |
| UT-COI-03 | Entrada deve ser preços brutos — erro explícito (`ValueError`) se passar log-retornos (valores negativos) | Determinístico | Sintético |
| UT-COI-04 | Spread retornado é I(0) quando cointegração é detectada | Determinístico | Sintético |
| UT-COI-05 | Retorno inclui campos `beta` e `alpha_const` da regressão OLS | Determinístico | Sintético |

### `test_selector.py` (predictors)

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-SEL-P-01 | Candidato com MI não significativa é descartado sem rodar Granger | Determinístico | Sintético |
| UT-SEL-P-02 | Candidato com MI significativa e Granger rejeitado é incluído como linear | Determinístico | Sintético |
| UT-SEL-P-03 | Candidato com MI significativa, Granger não rejeitado e TE significativa é incluído como não-linear | Determinístico | Sintético |
| UT-SEL-P-04 | Candidato com MI significativa, Granger não rejeitado e TE não significativa é descartado | Determinístico | Sintético |
| UT-SEL-P-05 | `SelectedCandidate` retornado contém todos os campos obrigatórios | Determinístico | Sintético |

---

## 6. Testes de Unit — `finalise.model`

### `test_features.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-FEAT-01 | Feature `candidato_i(t − τ*_i)` está corretamente defasada para τ* conhecido | Determinístico | Sintético |
| UT-FEAT-02 | Nenhuma feature em t usa dados de t+1 ou posterior (verificação de índice) | Determinístico | Sintético |
| UT-FEAT-03 | Matriz X tem shape (T − max(τ*), N_candidatos) | Determinístico | Sintético |
| UT-FEAT-04 | Features de candidatos com τ* diferentes são corretamente alinhadas por índice | Determinístico | Sintético |
| UT-FEAT-P01 | Para qualquer lista de candidatos válidos, X não contém NaN após construção | Property-based | hypothesis |

### `test_tree.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-TREE-01 | `tree.fit` treina sem erro para X e y válidos | Determinístico | Sintético |
| UT-TREE-02 | `feature_importance` soma 1.0 | Determinístico | Sintético |
| UT-TREE-03 | `feature_importance` tem comprimento igual ao número de features | Determinístico | Sintético |
| UT-TREE-04 | Kwargs adicionais são passados para `DecisionTreeRegressor` | Determinístico | Sintético |

### `test_validation.py`

| ID | Descrição | Tipo | Dado |
|---|---|---|---|
| UT-VAL-01 | Backtest produz `n_agents` retornos acumulados | Determinístico | Sintético |
| UT-VAL-02 | Agentes aleatórios não têm acesso às features do modelo (verificar isolamento) | Determinístico | Sintético |
| UT-VAL-03 | `z_score` calculado corretamente: `(ret_modelo − μ) / σ` | Determinístico | Sintético |
| UT-VAL-04 | `is_outlier=True` quando z_score > 2.5 | Determinístico | Sintético |
| UT-VAL-05 | `is_outlier=False` quando z_score ≤ 2.5 | Determinístico | Sintético |
| UT-VAL-06 | Modelo com retorno plantado como outlier é corretamente identificado | Determinístico | Sintético |

---

## 7. Testes de Integration

### `test_target_pipeline.py`

| ID | Descrição |
|---|---|
| IT-T-01 | `TargetAnalysis.run()` completo com série normal — horizonte selecionado e α=0.05 |
| IT-T-02 | `TargetAnalysis.run()` com série não-normal — α ajustado para 0.02 propagado internamente |
| IT-T-03 | Gate ativa warning quando ruído branco puro é passado como alvo |
| IT-T-04 | `ta.results` contém métricas para todas as escalas estacionárias |
| IT-T-05 | `ta.report()` executa sem erro e retorna figuras plotly |

### `test_predictors_pipeline.py`

| ID | Descrição |
|---|---|
| IT-P-01 | Expansão de candidatos gera exatamente 5 candidatos por preditor (bruto, suavizado, trend, seasonal, residual) |
| IT-P-02 | Pipeline descarta todos os candidatos de série independente do alvo |
| IT-P-03 | Pipeline seleciona candidato correto para série com causalidade linear plantada |
| IT-P-04 | Pipeline seleciona candidato correto para série com dependência não-linear plantada |
| IT-P-05 | α herdado de `ta.alpha` é usado em todos os testes do pipeline |
| IT-P-06 | `pa.cointegration` opera sobre preços brutos I(1), não log-retornos — pares acessíveis via `pa.cointegration[ticker]` |
| IT-P-07 | Candidatos no pipeline de seleção (MI, Granger, TE) são log-retornos I(0), verificados pela ausência de tendência unitária (ADF) |
| IT-P-08 | `pa.report()` executa sem erro e retorna figuras plotly |

### `test_model_pipeline.py`

| ID | Descrição |
|---|---|
| IT-M-01 | `Model.run()` completo com candidatos sintéticos selecionados |
| IT-M-02 | Modelo com retorno acumulado plantado acima de 2.5σ retorna `is_outlier=True` |
| IT-M-03 | Modelo com retorno aleatório retorna `is_outlier=False` na maioria das seeds |
| IT-M-04 | `m.report()` executa sem erro e retorna figuras plotly |

---

## 8. Testes End-to-End (`test_analyze.py`)

Usam `prices_real` (fixture cacheada, yfinance). Marcados com `@pytest.mark.e2e` para execução separada do CI padrão.

| ID | Descrição |
|---|---|
| E2E-01 | `analyze()` completo com PETR4.SA como alvo e {GC=F, DX-Y.NYB, XOM, CL=F} como preditores executa sem erro |
| E2E-02 | `result.target.horizon` é um dos valores em {1, 5, 21} |
| E2E-03 | `result.target.alpha` é 0.05 ou 0.02 |
| E2E-04 | `result.predictors.selected` é lista (pode ser vazia) |
| E2E-05 | Se `selected` não vazio, cada item tem todos os campos de `SelectedCandidate` |
| E2E-06 | `result.model.is_outlier` é bool |
| E2E-07 | `analyze()` com `config.force_continue=False` levanta exceção quando alvo é ruído branco puro |
| E2E-08 | `analyze()` com `config.force_continue=True` emite warning e continua quando alvo é ruído branco puro |
| E2E-09 | Todos os `report()` executam sem erro no pipeline completo |

---

## 9. Testes de Propriedade (hypothesis) — Invariantes Matemáticos

| ID | Invariante | Módulo |
|---|---|---|
| PB-01 | Log-retorno de preços positivos é sempre finito e real | `target.returns` |
| PB-02 | Entropia de Shannon ≥ 0 para qualquer distribuição | `target.entropy` |
| PB-03 | H ∈ (0, 1) para qualquer série de floats não-constante | `target.hurst` |
| PB-04 | MI ≥ 0 para qualquer par de séries | `predictors.mi` |
| PB-05 | `trend + seasonal + residual = series` para qualquer série válida | `predictors.decomposition` |
| PB-06 | Série suavizada tem comprimento ≤ comprimento da entrada | `predictors.smoothing` |
| PB-07 | Matriz de features não contém índices futuros em relação ao target | `model.features` |
| PB-08 | `feature_importance` soma 1.0 para qualquer conjunto de features | `model.tree` |
| PB-09 | `z_score = (ret_modelo − mean(agentes)) / std(agentes)` para quaisquer valores | `model.validation` |

---

## 10. Cobertura Mínima Esperada

| Submódulo | Cobertura mínima |
|---|---|
| `finalise.target` | 90% |
| `finalise.predictors` | 90% |
| `finalise.model` | 85% |
| `analyze.py` | 80% |
| **Total** | **88%** |

Cobertura medida com `pytest-cov`. Excluídos de cobertura: `report.py` de cada módulo (dependente de renderização `rich`/`plotly`).

---

## 11. Configuração (`pytest.ini`)

```ini
[pytest]
testpaths = tests
markers =
    e2e: testes end-to-end com dados reais (requer rede na primeira execução)
    slow: testes lentos (permutações de TE com N alto)
addopts = --strict-markers -q
```

Execução padrão (CI): `pytest -m "not e2e and not slow"`
Execução completa: `pytest`
