# finalise -- Analise estatística (descritiva e inferencial) sobre séries temporais financeiras

**Versão**: v1.0.0

## Escopo

Compreende a construção de um módulo em Python para análise estatística de séries temporais financeiras. As funções e métodos de classes devem ser, quando possível, informativos, seja criando gráficos e/ou gerando relatórios na CLI. Deve compilar ao máximo soluções já prontas, com a proposta de ser uma inteface mais amigável.
A inteção é fazer uma caixa de ferramentas para análise de ativos finânceiros, utilizando teoria da informação, métricas e métodos estatísticos como testes de causalidade de granger, co-integração, curtose, assimetria, entre outros, para identificar séries (outros ativos) que consigam descrever (séries explicativas) e serem utilizados para a previsão, seja direta ou indiretamente, de um ativo alvo (série).
Além de ajudar na seleção de ativos preditores, deve também um módulo para modelagem, utilizando machine learning. Mais especificamente, uma árvore de decisão, treinada nas séries escolhidas como preditoras, com o intuito de fazer o trading automático do ativo. O backtest de validação deve ser feito por *random walk*, com o previsor sendo colocado contra múltiplos agentes de trading, com estratégias aleatórias (sem acesso as séries explicativas), sendo considerado bom se aparecer como outlier nos retornos (retorno acumulado maior que 2.5 desvios-padrões).
   
## Recursos Funcionais

- RF01: Deve ser possível escolher somente um ativo alvo e múltiplos ativos previsores
- RF02: O módulo investigativo deve somente informar ao usuário do mesmo suas análises e relatórios de recomendação, ficando a cargo do usuário a decisão de prosseguir ou tentar novamente ou qualquer outra ação.
- RF03: Deve utilizar primáriamente log-retornos das séries para análise. Calcula-se o log-retorno diário, semanal (5 dias) e mensal (21 dias).
- RF04: Calcular Média, desvio padrão, assimetria (skewness) e curtose, juntamente com o teste de Anderson-Darling, compilando tudo para atestar a normalidade da série. Em caso de não normalidade, o $\alpha$ do teste p-valor, deve ser diminuido de cinco para dois por cento.
- RF05: Calcular o Expoente de Hurst utilizando Detrended Fluctuation Analysis (DFA)
- RF06: Número de *bins* para os histogramas deve ser estimado por Freedman-Diaconis
- RF07: Deve executar análise de transferência de entropia utilizando o teste de permutação, de modo a entender se o valor de transferência é significativo ou não. O número de permutações fica a critério do usuário, mas deve ter valor padrão em 500.
- RF08: Deve fazer a decomposição sazonal das séries preditoras. Os componentes resultantes devem sofrer análise de preditores.
- RF09: Cada módulo deve ter um método (ou métodos, em caso de necessidade de auxiliáres) específico em suas APIs para geração de relatórios finais, informando o usuário em um resumo executivo de toda a análise feita, com gráficos informativos.

## Recursos Não funcionais

- RNF01: O software deve ser um módulo python
- RNF02: Deve ser dividido em três subsistemas/submódulos/pastas, com o primeiro englobando toda a análise de previsibilidade do ativo alvo; o segundo, todo a parte de análise cruzada, i.e. descobrir quais séries são boas previsoras do alvo; e o terceiro sendo o módulo de *machine learning*, com uma árvore de decisão pronta para ser treinado com os dados descobertos.
- RNF03: Cada módulo deve expor uma API que compila o pipeline de execução projetado para cada módulo da biblioteca.
- RNF04: Ficam fora de escopo interfaces gráficas, APIs HTTPs, execução em tempo real / streaming, dados não financeiros.
- RNF05: Deve ser utilizado `yfinance` para obtenção das séries temporais.
- RNF06: Deve ser utilizado o módulo `rich` para escrita no terminal.
- RNF07: Deve ser utilizado o módulo `plotly` para plotagem de gráficos.
- RNF08: As 