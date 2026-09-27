# Injeção de falhas em pesos de uma rede neural

Código e resultados do experimento que acompanha o artigo **“Classificações Incorretas Silenciosas Causadas por Inversões de Bits em Pesos de uma Rede Neural”**, de Hiago Sousa Rocha, Universidade de Brasília.

O estudo investiga como a camada afetada e a posição do bit alteram a ocorrência de erros silenciosos em um classificador de dígitos. Cada injeção inverte um único bit de um peso `float32`, avalia as imagens de teste elegíveis e restaura o peso antes da próxima injeção.

## Consultar os resultados

**Não é necessário executar o experimento para consultar os resultados publicados.** Os arquivos principais são:

- [Resultados por injeção](results/bit_injection_results.csv): 1.920 registros, com os pesos e bits afetados, contagens de erros e confirmação da restauração.
- [Métricas por camada e bit](results/aggregated_metrics_by_layer_bit.csv): 96 combinações, com numeradores, denominadores e taxas.
- [Métricas por camada](results/aggregated_metrics_by_layer.csv): consolidação das 32 posições de bit de cada camada.
- [Resumo da execução principal](results/full_execution_summary.json): índices dos pesos sorteados, totais e duração da execução.
- [Resumo da agregação](results/aggregation_summary.json): totais e taxas globais.
- [Referência do modelo intacto](results/baseline_reference.json): configuração do treinamento e acurácia inicial.

## Protocolo do experimento

| Item | Configuração |
| --- | --- |
| Dados | `sklearn.datasets.load_digits`: 1.797 imagens de 8 × 8 pixels |
| Normalização | Intensidade dividida por 16 |
| Divisão | Estratificada; 1.257 imagens de treino e 540 de teste |
| Semente | `20260926` |
| Arquitetura | `64 → 64 → 32 → 10`, com ReLU nas duas camadas ocultas |
| Treinamento | Adam, taxa de aprendizado `0.001`, 300 épocas |
| Lote | Todas as 1.257 imagens de treino em cada época |
| Execução | CPU, pesos `float32`; avaliação sem gradientes |
| Imagens elegíveis | As 523 imagens de teste classificadas corretamente pelo modelo intacto |
| Amostragem | 20 pesos por camada, sem reposição e sem estratificação por sinal |
| Falhas | 32 posições de bit por peso; 1.920 injeções no total |

As camadas `network.0`, `network.2` e `network.4` correspondem, respectivamente, a `64 → 64`, `64 → 32` e `32 → 10`.

A posição 0 é o bit menos significativo: os bits 0–22 pertencem à fração (chamada de mantissa nos arquivos), os bits 23–30 ao expoente e o bit 31 ao sinal. A inversão usa XOR sobre uma visão `uint32` que compartilha a memória do peso. Uma segunda inversão em um bloco `finally` restaura o valor original, com verificação da representação binária.

### Definição das métricas

A confiança é a maior probabilidade após a aplicação de `softmax`. Todas as imagens avaliadas sob falha foram classificadas corretamente pelo modelo intacto.

- **Saída não finita:** ao menos um logit ou probabilidade da imagem é `NaN` ou infinito.
- **Novo erro:** a saída é finita e a classe prevista após a falha difere do rótulo verdadeiro.
- **Erro silencioso:** é um novo erro e a confiança após a falha é maior ou igual à confiança original menos `0.05`.

Assim, os erros silenciosos são um subconjunto dos novos erros. As saídas não finitas são contadas separadamente e não integram a coluna `n_new_errors`.

Para cada combinação de camada e bit:

```text
taxa = soma dos eventos / (número de injeções × número de imagens elegíveis)
     = soma dos eventos / (20 × 523)
     = soma dos eventos / 10.460
```

As taxas dos CSV são proporções entre 0 e 1; multiplique por 100 para obter percentuais.

## Resultados de referência

| Medida | Resultado |
| --- | ---: |
| Acurácia do modelo intacto | 96,85% (523/540) |
| Injeções | 1.920 |
| Avaliações de pares imagem–falha | 1.004.160 |
| Novos erros com saída finita | 9.072 |
| Erros silenciosos | 8.378 |
| Saídas não finitas | 2.055 |
| Taxa global de erros silenciosos | 0,834% |
| Taxa global de saídas não finitas | 0,205% |

O bit 30 concentrou 8.252 dos 8.378 erros silenciosos, ou 98,50%. Todas as 1.920 restaurações foram confirmadas.

### Relação com as tabelas do artigo

- **Tabela I:** selecionar `bit = 30` em `aggregated_metrics_by_layer_bit.csv`; o denominador de cada camada é 10.460.
- **Tabela II:** selecionar `bit = 30` em `bit_injection_results.csv`, separar por camada e pelo sinal de `value_before_float32`, somar `n_silent_errors` e dividir pela soma de `n_eligible_images` de cada grupo. Nas duas camadas ocultas, cada grupo contém 10 pesos e tem denominador 5.230.

O script de agregação gera as tabelas por camada e bit e por camada. Ainda não há um script específico que exporte a Tabela II ou as tabelas formatadas em LaTeX.

## Estrutura dos arquivos

```text
.
├── README.md
├── requirements.txt
├── config.py
├── resumo_execucao.md
├── scripts/
│   ├── 00_prepare.py
│   ├── 01_prepare_data.py
│   ├── 02_train_model.py
│   ├── 03_validate_bit_injection.py
│   ├── 04_run_pilot.py
│   ├── 05_run_full_experiment.py
│   ├── 06_aggregate_results.py
│   ├── model.py
│   └── bit_injection.py
├── artifacts/
│   ├── digits_split.npz
│   ├── model_baseline.pt
│   ├── eligible_test.npz
│   └── baseline_test_predictions.npz
└── results/
    ├── bit_injection_results.csv
    ├── aggregated_metrics_by_layer_bit.csv
    ├── aggregated_metrics_by_layer.csv
    └── ... metadados e registros de validação e piloto
```

Os parâmetros centrais estão em [config.py](config.py). Os scripts usam os nomes `artifacts/` e `results/`; mantenha essa estrutura.

### Artefatos da referência

| Arquivo | Conteúdo |
| --- | --- |
| `digits_split.npz` | `X_train`, `y_train`, `X_test` e `y_test`, com as entradas já normalizadas |
| `model_baseline.pt` | Estado do modelo treinado, arquitetura e semente |
| `eligible_test.npz` | Índices no conjunto de teste, rótulos, predições e confiança das imagens elegíveis |
| `baseline_test_predictions.npz` | Logits, probabilidades, predições, rótulos e confiança das 540 imagens de teste |

### Colunas do CSV por injeção

| Colunas | Significado |
| --- | --- |
| `seed`, `layer`, `weight_index` | Semente, camada e índice linear do peso na matriz da camada |
| `bit`, `field` | Posição do bit e campo: `mantissa`, `exponent` ou `sign` |
| `value_before_float32`, `value_after_float32` | Valor do peso antes e depois da inversão |
| `value_before_uint32`, `value_after_uint32` | Representações binárias reinterpretadas como inteiros sem sinal |
| `n_eligible_images` | Quantidade de imagens avaliadas na injeção |
| `n_finite_outputs`, `n_nonfinite_outputs` | Quantidades de saídas finitas e não finitas |
| `n_new_errors`, `n_silent_errors` | Quantidades de novos erros e erros silenciosos |
| `mean_confidence_delta_finite`, `min_confidence_delta_finite` | Média e mínimo de confiança após a falha menos confiança original, considerando todas as saídas finitas |
| `restored_exact` | Confirmação da restauração exata do peso |

## Ambiente e instalação

O ambiente registrado utilizou Windows 11, Python **3.13.15**, NumPy **2.5.3**, scikit-learn **1.9.1** e PyTorch **2.14.0**. As versões das bibliotecas estão fixadas em [requirements.txt](requirements.txt); os metadados estão em [results/run_metadata.json](results/run_metadata.json).

Os comandos abaixo são instruções para quem deseja executar o código. **Não são necessários para enviar os arquivos ao GitHub.** Execute-os na raiz de uma cópia local do repositório.

No PowerShell, crie um ambiente virtual e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Os exemplos a seguir usam esse ambiente sem precisar ativá-lo. Em Linux ou macOS, o caminho equivalente do interpretador é `.venv/bin/python`. A reprodução integral em outro ambiente ainda não foi validada.

## Opções de reprodução

**Escolha apenas a opção correspondente ao que deseja verificar.** Os scripts gravam nos caminhos fixos de `results/` e `artifacts/`. Para manter uma cópia intocada dos arquivos publicados, execute em outra cópia do repositório.

### 1. Recalcular somente as métricas do CSV publicado

```powershell
.\.venv\Scripts\python.exe scripts/06_aggregate_results.py
```

Lê `results/bit_injection_results.csv`, verifica cobertura, duplicatas, contagens e restaurações e regrava somente os dois CSV agregados e `aggregation_summary.json`. Não treina o modelo nem executa novas injeções. Esse script usa apenas a biblioteca padrão do Python; as dependências científicas não são necessárias para essa opção.

### 2. Reexecutar as falhas usando o modelo publicado

```powershell
.\.venv\Scripts\python.exe scripts/03_validate_bit_injection.py
.\.venv\Scripts\python.exe scripts/05_run_full_experiment.py
.\.venv\Scripts\python.exe scripts/06_aggregate_results.py
```

Usa o modelo, a divisão dos dados e as imagens elegíveis já salvos. Regrava a validação, os resultados das injeções, o resumo da execução principal e as métricas agregadas. Não retreina nem substitui os artefatos de referência.

### 3. Refazer o experimento desde a preparação dos dados

```powershell
.\.venv\Scripts\python.exe scripts/00_prepare.py
.\.venv\Scripts\python.exe scripts/01_prepare_data.py
.\.venv\Scripts\python.exe scripts/02_train_model.py
.\.venv\Scripts\python.exe scripts/03_validate_bit_injection.py
.\.venv\Scripts\python.exe scripts/04_run_pilot.py
.\.venv\Scripts\python.exe scripts/05_run_full_experiment.py
.\.venv\Scripts\python.exe scripts/06_aggregate_results.py
```

Essa sequência **sobrescreve os artefatos de referência e os resultados locais**, incluindo o modelo treinado. A preparação registra o ambiente atual; as etapas seguintes preparam os dados, treinam a referência, validam a restauração, executam um piloto de 27 injeções e realizam o experimento completo.

O conjunto `Digits` é carregado pela biblioteca scikit-learn, sem necessidade de fornecer um conjunto de dados externo. As sementes são fixadas no código. Diferenças de ambiente podem afetar os resultados numéricos, por isso os artefatos da execução de referência também são preservados no repositório.

## Escopo e limitações

Este é um estudo exploratório com uma arquitetura, um treinamento e 20 pesos amostrados por camada. As imagens avaliadas sob a mesma falha não são observações independentes. A comparação por sinal do peso é exploratória; a amostragem não foi balanceada intencionalmente por sinal.

“Silencioso” descreve apenas o critério operacional baseado em saídas finitas e queda de confiança de no máximo cinco pontos percentuais. Não significa ausência de detecção por qualquer outro mecanismo, nem confiança calibrada. A simulação caracteriza consequências funcionais de inversões de bits e não estima a frequência física de falhas de hardware.

