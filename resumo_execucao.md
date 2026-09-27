# Resumo da execução

## Etapas concluídas

1. **Preparação:** semente `20260926`, arquitetura, limiar `0,05` e diretórios definidos. Ambiente registrado em `results/run_metadata.json`.
2. **Dados:** `load_digits` normalizado por `16`, com divisão estratificada:
   - treino: `1.257` imagens;
   - teste: `540` imagens.
3. **Modelo:** rede `64 → 64 → 32 → 10`, ReLU nas camadas ocultas, Adam por 300 épocas.
   - acurácia intacta: `96,85%`;
   - imagens elegíveis: `523`.
4. **Injeção de bits:** implementada via XOR sobre a representação `uint32` dos pesos `float32`, com restauração em `finally`.
5. **Validação:** bits de mantissa, expoente e sinal testados nas três camadas. As 9 validações foram aprovadas, com restauração byte a byte.
6. **Piloto controlado:** 3 pesos sorteados por camada, com inversão dos bits `0`, `23` e `31` (27 injeções ao todo), avaliados nas mesmas 523 imagens elegíveis.
   - erros novos com saída finita: `13`;
   - erros silenciosos: `2`;
   - saídas não finitas: `0`;
   - todas as restaurações foram confirmadas.
7. **Execução completa:** 20 pesos sorteados sem reposição em cada camada e 32 bits testados em cada peso: `3 × 20 × 32 = 1.920` injeções e `1.004.160` avaliações de imagem.
   - erros novos com saída finita: `9.072`;
   - erros silenciosos: `8.378`;
   - saídas não finitas: `2.055`;
   - todas as 1.920 restaurações foram confirmadas.
8. **Agregação das métricas:** os resultados brutos foram agregados por camada e posição de bit.
   - combinações camada–bit: `96`;
   - injeções por combinação: `20`;
   - denominador por combinação: `20 × 523 = 10.460` avaliações de imagem;
   - foram calculados numeradores, denominadores e taxas de erros novos, erros silenciosos e saídas não finitas;
   - os totais agregados foram reconciliados com as `1.920` injeções do arquivo bruto.

## Principais arquivos

- `config.py`
- `scripts/00_prepare.py`
- `scripts/01_prepare_data.py`
- `scripts/02_train_model.py`
- `scripts/03_validate_bit_injection.py`
- `scripts/04_run_pilot.py`
- `scripts/05_run_full_experiment.py`
- `scripts/06_aggregate_results.py`
- `artifacts/digits_split.npz`
- `artifacts/model_baseline.pt`
- `artifacts/eligible_test.npz`
- `results/baseline_reference.json`
- `results/bit_injection_validation.json`
- `results/pilot_bit_injections.csv`
- `results/pilot_summary.json`
- `results/bit_injection_results.csv`
- `results/full_execution_summary.json`
- `results/aggregated_metrics_by_layer_bit.csv`
- `results/aggregated_metrics_by_layer.csv`
- `results/aggregation_summary.json`
- `outputs/etapa_9/metricas_agregadas.xlsx`
