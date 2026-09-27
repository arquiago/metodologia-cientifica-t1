"""Agrega as métricas do experimento completo por camada e posição de bit.

Executar na raiz do projeto com:

    py scripts/06_aggregate_results.py

Entradas:
    results/bit_injection_results.csv

Saídas:
    results/aggregated_metrics_by_layer_bit.csv
    results/aggregated_metrics_by_layer.csv
    results/aggregation_summary.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import BITS_PER_WEIGHT, SEED, WEIGHT_INDICES_PER_LAYER  # noqa: E402


SOURCE_PATH = PROJECT_ROOT / "results" / "bit_injection_results.csv"
BY_LAYER_BIT_PATH = PROJECT_ROOT / "results" / "aggregated_metrics_by_layer_bit.csv"
BY_LAYER_PATH = PROJECT_ROOT / "results" / "aggregated_metrics_by_layer.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "aggregation_summary.json"

COUNT_COLUMNS = (
    "n_new_errors",
    "n_silent_errors",
    "n_nonfinite_outputs",
)


def expected_field(bit: int) -> str:
    """Retorna o campo binary32 correspondente à posição do bit."""

    if 0 <= bit <= 22:
        return "mantissa"
    if 23 <= bit <= 30:
        return "exponent"
    if bit == 31:
        return "sign"
    raise ValueError(f"Posição de bit inválida: {bit}.")


def read_source() -> list[dict[str, str]]:
    """Lê o CSV bruto e confirma que as colunas necessárias existem."""

    with SOURCE_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "seed",
            "layer",
            "weight_index",
            "bit",
            "field",
            "n_eligible_images",
            "n_finite_outputs",
            "n_new_errors",
            "n_silent_errors",
            "n_nonfinite_outputs",
            "restored_exact",
        }
        missing = required_columns.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Colunas ausentes no CSV bruto: {sorted(missing)}")
        return list(reader)


def validate_source(rows: list[dict[str, str]]) -> tuple[list[str], int]:
    """Valida cobertura, chaves e invariantes antes de calcular as taxas."""

    expected_total = 3 * WEIGHT_INDICES_PER_LAYER * BITS_PER_WEIGHT
    if len(rows) != expected_total:
        raise ValueError(f"Esperadas {expected_total} injeções; encontradas {len(rows)}.")

    layers = list(dict.fromkeys(row["layer"] for row in rows))
    if len(layers) != 3:
        raise ValueError(f"Esperadas 3 camadas; encontradas {len(layers)}: {layers}.")

    seeds = {int(row["seed"]) for row in rows}
    if seeds != {SEED}:
        raise ValueError(f"Sementes inesperadas no CSV: {sorted(seeds)}.")

    eligible_values = {int(row["n_eligible_images"]) for row in rows}
    if len(eligible_values) != 1:
        raise ValueError(
            "Todas as injeções devem avaliar o mesmo número de imagens elegíveis."
        )
    n_eligible_images = eligible_values.pop()

    record_keys: set[tuple[str, int, int]] = set()
    for row in rows:
        layer = row["layer"]
        weight_index = int(row["weight_index"])
        bit = int(row["bit"])
        key = (layer, weight_index, bit)
        if key in record_keys:
            raise ValueError(f"Registro duplicado: {key}.")
        record_keys.add(key)

        if row["field"] != expected_field(bit):
            raise ValueError(
                f"Campo incorreto para {layer}[{weight_index}], bit {bit}: "
                f"{row['field']}."
            )

        n_finite = int(row["n_finite_outputs"])
        n_nonfinite = int(row["n_nonfinite_outputs"])
        n_new = int(row["n_new_errors"])
        n_silent = int(row["n_silent_errors"])
        if n_finite + n_nonfinite != n_eligible_images:
            raise ValueError(f"Contagens finitas/não finitas inconsistentes em {key}.")
        if not 0 <= n_silent <= n_new <= n_finite:
            raise ValueError(f"Contagens de erro inconsistentes em {key}.")
        if row["restored_exact"].strip().lower() != "true":
            raise ValueError(f"Peso não restaurado exatamente em {key}.")

    expected_keys = {
        (layer, weight_index, bit)
        for layer in layers
        for weight_index in {
            int(row["weight_index"])
            for row in rows
            if row["layer"] == layer
        }
        for bit in range(BITS_PER_WEIGHT)
    }
    if record_keys != expected_keys:
        raise ValueError("A grade camada × peso × bit está incompleta.")

    for layer in layers:
        layer_weights = {
            int(row["weight_index"]) for row in rows if row["layer"] == layer
        }
        if len(layer_weights) != WEIGHT_INDICES_PER_LAYER:
            raise ValueError(
                f"{layer}: esperados {WEIGHT_INDICES_PER_LAYER} pesos; "
                f"encontrados {len(layer_weights)}."
            )

    return layers, n_eligible_images


def aggregate_by_layer_bit(
    rows: list[dict[str, str]], layers: list[str], n_eligible_images: int
) -> list[dict[str, object]]:
    """Calcula numeradores, denominadores e taxas para as 96 células."""

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], int(row["bit"]))].append(row)

    aggregated: list[dict[str, object]] = []
    for layer in layers:
        for bit in range(BITS_PER_WEIGHT):
            group = grouped[(layer, bit)]
            n_injections = len(group)
            if n_injections != WEIGHT_INDICES_PER_LAYER:
                raise ValueError(
                    f"{layer}, bit {bit}: esperadas {WEIGHT_INDICES_PER_LAYER} "
                    f"injeções; encontradas {n_injections}."
                )
            denominator = n_injections * n_eligible_images
            totals = {
                column: sum(int(row[column]) for row in group)
                for column in COUNT_COLUMNS
            }
            aggregated.append(
                {
                    "seed": SEED,
                    "layer": layer,
                    "bit": bit,
                    "field": expected_field(bit),
                    "n_injections": n_injections,
                    "n_eligible_images_per_injection": n_eligible_images,
                    "denominator_image_evaluations": denominator,
                    "n_new_errors": totals["n_new_errors"],
                    "new_error_rate": totals["n_new_errors"] / denominator,
                    "n_silent_errors": totals["n_silent_errors"],
                    "silent_error_rate": totals["n_silent_errors"] / denominator,
                    "n_nonfinite_outputs": totals["n_nonfinite_outputs"],
                    "nonfinite_output_rate": (
                        totals["n_nonfinite_outputs"] / denominator
                    ),
                }
            )
    return aggregated


def aggregate_by_layer(
    layer_bit_rows: list[dict[str, object]], layers: list[str]
) -> list[dict[str, object]]:
    """Consolida as 32 posições de cada camada como controle de totais."""

    aggregated: list[dict[str, object]] = []
    for layer in layers:
        group = [row for row in layer_bit_rows if row["layer"] == layer]
        denominator = sum(
            int(row["denominator_image_evaluations"]) for row in group
        )
        n_new = sum(int(row["n_new_errors"]) for row in group)
        n_silent = sum(int(row["n_silent_errors"]) for row in group)
        n_nonfinite = sum(int(row["n_nonfinite_outputs"]) for row in group)
        aggregated.append(
            {
                "seed": SEED,
                "layer": layer,
                "n_bits": len(group),
                "n_injections": sum(int(row["n_injections"]) for row in group),
                "denominator_image_evaluations": denominator,
                "n_new_errors": n_new,
                "new_error_rate": n_new / denominator,
                "n_silent_errors": n_silent,
                "silent_error_rate": n_silent / denominator,
                "n_nonfinite_outputs": n_nonfinite,
                "nonfinite_output_rate": n_nonfinite / denominator,
            }
        )
    return aggregated


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Escreve uma tabela preservando números como campos numéricos."""

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_source()
    layers, n_eligible_images = validate_source(rows)
    by_layer_bit = aggregate_by_layer_bit(rows, layers, n_eligible_images)
    by_layer = aggregate_by_layer(by_layer_bit, layers)

    write_csv(BY_LAYER_BIT_PATH, by_layer_bit)
    write_csv(BY_LAYER_PATH, by_layer)

    denominator = sum(
        int(row["denominator_image_evaluations"]) for row in by_layer
    )
    totals = {
        "new_errors": sum(int(row["n_new_errors"]) for row in by_layer),
        "silent_errors": sum(int(row["n_silent_errors"]) for row in by_layer),
        "nonfinite_outputs": sum(
            int(row["n_nonfinite_outputs"]) for row in by_layer
        ),
    }
    summary = {
        "seed": SEED,
        "source_file": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "n_source_rows": len(rows),
        "n_layers": len(layers),
        "n_bits_per_layer": BITS_PER_WEIGHT,
        "n_layer_bit_cells": len(by_layer_bit),
        "n_weight_injections_per_cell": WEIGHT_INDICES_PER_LAYER,
        "n_eligible_images_per_injection": n_eligible_images,
        "denominator_per_layer_bit_cell": (
            WEIGHT_INDICES_PER_LAYER * n_eligible_images
        ),
        "n_image_evaluations": denominator,
        "totals": totals,
        "overall_rates": {
            "new_error_rate": totals["new_errors"] / denominator,
            "silent_error_rate": totals["silent_errors"] / denominator,
            "nonfinite_output_rate": totals["nonfinite_outputs"] / denominator,
        },
        "zero_event_cells": {
            "new_errors": sum(row["n_new_errors"] == 0 for row in by_layer_bit),
            "silent_errors": sum(
                row["n_silent_errors"] == 0 for row in by_layer_bit
            ),
            "nonfinite_outputs": sum(
                row["n_nonfinite_outputs"] == 0 for row in by_layer_bit
            ),
        },
        "definitions": {
            "denominator": (
                "Número de injeções na célula multiplicado pelo número de imagens "
                "elegíveis avaliadas em cada injeção."
            ),
            "new_error_rate": "n_new_errors / denominator_image_evaluations",
            "silent_error_rate": "n_silent_errors / denominator_image_evaluations",
            "nonfinite_output_rate": (
                "n_nonfinite_outputs / denominator_image_evaluations"
            ),
        },
        "output_files": [
            str(BY_LAYER_BIT_PATH.relative_to(PROJECT_ROOT)),
            str(BY_LAYER_PATH.relative_to(PROJECT_ROOT)),
        ],
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
