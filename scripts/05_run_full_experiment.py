"""Executa as 1.920 injeções previstas no experimento principal.

Para cada camada linear, sorteia 20 pesos sem reposição com a semente fixa.
Cada peso é submetido às 32 posições de bit, sendo restaurado antes do caso
seguinte. Uma linha de resultado é salva para cada injeção.

Executar na raiz do projeto com:

    python scripts/05_run_full_experiment.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bit_injection import FIELD_BY_BIT, flipped_bit, uint32_view  # noqa: E402
from config import (  # noqa: E402
    BITS_PER_WEIGHT,
    CONFIDENCE_DROP_MAX,
    SEED,
    WEIGHT_INDICES_PER_LAYER,
)
from model import DigitClassifier  # noqa: E402


def set_seeds() -> None:
    """Fixa as fontes de aleatoriedade usadas pela amostragem e inferência."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def float_for_csv(value: float) -> str:
    """Serializa float32, inclusive NaN e infinito, sem perder legibilidade."""

    return format(float(value), ".9g")


def load_reference() -> tuple[DigitClassifier, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carrega o modelo intacto e a referência das imagens elegíveis."""

    checkpoint = torch.load(
        PROJECT_ROOT / "artifacts" / "model_baseline.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = DigitClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split = np.load(PROJECT_ROOT / "artifacts" / "digits_split.npz")
    eligible = np.load(PROJECT_ROOT / "artifacts" / "eligible_test.npz")
    x_test = torch.from_numpy(split["X_test"])
    return (
        model,
        x_test[eligible["test_indices"]],
        torch.from_numpy(eligible["labels"]),
        torch.from_numpy(eligible["confidence"]),
    )


def main() -> None:
    set_seeds()
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    model, x_eligible, labels, baseline_confidence = load_reference()
    generator = np.random.default_rng(SEED)

    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    selected_indices: dict[str, list[int]] = {}
    records: list[dict[str, object]] = []

    with torch.inference_mode():
        for layer_name, layer in layers:
            words = uint32_view(layer.weight)
            indices = np.sort(
                generator.choice(
                    words.size,
                    size=WEIGHT_INDICES_PER_LAYER,
                    replace=False,
                )
            )
            selected_indices[layer_name] = [int(index) for index in indices]

            for weight_index_np in indices:
                weight_index = int(weight_index_np)
                for bit in range(BITS_PER_WEIGHT):
                    original_word = int(uint32_view(layer.weight)[weight_index])
                    value_before_float = float(
                        layer.weight.detach().numpy().reshape(-1)[weight_index]
                    )

                    with flipped_bit(layer.weight, weight_index, bit) as (
                        value_before_uint32,
                        value_after_uint32,
                    ):
                        value_after_float = float(
                            layer.weight.detach().numpy().reshape(-1)[weight_index]
                        )
                        logits = model(x_eligible)
                        probabilities = torch.softmax(logits, dim=1)
                        confidence_after, predictions_after = probabilities.max(dim=1)

                        finite_per_image = torch.isfinite(logits).all(dim=1) & torch.isfinite(
                            probabilities
                        ).all(dim=1)
                        new_errors = finite_per_image & predictions_after.ne(labels)
                        silent_errors = new_errors & confidence_after.ge(
                            baseline_confidence - CONFIDENCE_DROP_MAX
                        )
                        finite_confidence_deltas = (
                            confidence_after[finite_per_image]
                            - baseline_confidence[finite_per_image]
                        )

                    restored_word = int(uint32_view(layer.weight)[weight_index])
                    restored_exact = restored_word == original_word
                    if not restored_exact:
                        raise RuntimeError(
                            f"Peso não restaurado: {layer_name}[{weight_index}], bit {bit}."
                        )

                    records.append(
                        {
                            "seed": SEED,
                            "layer": layer_name,
                            "weight_index": weight_index,
                            "bit": bit,
                            "field": FIELD_BY_BIT[bit],
                            "value_before_float32": float_for_csv(value_before_float),
                            "value_after_float32": float_for_csv(value_after_float),
                            "value_before_uint32": value_before_uint32,
                            "value_after_uint32": value_after_uint32,
                            "n_eligible_images": int(len(x_eligible)),
                            "n_finite_outputs": int(finite_per_image.sum().item()),
                            "n_new_errors": int(new_errors.sum().item()),
                            "n_silent_errors": int(silent_errors.sum().item()),
                            "n_nonfinite_outputs": int((~finite_per_image).sum().item()),
                            "mean_confidence_delta_finite": float_for_csv(
                                finite_confidence_deltas.mean().item()
                                if finite_confidence_deltas.numel()
                                else float("nan")
                            ),
                            "min_confidence_delta_finite": float_for_csv(
                                finite_confidence_deltas.min().item()
                                if finite_confidence_deltas.numel()
                                else float("nan")
                            ),
                            "restored_exact": restored_exact,
                        }
                    )

    expected_injections = len(layers) * WEIGHT_INDICES_PER_LAYER * BITS_PER_WEIGHT
    if len(records) != expected_injections:
        raise RuntimeError(
            f"Foram registradas {len(records)} injeções; esperadas {expected_injections}."
        )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    records_path = results_dir / "bit_injection_results.csv"
    with records_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    elapsed_seconds = time.perf_counter() - started_monotonic
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "seed": SEED,
        "n_layers": len(layers),
        "weight_indices_per_layer": WEIGHT_INDICES_PER_LAYER,
        "bits_per_weight": BITS_PER_WEIGHT,
        "n_injections": len(records),
        "n_eligible_images": int(len(x_eligible)),
        "n_image_evaluations": len(records) * int(len(x_eligible)),
        "selected_weight_indices": selected_indices,
        "totals": {
            "new_errors": sum(int(record["n_new_errors"]) for record in records),
            "silent_errors": sum(int(record["n_silent_errors"]) for record in records),
            "nonfinite_outputs": sum(
                int(record["n_nonfinite_outputs"]) for record in records
            ),
        },
        "all_weights_restored": all(bool(record["restored_exact"]) for record in records),
        "definitions": {
            "finite_output": "Todos os logits e probabilidades da imagem são finitos.",
            "new_error": "Saída finita e classe prevista após a injeção diferente do rótulo real.",
            "silent_error": (
                "Erro novo com saída finita e confiança após a injeção maior ou igual "
                "à confiança intacta menos 0,05."
            ),
        },
        "records_file": str(records_path.relative_to(PROJECT_ROOT)),
    }
    summary_path = results_dir / "full_execution_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Resultados por injeção salvos em: {records_path}")


if __name__ == "__main__":
    main()
