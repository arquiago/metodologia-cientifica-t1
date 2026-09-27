"""Valida a inversão e a restauração de bits nos pesos do modelo."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import SEED  # noqa: E402
from model import DigitClassifier  # noqa: E402
from bit_injection import FIELD_BY_BIT, flipped_bit, flip_bit_in_place, uint32_view  # noqa: E402


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)

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
    X_test = torch.from_numpy(split["X_test"])
    X_eligible = X_test[eligible["test_indices"]]

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    validation = []

    with torch.inference_mode():
        for layer_name, layer in linear_layers:
            weight = layer.weight
            original_bytes = weight.detach().numpy().tobytes()
            original_words = uint32_view(weight).copy()

            for bit in (0, 23, 31):
                # Testa explicitamente que duas inversões recuperam os bytes originais.
                double_test_before = weight.detach().numpy().tobytes()
                first_before, first_after = flip_bit_in_place(weight, 0, bit)
                second_before, second_after = flip_bit_in_place(weight, 0, bit)
                double_test_after = weight.detach().numpy().tobytes()

                with flipped_bit(weight, 0, bit) as (value_before, value_after):
                    logits = model(X_eligible)
                    finite_outputs = bool(torch.isfinite(logits).all().item())
                    n_predictions = int(logits.shape[0])

                restored_bytes = weight.detach().numpy().tobytes()
                restored_words = uint32_view(weight).copy()
                validation.append(
                    {
                        "layer": layer_name,
                        "weight_index": 0,
                        "bit": bit,
                        "field": FIELD_BY_BIT[bit],
                        "value_before_uint32": value_before,
                        "value_after_uint32": value_after,
                        "bit_changed_only": value_after == (value_before ^ (1 << bit)),
                        "double_inversion_exact": double_test_before == double_test_after,
                        "double_inversion_values": {
                            "first_before": first_before,
                            "first_after": first_after,
                            "second_before": second_before,
                            "second_after": second_after,
                        },
                        "finite_outputs": finite_outputs,
                        "n_predictions": n_predictions,
                        "restored_exact": restored_bytes == original_bytes,
                        "restored_words_exact": bool(np.array_equal(restored_words, original_words)),
                    }
                )

    passed = all(
        item["bit_changed_only"]
        and item["double_inversion_exact"]
        and item["finite_outputs"]
        and item["restored_exact"]
        and item["restored_words_exact"]
        for item in validation
    )
    result = {
        "seed": SEED,
        "n_cases": len(validation),
        "n_layers": len(linear_layers),
        "tested_bits": [0, 23, 31],
        "n_eligible_images": int(len(X_eligible)),
        "passed": passed,
        "cases": validation,
    }
    output_path = PROJECT_ROOT / "results" / "bit_injection_validation.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("n_cases", "n_layers", "n_eligible_images", "passed")}, indent=2))
    print(f"Validação salva em: {output_path}")

    if not passed:
        raise SystemExit("A validação da inversão/restauração falhou.")


if __name__ == "__main__":
    main()

