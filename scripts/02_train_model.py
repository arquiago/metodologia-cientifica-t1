"""Treina o modelo de referência e salva as imagens de teste elegíveis.

Executar na raiz do projeto com:

    py scripts/02_train_model.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    HIDDEN_SIZES,
    INPUT_SIZE,
    LEARNING_RATE,
    OUTPUT_SIZE,
    SEED,
    TRAINING_EPOCHS,
)
from model import DigitClassifier  # noqa: E402


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def main() -> None:
    set_seeds()
    device = torch.device("cpu")

    split = np.load(PROJECT_ROOT / "artifacts" / "digits_split.npz")
    X_train = torch.from_numpy(split["X_train"]).to(device)
    y_train = torch.from_numpy(split["y_train"]).to(device)
    X_test = torch.from_numpy(split["X_test"]).to(device)
    y_test = torch.from_numpy(split["y_test"]).to(device)

    model = DigitClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_probabilities = torch.softmax(test_logits, dim=1)
        test_confidence, test_predictions = test_probabilities.max(dim=1)

    correct = test_predictions.eq(y_test)
    eligible_indices = torch.nonzero(correct, as_tuple=False).flatten()
    accuracy = float(correct.float().mean().item())

    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_dir = PROJECT_ROOT / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": [INPUT_SIZE, *HIDDEN_SIZES, OUTPUT_SIZE],
            "seed": SEED,
        },
        artifacts_dir / "model_baseline.pt",
    )
    np.savez_compressed(
        artifacts_dir / "baseline_test_predictions.npz",
        logits=test_logits.cpu().numpy().astype(np.float32),
        probabilities=test_probabilities.cpu().numpy().astype(np.float32),
        predictions=test_predictions.cpu().numpy().astype(np.int64),
        labels=y_test.cpu().numpy().astype(np.int64),
        confidence=test_confidence.cpu().numpy().astype(np.float32),
    )
    np.savez_compressed(
        artifacts_dir / "eligible_test.npz",
        test_indices=eligible_indices.cpu().numpy().astype(np.int64),
        labels=y_test[eligible_indices].cpu().numpy().astype(np.int64),
        predictions=test_predictions[eligible_indices].cpu().numpy().astype(np.int64),
        confidence=test_confidence[eligible_indices].cpu().numpy().astype(np.float32),
    )

    metadata = {
        "architecture": [INPUT_SIZE, *HIDDEN_SIZES, OUTPUT_SIZE],
        "activation": "ReLU between hidden linear layers",
        "output": "logits; softmax only for evaluation confidence",
        "device": str(device),
        "dtype": "float32",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "epochs": TRAINING_EPOCHS,
        "seed": SEED,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_correct_test": int(correct.sum().item()),
        "n_eligible": int(eligible_indices.numel()),
        "test_accuracy": accuracy,
    }
    (results_dir / "baseline_reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Modelo salvo em: {artifacts_dir / 'model_baseline.pt'}")
    print(f"Elegíveis salvas em: {artifacts_dir / 'eligible_test.npz'}")


if __name__ == "__main__":
    main()
