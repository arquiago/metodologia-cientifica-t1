"""Carrega, normaliza e divide o conjunto load_digits.

Executar na raiz do projeto com:

    py scripts/01_prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import SEED, TEST_SIZE  # noqa: E402


def main() -> None:
    dataset = load_digits()
    X = dataset.data.astype(np.float32) / 16.0
    y = dataset.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    artifacts_dir = PROJECT_ROOT / "artifacts"
    results_dir = PROJECT_ROOT / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        artifacts_dir / "digits_split.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    metadata = {
        "dataset": "sklearn.datasets.load_digits",
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "normalization": "raw_intensity / 16.0",
        "dtype_X": str(X.dtype),
        "dtype_y": str(y.dtype),
        "test_size": TEST_SIZE,
        "random_state": SEED,
        "train_class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(y_train, return_counts=True))
        },
        "test_class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(y_test, return_counts=True))
        },
    }
    (results_dir / "data_split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Dados salvos em: {artifacts_dir / 'digits_split.npz'}")


if __name__ == "__main__":
    main()

