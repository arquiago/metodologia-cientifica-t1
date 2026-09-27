"""Prepara a execução e registra o ambiente de reprodutibilidade.

Executar na raiz do projeto com:

    py scripts/00_prepare.py
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BITS_PER_WEIGHT,
    CONFIDENCE_DROP_MAX,
    HIDDEN_SIZES,
    INPUT_SIZE,
    OUTPUT_SIZE,
    SEED,
    TEST_SIZE,
    WEIGHT_INDICES_PER_LAYER,
)


def package_version(distribution: str) -> str | None:
    """Retorna a versão instalada ou None quando o pacote não existe."""

    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    output_dirs = {
        "results": PROJECT_ROOT / "results",
        "figures": PROJECT_ROOT / "figures",
        "logs": PROJECT_ROOT / "logs",
        "artifacts": PROJECT_ROOT / "artifacts",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    # A semente será reutilizada nas etapas de dados, treinamento e amostragem.
    random.seed(SEED)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "test_size": TEST_SIZE,
        "architecture": [INPUT_SIZE, *HIDDEN_SIZES, OUTPUT_SIZE],
        "confidence_drop_max": CONFIDENCE_DROP_MAX,
        "weight_indices_per_layer": WEIGHT_INDICES_PER_LAYER,
        "bits_per_weight": BITS_PER_WEIGHT,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
            "torch": package_version("torch"),
        },
        "output_directories": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in output_dirs.items()
        },
    }

    metadata_path = PROJECT_ROOT / "results" / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Preparação concluída: {metadata_path}")


if __name__ == "__main__":
    main()

