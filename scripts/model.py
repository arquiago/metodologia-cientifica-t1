"""Arquitetura do classificador usada em todas as etapas."""

from __future__ import annotations

import torch
from torch import nn

from config import HIDDEN_SIZES, INPUT_SIZE, OUTPUT_SIZE


class DigitClassifier(nn.Module):
    """Rede 64 -> 64 -> 32 -> 10; a saída são logits."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[1], OUTPUT_SIZE),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

