"""Primitivas para inverter e restaurar um bit de um peso float32."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch


FIELD_BY_BIT = {
    **{bit: "mantissa" for bit in range(0, 23)},
    **{bit: "exponent" for bit in range(23, 31)},
    31: "sign",
}


def uint32_view(weight: torch.Tensor) -> np.ndarray:
    """Retorna uma visão uint32 que compartilha a memória do peso float32."""

    if weight.device.type != "cpu":
        raise ValueError("A injeção desta etapa exige pesos na CPU.")
    if weight.dtype != torch.float32:
        raise TypeError("A injeção desta etapa exige pesos float32.")

    floats = weight.detach().numpy()
    if not floats.flags["C_CONTIGUOUS"]:
        raise ValueError("O peso precisa estar em memória contígua.")
    words = floats.view(np.uint32).reshape(-1)
    if not np.shares_memory(floats, words):
        raise RuntimeError("A visão uint32 não compartilha memória com o tensor.")
    return words


def flip_bit_in_place(weight: torch.Tensor, index: int, bit: int) -> tuple[int, int]:
    """Inverte um bit e retorna as palavras uint32 antes e depois da inversão."""

    if not 0 <= bit <= 31:
        raise ValueError("bit deve estar entre 0 e 31.")

    words = uint32_view(weight)
    if not 0 <= index < words.size:
        raise IndexError(f"index deve estar entre 0 e {words.size - 1}.")

    before = int(words[index])
    words[index] ^= np.uint32(1 << bit)
    after = int(words[index])
    return before, after


@contextmanager
def flipped_bit(weight: torch.Tensor, index: int, bit: int) -> Iterator[tuple[int, int]]:
    """Mantém uma inversão durante o bloco e restaura o peso em finally."""

    before, after = flip_bit_in_place(weight, index, bit)
    try:
        yield before, after
    finally:
        restored_before, restored_after = flip_bit_in_place(weight, index, bit)
        if restored_before != after or restored_after != before:
            raise RuntimeError("A restauração do peso não recuperou os bytes originais.")

