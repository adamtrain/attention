"""Quantization: storing each weight in fewer bits, the way big models are shrunk to fit.

Your model keeps every weight as a 64-bit float, NumPy's default. Large models are trained with
16-bit numbers (bfloat16), and are often run with 8 or 4 bits per weight. The trick: cut each
grid of weights into blocks of 32 numbers, find the biggest in each block, and store every number
as a small whole number of steps of that block's size. To use a weight, multiply it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Array, Transformer

BLOCK = 32  # numbers sharing one scale, as in GGUF's Q8_0 and Q4_0 files
LEVELS = (16, 8, 4, 3, 2)  # bits per weight to try; 16 means bfloat16


def bfloat16(x: Array) -> Array:
    """Round to bfloat16: a 32-bit float's sign, its exponent, and 7 bits of the rest."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)  # to nearest
    return bits.view(np.float32).astype(np.float64)


def quantize(rows: Array, bits: int, block: int = BLOCK) -> Array:
    """Round each block of `block` numbers along a row to one of 2^bits - 1 evenly spaced steps.

    Each block keeps a scale, its biggest number divided by `top`, and every number becomes a
    whole number of scales between -top and top: 127 for 8 bits, 7 for 4 bits, and just -1, 0
    or 1 for 2 bits. Returned already multiplied back, the way it's used.
    """
    top = 2 ** (bits - 1) - 1
    blocks = rows.reshape(-1, block)
    scale = np.abs(blocks).max(axis=1, keepdims=True) / top
    scale[scale == 0] = 1.0
    return (np.clip(np.round(blocks / scale), -top, top) * scale).reshape(rows.shape)


@dataclass(frozen=True, slots=True)
class Shrunk:
    bits: int  # per weight, before counting the scales
    model: Transformer
    size: int  # bytes, scales included
    error: float  # how far the weights moved, relative to their size

    @property
    def label(self) -> str:
        return "bfloat16" if self.bits == 16 else f"{self.bits}-bit"


def shrink(model: Transformer, bits: int, block: int = BLOCK) -> Shrunk:
    """A copy of the model with every grid of weights rounded to `bits` bits.

    The norms' few numbers are kept in 16 bits either way, as real quantized files do.
    """
    params, size, moved, total = {}, 0, 0.0, 0.0
    for name, w in model.params.items():
        if bits == 16 or w.ndim == 1:
            params[name] = bfloat16(w)
            size += w.size * 2
        else:
            # Block along each row as a checkpoint stores it: a token's vector in the embedding
            # table, and one output's weights in every other grid (the columns of ours).
            if name == "embed":
                params[name] = quantize(w, bits, block)
            else:
                params[name] = quantize(w.T, bits, block).T
            size += w.size * bits // 8 + w.size // block * 2  # each scale is a 16-bit number
        moved += float(((params[name] - w) ** 2).sum())
        total += float((w**2).sum())
    return Shrunk(bits, Transformer(model.config, params), size, (moved / total) ** 0.5)
