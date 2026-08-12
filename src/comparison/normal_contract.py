"""Explicit source-normal encoding validation and decoding."""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_ground_truth_normal(
    encoded: Any,
    encoding: str,
    *,
    label: str = "ground-truth normal",
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    try:
        array = np.asarray(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric H,W,3 array") from exc
    if (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.issubdtype(array.dtype, np.number)
        or array.ndim != 3
        or array.shape[2] != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
    ):
        raise ValueError(f"{label} must be a numeric nonempty H,W,3 array")
    values = np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if encoding == "signed":
        return np.ascontiguousarray(values.copy(), dtype=np.float32)
    if encoding != "unsigned":
        raise ValueError("normal encoding must be one of: signed, unsigned")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("normal endpoint tolerance must be finite and non-negative")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"{label} unsigned values must be within [0, 1] "
            f"(tolerance {tolerance:g}); observed [{minimum:g}, {maximum:g}]"
        )
    clipped = np.clip(values, 0.0, 1.0)
    return np.ascontiguousarray(2.0 * clipped - 1.0, dtype=np.float32)


__all__ = ["decode_ground_truth_normal"]
