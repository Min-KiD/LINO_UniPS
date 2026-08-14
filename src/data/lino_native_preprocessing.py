"""Shared, versioned preprocessing for LINO observations and geometry.

The released transfer path and private training use the same crop/resize and
mean-to-maximum normalization primitives, but intentionally derive their
random streams differently.  Keeping the two seed algorithms named at the
call sites makes it possible to evolve private training without changing the
released checkpoint contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.comparison.manifest import stable_seed as legacy_manifest_seed
from src.data.data_module import get_roi
from src.training.reproducibility import stable_seed


RELEASED_TRANSFER_VERSION = "released_transfer_v1"
PRIVATE_EXTERNAL_VERSION = "private_external_lino_native_v1"
_SUPPORTED_VERSIONS = frozenset({RELEASED_TRANSFER_VERSION, PRIVATE_EXTERNAL_VERSION})


@dataclass(frozen=True)
class PreparedLinoGeometry:
    """Source and model-space fields sharing one native LINO ROI."""

    images: np.ndarray
    model_mask: np.ndarray
    source_model_mask: np.ndarray
    roi: np.ndarray
    target_normal: np.ndarray | None
    target_mask: np.ndarray | None
    source_target_normal: np.ndarray | None
    source_target_mask: np.ndarray | None


@dataclass(frozen=True)
class NormalizedLinoObservations:
    """Normalized model observations and the exact randomization metadata."""

    images: np.ndarray
    alpha: np.ndarray
    scales: np.ndarray
    seed: int


def _validate_version(version: str) -> str:
    if not isinstance(version, str) or version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported preprocessing_version: {version}")
    return version


def _as_finite_float32(value: Any, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_model_mask(mask: Any, *, shape: tuple[int, int], label: str) -> np.ndarray:
    array = _as_finite_float32(mask, label=label)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}")
    binary = np.asarray(array > 0, dtype=np.float32)
    if not np.any(binary > 0):
        raise ValueError(f"{label} support is empty")
    return np.ascontiguousarray(binary, dtype=np.float32)


def _as_target_mask(mask: Any, *, shape: tuple[int, int]) -> np.ndarray:
    array = _as_finite_float32(mask, label="target_mask")
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"target_mask must have shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array > 0, dtype=np.float32)


def _normalize_target(normal: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Normalize normals on target support and keep background exactly zero."""

    lengths = np.linalg.norm(normal.astype(np.float64), axis=2, keepdims=True)
    output = np.zeros_like(normal, dtype=np.float32)
    np.divide(
        normal,
        lengths,
        out=output,
        where=(support[..., None] > 0) & (lengths > 0),
    )
    return np.ascontiguousarray(output, dtype=np.float32)


def _validate_roi(roi: np.ndarray, *, source_shape: tuple[int, int]) -> np.ndarray:
    if roi.shape != (6,):
        raise ValueError(f"ROI geometry is invalid: {roi!r}")
    h0, w0 = source_shape
    values = tuple(int(value) for value in roi)
    if values[:2] != (h0, w0):
        raise ValueError(f"ROI source geometry is invalid: {roi.tolist()}")
    _, _, row_start, row_end, col_start, col_end = values
    if not (0 <= row_start < row_end <= h0 and 0 <= col_start < col_end <= w0):
        raise ValueError(f"ROI geometry is invalid: {roi.tolist()}")
    return np.asarray(values, dtype=np.int64)


def _validate_resolution(target_resolution: int, *, version: str) -> int:
    if isinstance(target_resolution, bool) or not isinstance(target_resolution, int):
        raise ValueError("target_resolution must be an integer")
    if target_resolution < 512 or target_resolution % 512:
        raise ValueError("target_resolution must be at least 512 and divisible by 512")
    if version == PRIVATE_EXTERNAL_VERSION and target_resolution != 512:
        raise ValueError(
            "private_external_lino_native_v1 requires target_resolution 512"
        )
    return target_resolution


def prepare_lino_native_geometry(
    images: Any,
    model_mask: Any,
    *,
    target_normal: Any | None = None,
    target_mask: Any | None = None,
    margin: int = 8,
    target_resolution: int = 512,
    object_name: str = "object",
    preprocessing_version: str = PRIVATE_EXTERNAL_VERSION,
) -> PreparedLinoGeometry:
    """Crop and resize all LINO fields using one released ROI policy.

    ``model_mask`` controls model support and ROI selection.  Ground-truth
    ``target_mask`` is carried independently, so an external mask halo remains
    available to the model without becoming target support.
    """

    version = _validate_version(preprocessing_version)
    resolution = _validate_resolution(target_resolution, version=version)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("margin must be a non-negative integer")

    image_array = _as_finite_float32(images, label="images")
    if (
        image_array.ndim != 4
        or image_array.shape[0] == 0
        or image_array.shape[1] == 0
        or image_array.shape[2] != 3
        or image_array.shape[3] == 0
    ):
        raise ValueError("images must be a nonempty H,W,3,N array")
    source_shape = (int(image_array.shape[0]), int(image_array.shape[1]))
    source_model_mask = _as_model_mask(
        model_mask,
        shape=source_shape,
        label="model_mask",
    )

    source_target_mask: np.ndarray | None
    if target_mask is None:
        source_target_mask = None
    else:
        source_target_mask = _as_target_mask(target_mask, shape=source_shape)

    source_target_normal: np.ndarray | None
    if target_normal is None:
        source_target_normal = None
    else:
        source_target_normal = _as_finite_float32(target_normal, label="target_normal")
        if source_target_normal.ndim != 3 or source_target_normal.shape != (*source_shape, 3):
            raise ValueError(
                "target_normal must have shape "
                f"{(*source_shape, 3)}, got {source_target_normal.shape}"
            )
        if source_target_mask is None:
            source_target_mask = np.asarray(
                np.linalg.norm(source_target_normal.astype(np.float64), axis=2) > 0,
                dtype=np.float32,
            )

    if source_target_mask is not None:
        outside_support = (source_target_mask > 0) & (source_model_mask <= 0)
        outside_count = int(np.count_nonzero(outside_support))
        if outside_count:
            raise ValueError(
                f"{object_name}: target_mask has {outside_count} pixel(s) "
                "outside model support"
            )

    roi = _validate_roi(
        np.asarray(get_roi(source_model_mask, margin=margin), dtype=np.int64),
        source_shape=source_shape,
    )
    _, _, row_start, row_end, col_start, col_end = (int(value) for value in roi)
    crop = image_array[row_start:row_end, col_start:col_end, :, :]
    resized_images = np.stack(
        [
            cv2.resize(
                crop[..., index],
                (resolution, resolution),
                interpolation=cv2.INTER_CUBIC,
            )
            for index in range(image_array.shape[3])
        ],
        axis=-1,
    )
    model_mask_crop = source_model_mask[row_start:row_end, col_start:col_end]
    resized_model_mask = np.asarray(
        cv2.resize(
            model_mask_crop,
            (resolution, resolution),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0.5,
        dtype=np.float32,
    )

    target_mask_resized: np.ndarray | None = None
    target_normal_resized: np.ndarray | None = None
    if source_target_mask is not None:
        target_mask_resized = np.asarray(
            cv2.resize(
                source_target_mask[row_start:row_end, col_start:col_end],
                (resolution, resolution),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0.5,
            dtype=np.float32,
        )
    if source_target_normal is not None:
        target_normal_resized = np.asarray(
            cv2.resize(
                source_target_normal[row_start:row_end, col_start:col_end, :],
                (resolution, resolution),
                interpolation=cv2.INTER_LINEAR,
            ),
            dtype=np.float32,
        )
        # A normal is meaningful only where the independently decoded target
        # support says it is valid; interpolation never expands that support.
        assert target_mask_resized is not None
        target_normal_resized = _normalize_target(
            target_normal_resized,
            target_mask_resized,
        )

    return PreparedLinoGeometry(
        images=np.ascontiguousarray(resized_images, dtype=np.float32),
        model_mask=np.ascontiguousarray(resized_model_mask, dtype=np.float32),
        source_model_mask=np.ascontiguousarray(source_model_mask, dtype=np.float32),
        roi=roi,
        target_normal=target_normal_resized,
        target_mask=target_mask_resized,
        source_target_normal=(
            None
            if source_target_normal is None
            else np.ascontiguousarray(source_target_normal, dtype=np.float32)
        ),
        source_target_mask=(
            None
            if source_target_mask is None
            else np.ascontiguousarray(source_target_mask, dtype=np.float32)
        ),
    )


def _normalization_seed(
    base_seed: int,
    split: str,
    epoch: int,
    object_name: str,
    version: str,
) -> int:
    version_name = _validate_version(version)
    if version_name == RELEASED_TRANSFER_VERSION:
        return legacy_manifest_seed(base_seed, object_name, "lino_normalization")
    effective_epoch = epoch if split.lower() == "train" else 0
    return stable_seed(
        base_seed,
        split.lower(),
        effective_epoch,
        object_name,
        "lino_normalization",
    )


def normalize_lino_observations(
    images: Any,
    mask: Any,
    *,
    base_seed: int,
    split: str,
    epoch: int,
    object_name: str,
    version: str,
) -> NormalizedLinoObservations:
    """Apply deterministic mean-to-maximum intensity normalization."""

    version_name = _validate_version(version)
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not isinstance(object_name, str) or not object_name:
        raise ValueError("object_name must be a non-empty string")

    image_array = _as_finite_float32(images, label="images")
    if (
        image_array.ndim != 4
        or image_array.shape[0] == 0
        or image_array.shape[1] == 0
        or image_array.shape[2] != 3
        or image_array.shape[3] == 0
    ):
        raise ValueError("images must be a nonempty H,W,3,N array")
    mask_array = _as_model_mask(
        mask,
        shape=(int(image_array.shape[0]), int(image_array.shape[1])),
        label="mask",
    )
    foreground = image_array[mask_array > 0]
    if foreground.size == 0:  # pragma: no cover - _as_model_mask guards this
        raise ValueError("mask support is empty")
    intensity = np.mean(foreground, axis=1, dtype=np.float64)
    spatial_mean = np.mean(intensity, axis=0, dtype=np.float64)
    spatial_max = np.max(intensity, axis=0)
    seed = _normalization_seed(
        base_seed,
        split,
        epoch,
        object_name,
        version_name,
    )
    alpha = np.random.default_rng(seed).random(image_array.shape[-1])
    scales = (1.0 - alpha) * spatial_mean + alpha * spatial_max
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("normalization scales must be finite and positive")
    normalized = np.asarray(
        image_array / (scales.reshape(1, 1, 1, image_array.shape[-1]) + 1.0e-6),
        dtype=np.float32,
    )
    if not np.isfinite(normalized).all():
        raise ValueError("normalized images contain non-finite values")
    return NormalizedLinoObservations(
        images=np.ascontiguousarray(normalized, dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
        seed=int(seed),
    )


def _normalize_all_nonzero(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values.astype(np.float64), axis=2, keepdims=True)
    normalized = np.zeros_like(values, dtype=np.float32)
    np.divide(values, lengths, out=normalized, where=lengths > 0)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def restore_lino_prediction(
    prediction: Any,
    roi: Any,
    *,
    source_height: int,
    source_width: int,
) -> np.ndarray:
    """Restore a model-space normal prediction to source image geometry."""

    if (
        isinstance(source_height, bool)
        or isinstance(source_width, bool)
        or not isinstance(source_height, int)
        or not isinstance(source_width, int)
        or source_height <= 0
        or source_width <= 0
    ):
        raise ValueError("source geometry must be positive integer dimensions")
    values = _as_finite_float32(prediction, label="prediction")
    if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("prediction must be a nonempty H,W,3 array")
    roi_array = np.asarray(roi, dtype=np.int64)
    if roi_array.shape != (6,):
        raise ValueError(f"ROI geometry is invalid: {roi!r}")
    row_start, row_end, col_start, col_end = map(int, roi_array[2:])
    if not (0 <= row_start < row_end <= source_height and 0 <= col_start < col_end <= source_width):
        raise ValueError(f"ROI geometry is invalid: {roi_array.tolist()}")
    cropped = cv2.resize(
        values,
        (col_end - col_start, row_end - row_start),
        interpolation=cv2.INTER_AREA,
    )
    restored = np.zeros((source_height, source_width, 3), dtype=np.float32)
    restored[row_start:row_end, col_start:col_end] = _normalize_all_nonzero(cropped)
    return restored


__all__ = [
    "NormalizedLinoObservations",
    "PreparedLinoGeometry",
    "normalize_lino_observations",
    "prepare_lino_native_geometry",
    "restore_lino_prediction",
]
