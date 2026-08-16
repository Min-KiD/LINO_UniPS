"""Manifest-bound preflight validation for private LINO transfer sources."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from .config import SdmExrInferenceConfig
from .exr_io import read_file_bytes, read_mask_exr_bytes, read_rgb_exr_bytes
from .manifest import DatasetManifest, ObjectRecord
from .metrics import (
    GT_VALIDITY_POLICY,
    angular_metrics,
    load_source_gt,
    normal_validity_mask,
)
from .provenance import sha256_bytes


def _safe_basename(value: str) -> bool:
    path = Path(value)
    windows = PureWindowsPath(value)
    return bool(
        value
        and value not in {".", ".."}
        and "\\" not in value
        and path.name == value
        and not path.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
    )


def _verified_object_file(
    config: SdmExrInferenceConfig,
    record: ObjectRecord,
    basename: str,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, bytes]:
    if not _safe_basename(basename):
        raise ValueError(f"{label} is not a safe object-relative basename for {record.name}")
    root = Path(config.data_root).resolve(strict=True)
    object_dir = (root / record.relative_dir).resolve(strict=True)
    source = object_dir / basename
    try:
        object_dir.relative_to(root)
        source.resolve(strict=True).relative_to(object_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{label} escapes the manifest object for {record.name}") from exc
    raw = read_file_bytes(source, label=f"{label} for {record.name}")
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError(f"{label} digest mismatch for {record.name}")
    return source, raw


def preflight_transfer_sources(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
) -> dict[str, dict[str, Any]]:
    if Path(manifest.data_root).resolve(strict=False) != Path(config.data_root).resolve(
        strict=False
    ):
        raise ValueError("manifest data_root does not match transfer config data_root")
    expected_names = [record.name for record in manifest.objects]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("manifest object names must be unique")

    reports: dict[str, dict[str, Any]] = {}
    for record in manifest.objects:
        geometry = (int(record.height), int(record.width))
        if (
            config.expected_source_geometry is not None
            and geometry != config.expected_source_geometry
        ):
            raise ValueError(
                f"expected source geometry {config.expected_source_geometry} for "
                f"{record.name}, got {geometry}"
            )
        if len(record.selected_images) != config.max_image_num:
            raise ValueError(
                f"selected observation count mismatch for {record.name}: "
                f"{len(record.selected_images)} != {config.max_image_num}"
            )
        if len(record.image_sha256) != len(record.selected_images):
            raise ValueError(f"selected observation/hash count mismatch for {record.name}")

        observation_stats: list[dict[str, Any]] = []
        for filename, expected_digest in zip(record.selected_images, record.image_sha256):
            _, raw = _verified_object_file(
                config,
                record,
                filename,
                expected_digest,
                label="selected observation",
            )
            image = read_rgb_exr_bytes(raw, label=f"selected observation for {record.name}")
            if image.shape != (record.height, record.width, 3):
                raise ValueError(f"selected observation geometry mismatch for {record.name}")
            observation_stats.append(
                {
                    "filename": filename,
                    "sha256": expected_digest,
                    "minimum": float(np.min(image)),
                    "maximum": float(np.max(image)),
                    "mean": float(np.mean(image, dtype=np.float64)),
                }
            )

        gt, _ = load_source_gt(config, record)
        gt_support = normal_validity_mask(gt)
        gt_count = int(np.count_nonzero(gt_support))
        if gt_count == 0:
            raise ValueError(f"empty decoded GT support for {record.name}")

        mask_count: int | None = None
        intersection_count: int | None = None
        gt_outside_count: int | None = None
        mask_only_count: int | None = None
        if config.mask_policy == "external":
            if record.mask_file is None or record.mask_sha256 is None:
                raise ValueError(f"external mask metadata is missing for {record.name}")
            _, raw = _verified_object_file(
                config,
                record,
                record.mask_file,
                record.mask_sha256,
                label="external mask",
            )
            mask = read_mask_exr_bytes(raw, label=f"external mask for {record.name}") > 0
            if mask.shape != geometry:
                raise ValueError(f"external mask geometry mismatch for {record.name}")
            mask_count = int(np.count_nonzero(mask))
            if mask_count == 0:
                raise ValueError(f"external mask is empty for {record.name}")
            intersection_count = int(np.count_nonzero(gt_support & mask))
            gt_outside_count = int(np.count_nonzero(gt_support & ~mask))
            mask_only_count = int(np.count_nonzero(mask & ~gt_support))
            if gt_outside_count:
                raise ValueError(
                    f"{gt_outside_count} GT-valid pixel(s) lie outside the external mask "
                    f"for {record.name}"
                )

        reports[record.name] = {
            "source_geometry": {"height": geometry[0], "width": geometry[1]},
            "selected_observations": observation_stats,
            "decoded_gt_valid_pixel_count": gt_count,
            "gt_validity_policy": GT_VALIDITY_POLICY,
            "external_mask_pixel_count": mask_count,
            "intersection_pixel_count": intersection_count,
            "gt_outside_mask_pixel_count": gt_outside_count,
            "mask_only_pixel_count": mask_only_count,
        }
    return reports


_AXIS_NAMES = ("x", "y", "z")


def _coordinate_specs() -> tuple[tuple[str, tuple[int, int, int], tuple[int, int, int]], ...]:
    specs = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            label = ",".join(
                f"{'+' if sign > 0 else '-'}{_AXIS_NAMES[index]}"
                for index, sign in zip(permutation, signs)
            )
            specs.append((label, permutation, signs))
    identity = ("+x,+y,+z", (0, 1, 2), (1, 1, 1))
    return (identity,) + tuple(spec for spec in specs if spec != identity)


_COORDINATE_SPECS = _coordinate_specs()


def coordinate_transform_maes(gt: Any, prediction: Any, support: Any) -> dict[str, float]:
    prediction_array = np.asarray(prediction)
    results: dict[str, float] = {}
    for label, permutation, signs in _COORDINATE_SPECS:
        transformed = np.take(prediction_array, permutation, axis=2) * np.asarray(
            signs, dtype=np.float64
        )
        results[label] = float(angular_metrics(gt, transformed, support)["mae"])
    return results


def constant_front_facing_mae(gt: Any, support: Any) -> float:
    gt_array = np.asarray(gt)
    baseline = np.zeros(gt_array.shape, dtype=np.float32)
    baseline[..., 2] = 1.0
    return float(angular_metrics(gt_array, baseline, support)["mae"])


def _transform_payload(label: str) -> dict[str, Any]:
    for candidate, permutation, signs in _COORDINATE_SPECS:
        if candidate == label:
            return {
                "label": candidate,
                "permutation": [_AXIS_NAMES[index] for index in permutation],
                "signs": list(signs),
            }
    raise ValueError(f"unknown coordinate transform: {label}")


def summarize_transfer_metrics(
    identity_maes: Sequence[float],
    baseline_maes: Sequence[float],
    coordinate_sweeps: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    object_count = len(identity_maes)
    if object_count == 0 or len(baseline_maes) != object_count or len(coordinate_sweeps) != object_count:
        raise ValueError("transfer metric sequences must be nonempty and equally sized")
    expected_labels = tuple(label for label, _, _ in _COORDINATE_SPECS)
    if any(tuple(sweep) != expected_labels for sweep in coordinate_sweeps):
        raise ValueError("coordinate sweeps must contain the canonical 48 transforms in order")
    macro_by_transform = {
        label: float(sum(float(sweep[label]) for sweep in coordinate_sweeps) / object_count)
        for label in expected_labels
    }
    best_label = min(expected_labels, key=lambda label: macro_by_transform[label])
    return {
        "object_count": object_count,
        "identity_macro_mae": float(sum(map(float, identity_maes)) / object_count),
        "constant_normal_macro_mae": float(sum(map(float, baseline_maes)) / object_count),
        "best_coordinate_macro_mae": macro_by_transform[best_label],
        "best_coordinate_transform": _transform_payload(best_label),
        "evaluated_coordinate_transform_count": len(expected_labels),
    }


__all__ = [
    "preflight_transfer_sources",
    "coordinate_transform_maes",
    "constant_front_facing_mae",
    "summarize_transfer_metrics",
]
