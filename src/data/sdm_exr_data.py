"""LINO's dependency-light dataset adapter for the SDM EXR contract.

The released ``DemoData``/``TestData`` loaders combine model masks and ground
truth normals.  This adapter keeps only model-input data in each batch so a
comparison runner can load ground truth later, after prediction, for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import (
    read_file_bytes,
    read_mask_exr_bytes,
    read_rgb_exr_bytes,
)
from src.comparison.manifest import DatasetManifest, ObjectRecord, stable_seed
from src.comparison.provenance import sha256_bytes
from .data_module import get_roi


_SAMPLE_FIELDS = frozenset({"imgs", "mask", "mask_original", "roi", "metadata"})
_TENSOR_FIELDS = ("imgs", "mask", "mask_original", "roi")


def _as_binary_mask(mask: np.ndarray, *, object_name: str) -> np.ndarray:
    """Return a finite binary float32 mask with at least one support pixel."""

    array = np.asarray(mask, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"mask geometry is invalid for {object_name}")
    if not np.isfinite(array).all():
        raise ValueError(f"mask contains non-finite values for {object_name}")
    binary = (array > 0).astype(np.float32, copy=False)
    if not np.any(binary > 0):
        raise ValueError(f"model support is empty for {object_name}")
    return np.ascontiguousarray(binary, dtype=np.float32)


def _safe_basename(value: Any) -> bool:
    """Accept only portable direct basenames, never path-like components."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if "\\" in value:
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        path.name == value
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
    )


def _resolve_within(path: Path, root: Path, *, label: str) -> Path:
    """Resolve a path and reject symlink/path traversal outside ``root``."""

    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes resolved root/object: {path}") from exc
    return resolved_path


class SdmExrDataset(Dataset):
    """Load one model-input sample per object from an ordered EXR manifest."""

    def __init__(self, config: SdmExrInferenceConfig, manifest: DatasetManifest):
        if not isinstance(config, SdmExrInferenceConfig):
            raise TypeError("config must be an SdmExrInferenceConfig")
        if not isinstance(manifest, DatasetManifest):
            raise TypeError("manifest must be a DatasetManifest")
        config_root = Path(config.data_root).resolve(strict=False)
        try:
            manifest_root = Path(manifest.data_root).resolve(strict=False)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("manifest data_root is invalid") from exc
        if manifest_root != config_root:
            raise ValueError(
                "manifest data_root does not match config data_root: "
                f"{manifest_root} != {config_root}"
            )
        self.config = config
        self.manifest = manifest
        self._data_root = config_root
        self.records = manifest.objects
        for record in self.records:
            self._validate_record(record)

    def _resolve_object_dir(self, record: ObjectRecord) -> Path:
        if not _safe_basename(record.name) or record.relative_dir != record.name:
            raise ValueError(f"manifest relative_dir is unsafe for {record.name}")
        return _resolve_within(
            self._data_root / record.relative_dir,
            self._data_root,
            label=f"manifest object {record.name}",
        )

    def _resolve_source_path(self, object_dir: Path, name: str, *, label: str) -> Path:
        if not _safe_basename(name):
            raise ValueError(f"manifest {label} is unsafe: {name!r}")
        candidate = object_dir / name
        resolved = _resolve_within(candidate, object_dir, label=f"manifest {label}")
        if candidate.is_symlink():
            raise ValueError(f"manifest {label} must be a regular non-symlink file: {candidate}")
        return resolved

    def _validate_record(self, record: ObjectRecord) -> None:
        if not isinstance(record, ObjectRecord):
            raise ValueError("manifest records must contain ObjectRecord values")
        if not _safe_basename(record.name):
            raise ValueError(f"manifest object name is unsafe: {record.name!r}")
        if not isinstance(record.relative_dir, str) or record.relative_dir != record.name:
            raise ValueError(f"manifest relative_dir is unsafe for {record.name}")
        object_dir = self._resolve_object_dir(record)

        if not isinstance(record.selected_images, (tuple, list)) or not record.selected_images:
            raise ValueError(f"manifest selected_images must be nonempty for {record.name}")
        if not isinstance(record.image_sha256, (tuple, list)):
            raise ValueError(f"manifest image_sha256 must be a sequence for {record.name}")
        if len(record.selected_images) != len(record.image_sha256):
            raise ValueError(
                f"manifest image/hash cardinality mismatch for {record.name}"
            )
        seen_images: set[str] = set()
        for image_name, image_digest in zip(record.selected_images, record.image_sha256):
            self._resolve_source_path(object_dir, image_name, label="selected image")
            if image_name in seen_images:
                raise ValueError(f"manifest selected_images contains duplicates for {record.name}")
            seen_images.add(image_name)
            if not isinstance(image_digest, str) or not image_digest:
                raise ValueError(f"manifest selected image digest is invalid for {record.name}")

        if self.config.mask_policy == "external":
            if not isinstance(record.mask_file, str) or not record.mask_file:
                raise ValueError(f"manifest external mask metadata is missing for {record.name}")
            self._resolve_source_path(object_dir, record.mask_file, label="external mask")
            if not isinstance(record.mask_sha256, str) or not record.mask_sha256:
                raise ValueError(f"manifest external mask digest is invalid for {record.name}")

    @staticmethod
    def _read_verified_snapshot(
        path: Path,
        expected: str,
        *,
        label: str,
    ) -> tuple[bytes, str]:
        """Read, hash, and retain one immutable source-file snapshot."""

        try:
            raw = read_file_bytes(path, label=label)
        except (OSError, ValueError) as exc:
            raise ValueError(f"failed to read {label}: {path}") from exc
        actual = sha256_bytes(raw)
        if actual != expected:
            raise ValueError(
                f"{label} digest mismatch for {path}: expected {expected}, got {actual}"
            )
        return raw, actual

    def __len__(self) -> int:
        return len(self.records)

    def _load_source_mask(self, record: ObjectRecord, object_dir: Path) -> tuple[np.ndarray, str, str | None]:
        if self.config.mask_policy == "external":
            # _validate_record has already checked both metadata fields; keep
            # this guard for callers invoking the helper directly.
            if not isinstance(record.mask_file, str) or not isinstance(record.mask_sha256, str):
                raise ValueError(f"external mask metadata is missing for {record.name}")
            mask_path = self._resolve_source_path(
                object_dir, record.mask_file, label="external mask"
            )
            if not mask_path.is_file():
                raise ValueError(f"external mask is missing for {record.name}: {mask_path}")
            mask_raw, mask_digest = self._read_verified_snapshot(
                mask_path, record.mask_sha256, label=f"external mask for {record.name}"
            )
            source_mask = read_mask_exr_bytes(
                mask_raw,
                label=f"external mask for {record.name}",
            )
            return source_mask, record.mask_file, mask_digest

        # A full-policy model must not receive the source/GT mask.  The source
        # geometry is supplied by the observation EXRs below.
        return np.empty((0, 0), dtype=np.float32), "full", None

    def _load_pre_normalization(self, record: ObjectRecord) -> dict[str, Any]:
        """Load and resize ordered images/mask before intensity normalization.

        The helper intentionally exposes the selected-image order and raw
        pre-normalization tensor for focused contract tests.  It is internal to
        the adapter; callers should use :meth:`__getitem__` for model batches.
        """

        object_dir = self._resolve_object_dir(record)
        if not object_dir.is_dir():
            raise ValueError(f"object directory is missing for {record.name}: {object_dir}")
        if not record.selected_images:
            raise ValueError(f"manifest selects no images for {record.name}")

        selected_paths: list[Path] = []
        images: list[np.ndarray] = []
        expected_shape = (int(record.height), int(record.width))
        if expected_shape[0] <= 0 or expected_shape[1] <= 0:
            raise ValueError(f"source geometry is invalid for {record.name}")
        image_digests: list[str] = []
        for image_name, image_digest in zip(record.selected_images, record.image_sha256):
            image_path = self._resolve_source_path(
                object_dir, image_name, label="selected image"
            )
            if not image_path.is_file():
                raise ValueError(f"selected image is missing for {record.name}: {image_path}")
            image_raw, verified_digest = self._read_verified_snapshot(
                image_path, image_digest, label=f"selected image for {record.name}"
            )
            image = read_rgb_exr_bytes(
                image_raw,
                label=f"selected image for {record.name}",
            )
            if image.shape[:2] != expected_shape:
                raise ValueError(
                    f"source geometry mismatch for {record.name}: "
                    f"{image_path} has {image.shape[:2]}, expected {expected_shape}"
                )
            images.append(np.asarray(image, dtype=np.float32))
            selected_paths.append(image_path)
            image_digests.append(verified_digest)

        h0, w0 = expected_shape
        if self.config.mask_policy == "external":
            source_mask, mask_source, mask_digest = self._load_source_mask(record, object_dir)
            if source_mask.shape != expected_shape:
                raise ValueError(
                    f"source geometry mismatch for {record.name}: external mask "
                    f"has {source_mask.shape}, expected {expected_shape}"
                )
            source_mask = _as_binary_mask(source_mask, object_name=record.name)
        else:
            source_mask = np.ones((h0, w0), dtype=np.float32)
            mask_source, mask_digest = "full", None

        # get_roi is the released loader's crop policy.  Passing the configured
        # margin explicitly avoids relying on its default and keeps external and
        # full masks on the same native geometry path.
        roi = np.asarray(get_roi(source_mask, margin=int(self.config.mask_margin)), dtype=np.int64)
        if roi.shape != (6,):
            raise ValueError(f"ROI geometry is invalid for {record.name}: {roi!r}")
        _, _, row_start, row_end, col_start, col_end = (int(value) for value in roi)
        if not (0 <= row_start < row_end <= h0 and 0 <= col_start < col_end <= w0):
            raise ValueError(f"ROI geometry is invalid for {record.name}: {roi.tolist()}")

        cropped_height = row_end - row_start
        cropped_width = col_end - col_start
        long_side = max(cropped_height, cropped_width)
        target = max(
            512,
            min(
                int(self.config.max_image_resolution),
                (long_side // 512) * 512,
            ),
        )
        if target <= 0 or target % 512:
            raise ValueError(f"resized geometry is invalid for {record.name}: {target}")

        resized_images = [
            np.asarray(
                cv2.resize(
                    image[row_start:row_end, col_start:col_end, :],
                    (target, target),
                    interpolation=cv2.INTER_CUBIC,
                ),
                dtype=np.float32,
            )
            for image in images
        ]
        # N,H,W,3 is convenient while loading; transpose to H,W,3,N before
        # normalization so foreground indexing remains straightforward.
        image_stack = np.stack(resized_images, axis=-1)
        cropped_mask = source_mask[row_start:row_end, col_start:col_end]
        resized_mask = np.asarray(
            cv2.resize(cropped_mask, (target, target), interpolation=cv2.INTER_NEAREST) > 0.5,
            dtype=np.float32,
        )
        resized_mask = _as_binary_mask(resized_mask, object_name=record.name)
        if not np.isfinite(image_stack).all():
            raise ValueError(f"selected images contain non-finite values for {record.name}")

        return {
            "images": np.ascontiguousarray(image_stack, dtype=np.float32),
            "mask": resized_mask,
            "source_mask": source_mask,
            "roi": roi,
            "selected_images": tuple(record.selected_images),
            "selected_paths": tuple(selected_paths),
            "image_digests": tuple(image_digests),
            "mask_source": mask_source,
            "mask_digest": mask_digest,
            "source_geometry": {"height": h0, "width": w0},
            "resized_geometry": {"height": int(target), "width": int(target)},
        }

    def _normalize_images(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        record: ObjectRecord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply deterministic released-style mean-to-maximum normalization."""

        image_array = np.asarray(images, dtype=np.float32)
        mask_array = _as_binary_mask(mask, object_name=record.name)
        support = mask_array > 0
        foreground = image_array[support]
        if foreground.size == 0:
            raise ValueError(f"model support is empty for {record.name}")
        # Foreground mean-RGB intensity per pixel, then spatial mean/max per
        # light.  The mask here is the policy-specific resized model mask.
        intensity = np.mean(foreground, axis=1, dtype=np.float64)
        spatial_mean = np.mean(intensity, axis=0, dtype=np.float64)
        spatial_max = np.max(intensity, axis=0)
        light_count = image_array.shape[-1]
        alpha = np.random.default_rng(
            stable_seed(self.config.seed, record.name, "lino_normalization")
        ).random(light_count)
        scales = (1.0 - alpha) * spatial_mean + alpha * spatial_max
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError(f"normalization scales must be finite and positive for {record.name}")
        normalized = image_array / (scales.reshape(1, 1, 1, light_count) + 1.0e-6)
        normalized = np.asarray(normalized, dtype=np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError(f"normalized images contain non-finite values for {record.name}")
        return normalized, np.asarray(alpha, dtype=np.float64), np.asarray(scales, dtype=np.float64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        prepared = self._load_pre_normalization(record)
        normalized, alpha, scales = self._normalize_images(
            prepared["images"], prepared["mask"], record
        )

        alpha_values = [float(value) for value in alpha]
        scale_values = [float(value) for value in scales]
        roi_values = [int(value) for value in prepared["roi"]]
        metadata: dict[str, Any] = {
            "object_name": record.name,
            "object_relative_dir": record.relative_dir,
            "selected_images": list(prepared["selected_images"]),
            "selected_image_sha256": list(prepared["image_digests"]),
            "selected_image_digests": list(prepared["image_digests"]),
            "source_geometry": dict(prepared["source_geometry"]),
            "roi": roi_values,
            "resized_geometry": dict(prepared["resized_geometry"]),
            "mask_policy": self.config.mask_policy,
            "mask_source": prepared["mask_source"],
            "mask_digest": prepared["mask_digest"],
            "normalization": {"alpha": alpha_values, "scales": scale_values},
            "normalization_alpha": alpha_values,
            "normalization_scales": scale_values,
            "seed": int(self.config.seed),
            "normalization_seed": int(
                stable_seed(self.config.seed, record.name, "lino_normalization")
            ),
        }
        sample: dict[str, Any] = {
            "imgs": torch.from_numpy(normalized.transpose(2, 0, 1, 3).copy()),
            "mask": torch.from_numpy(np.asarray(prepared["mask"][None], dtype=np.float32).copy()),
            "mask_original": torch.from_numpy(
                np.asarray(prepared["source_mask"][None], dtype=np.float32).copy()
            ),
            "roi": torch.tensor(roi_values, dtype=torch.int64),
            "metadata": metadata,
        }
        if set(sample) != _SAMPLE_FIELDS:  # pragma: no cover - defensive contract guard
            raise RuntimeError("internal sample contract violation")
        return sample


def collate_single_sdm_exr(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate exactly one sample and add a leading batch dimension."""

    if len(samples) != 1:
        raise ValueError("collate_single_sdm_exr requires exactly one sample")
    sample = samples[0]
    if not isinstance(sample, Mapping):
        raise TypeError("SDM-EXR samples must be mappings")
    if set(sample) != _SAMPLE_FIELDS:
        raise ValueError("SDM-EXR sample fields do not match the adapter contract")
    result: dict[str, Any] = {
        field: sample[field].unsqueeze(0) for field in _TENSOR_FIELDS
    }
    # Preserve the exact metadata mapping; callers may attach provenance to it
    # after collation without a hidden copy.
    result["metadata"] = sample["metadata"]
    return result


__all__ = ["SdmExrDataset", "collate_single_sdm_exr"]
