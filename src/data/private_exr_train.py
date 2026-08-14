"""Epoch-aware, manifest-verified private EXR training dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.comparison.exr_io import (
    read_file_bytes,
    read_mask_exr_bytes,
    read_rgb_exr_bytes,
)
from src.comparison.metrics import normal_validity_mask
from src.comparison.normal_contract import decode_ground_truth_normal
from src.comparison.provenance import sha256_bytes
from src.data.lino_native_preprocessing import (
    normalize_lino_observations,
    prepare_lino_native_geometry,
)
from src.training.config import PrivateTrainConfig
from src.training.private_manifest import PrivateObjectRecord, PrivateSplitManifest
from src.training.reproducibility import select_observation_names


EXPECTED_FIELDS = frozenset(
    {
        "imgs",
        "model_mask",
        "target_normal",
        "target_mask",
        "source_target_normal",
        "source_target_mask",
        "source_model_mask",
        "roi",
        "metadata",
    }
)
_TENSOR_FIELDS = (
    "imgs",
    "model_mask",
    "target_normal",
    "target_mask",
    "source_target_normal",
    "source_target_mask",
    "source_model_mask",
    "roi",
)


def _safe_basename(value: Any) -> bool:
    """Accept one portable direct filename component only."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if "\\" in value or "\x00" in value:
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
    """Resolve a path and reject traversal or a symlink escape."""

    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes resolved root/object: {path}") from exc
    return resolved_path


def _finite_source_shape(record: PrivateObjectRecord, config: PrivateTrainConfig) -> tuple[int, int]:
    expected = tuple(int(value) for value in config.expected_source_geometry)
    shape = (int(record.height), int(record.width))
    if shape != expected or any(value <= 0 for value in shape):
        raise ValueError(
            f"manifest source geometry is invalid for {record.name}: "
            f"{shape}, expected {expected}"
        )
    return expected


class PrivateExrTrainDataset(Dataset):
    """Load one private LINO training sample from a verified split manifest."""

    def __init__(
        self,
        config: PrivateTrainConfig,
        manifest: PrivateSplitManifest,
        *,
        split: str,
    ) -> None:
        if not isinstance(config, PrivateTrainConfig):
            raise TypeError("config must be a PrivateTrainConfig")
        if not isinstance(manifest, PrivateSplitManifest):
            raise TypeError("manifest must be a PrivateSplitManifest")
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        if manifest.split != split:
            raise ValueError(
                f"manifest split does not match dataset split: {manifest.split!r} != {split!r}"
            )
        if manifest.version != 1:
            raise ValueError(f"unsupported private split manifest version: {manifest.version}")

        configured_root = config.train_dir if split == "train" else config.test_dir
        config_root = Path(configured_root).resolve(strict=False)
        try:
            manifest_root = Path(manifest.data_root).resolve(strict=False)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError("manifest data_root is invalid") from exc
        if manifest_root != config_root:
            raise ValueError(
                "manifest data_root does not match config split root: "
                f"{manifest_root} != {config_root}"
            )

        self.config = config
        self.manifest = manifest
        self.split = split
        self.epoch = 0
        self._data_root = config_root
        self.records = manifest.objects
        for record in self.records:
            self._validate_record(record)

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
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"failed to read {label}: {path}") from exc
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise ValueError(f"{label} reader did not return immutable bytes: {path}")
        immutable = bytes(raw)
        actual = sha256_bytes(immutable)
        if actual != expected:
            raise ValueError(
                f"{label} digest mismatch for {path}: expected {expected}, got {actual}"
            )
        return immutable, actual

    def _validate_record(self, record: PrivateObjectRecord) -> None:
        if not isinstance(record, PrivateObjectRecord):
            raise ValueError("manifest records must contain PrivateObjectRecord values")
        if not _safe_basename(record.name):
            raise ValueError(f"manifest object name is unsafe: {record.name!r}")
        if not isinstance(record.relative_dir, str) or record.relative_dir != record.name:
            raise ValueError(f"manifest relative_dir is unsafe for {record.name}")
        _finite_source_shape(record, self.config)

        observations = record.observation_files
        digests = record.observation_sha256
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise ValueError(f"manifest observation_files must be a sequence for {record.name}")
        if not isinstance(digests, Sequence) or isinstance(digests, (str, bytes)):
            raise ValueError(f"manifest observation_sha256 must be a sequence for {record.name}")
        if len(observations) != len(digests):
            raise ValueError(f"manifest observation/hash cardinality mismatch for {record.name}")
        if len(observations) < self.config.max_image_num:
            raise ValueError(
                f"manifest has only {len(observations)} observations for {record.name}; "
                f"requires {self.config.max_image_num}"
            )
        seen: set[str] = set()
        for filename, digest in zip(observations, digests):
            if not _safe_basename(filename):
                raise ValueError(f"manifest observation filename is unsafe for {record.name}: {filename!r}")
            if filename in seen:
                raise ValueError(f"manifest observation filenames contain duplicates for {record.name}")
            seen.add(filename)
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"manifest observation digest is invalid for {record.name}")

        for label, filename, digest in (
            ("ground truth", record.normal_file, record.normal_sha256),
            ("external mask", record.mask_file, record.mask_sha256),
        ):
            if not _safe_basename(filename):
                raise ValueError(f"manifest {label} filename is unsafe for {record.name}")
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"manifest {label} digest is invalid for {record.name}")

        self._resolve_object_dir(record)

    def _resolve_object_dir(self, record: PrivateObjectRecord) -> Path:
        if not _safe_basename(record.name) or record.relative_dir != record.name:
            raise ValueError(f"manifest relative_dir is unsafe for {record.name}")
        candidate = self._data_root / record.relative_dir
        resolved = _resolve_within(candidate, self._data_root, label=f"manifest object {record.name}")
        if candidate.is_symlink():
            raise ValueError(f"manifest object must be a regular non-symlink directory: {candidate}")
        if not resolved.is_dir():
            raise ValueError(f"manifest object directory is missing for {record.name}: {candidate}")
        return resolved

    def _resolve_source_path(self, object_dir: Path, name: str, *, label: str) -> Path:
        if not _safe_basename(name):
            raise ValueError(f"manifest {label} is unsafe: {name!r}")
        candidate = object_dir / name
        resolved = _resolve_within(candidate, object_dir, label=f"manifest {label}")
        if candidate.is_symlink():
            raise ValueError(f"manifest {label} must be a regular non-symlink file: {candidate}")
        return resolved

    def _selected_names(self, record: PrivateObjectRecord) -> tuple[str, ...]:
        return select_observation_names(
            record.observation_files,
            count=self.config.max_image_num,
            base_seed=self.config.seed,
            split=self.split,
            epoch=self.epoch,
            object_name=record.name,
        )

    def _effective_epoch(self) -> int:
        return self.epoch if self.split == "train" else 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _load_verified_sources(self, record: PrivateObjectRecord) -> dict[str, Any]:
        object_dir = self._resolve_object_dir(record)
        source_shape = _finite_source_shape(record, self.config)
        digest_by_name = dict(zip(record.observation_files, record.observation_sha256))
        selected_names = self._selected_names(record)

        images: list[np.ndarray] = []
        selected_digests: list[str] = []
        for filename in selected_names:
            image_path = self._resolve_source_path(object_dir, filename, label="selected observation")
            image_raw, image_digest = self._read_verified_snapshot(
                image_path,
                digest_by_name[filename],
                label=f"selected observation {filename} for {record.name} in {self.split}",
            )
            try:
                image = read_rgb_exr_bytes(
                    image_raw,
                    label=f"selected observation {filename} for {record.name} in {self.split}",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"failed to decode selected observation {filename} for {record.name} in {self.split}"
                ) from exc
            if image.shape[:2] != source_shape:
                raise ValueError(
                    f"selected observation {filename} geometry mismatch for {record.name}: "
                    f"{image.shape[:2]}, expected {source_shape}"
                )
            images.append(np.ascontiguousarray(image, dtype=np.float32))
            selected_digests.append(image_digest)

        normal_path = self._resolve_source_path(object_dir, record.normal_file, label="ground truth")
        normal_raw, normal_digest = self._read_verified_snapshot(
            normal_path,
            record.normal_sha256,
            label=f"ground truth {record.normal_file} for {record.name} in {self.split}",
        )
        try:
            encoded_normal = read_rgb_exr_bytes(
                normal_raw,
                label=f"ground truth {record.normal_file} for {record.name} in {self.split}",
            )
            if encoded_normal.shape[:2] != source_shape:
                raise ValueError(
                    f"ground truth {record.normal_file} geometry mismatch for {record.name}: "
                    f"{encoded_normal.shape[:2]}, expected {source_shape}"
                )
            target_normal = decode_ground_truth_normal(
                encoded_normal,
                self.config.normal_encoding,
                label=f"ground truth {record.normal_file} for {record.name} in {self.split}",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if "geometry mismatch" in str(exc):
                raise
            raise ValueError(
                f"failed to decode ground truth {record.normal_file} for {record.name} in {self.split}"
            ) from exc
        target_mask = np.asarray(normal_validity_mask(target_normal), dtype=np.float32)

        mask_path = self._resolve_source_path(object_dir, record.mask_file, label="external mask")
        mask_raw, mask_digest = self._read_verified_snapshot(
            mask_path,
            record.mask_sha256,
            label=f"external mask {record.mask_file} for {record.name} in {self.split}",
        )
        try:
            model_mask = read_mask_exr_bytes(
                mask_raw,
                label=f"external mask {record.mask_file} for {record.name} in {self.split}",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"failed to decode external mask {record.mask_file} for {record.name} in {self.split}"
            ) from exc
        if model_mask.shape != source_shape:
            raise ValueError(
                f"external mask {record.mask_file} geometry mismatch for {record.name}: "
                f"{model_mask.shape}, expected {source_shape}"
            )

        return {
            "images": np.ascontiguousarray(np.stack(images, axis=-1), dtype=np.float32),
            "model_mask": np.ascontiguousarray(model_mask, dtype=np.float32),
            "target_normal": np.ascontiguousarray(target_normal, dtype=np.float32),
            "target_mask": np.ascontiguousarray(target_mask, dtype=np.float32),
            "selected_images": selected_names,
            "selected_digests": tuple(selected_digests),
            "normal_digest": normal_digest,
            "mask_digest": mask_digest,
            "source_geometry": {"height": source_shape[0], "width": source_shape[1]},
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sources = self._load_verified_sources(record)
        geometry = prepare_lino_native_geometry(
            sources["images"],
            sources["model_mask"],
            target_normal=sources["target_normal"],
            target_mask=sources["target_mask"],
            margin=int(self.config.mask_margin),
            target_resolution=int(self.config.max_image_resolution),
            object_name=record.name,
            preprocessing_version=self.config.preprocessing_version,
        )
        normalized = normalize_lino_observations(
            geometry.images,
            geometry.model_mask,
            base_seed=int(self.config.seed),
            split=self.split,
            epoch=self.epoch,
            object_name=record.name,
            version=self.config.preprocessing_version,
        )

        effective_epoch = self._effective_epoch()
        selected_names = tuple(sources["selected_images"])
        selected_digests = tuple(sources["selected_digests"])
        roi_values = [int(value) for value in geometry.roi]
        source_geometry = sources["source_geometry"]
        resized_geometry = {
            "height": int(geometry.images.shape[0]),
            "width": int(geometry.images.shape[1]),
        }
        mask_counts = {
            "source_model": int(np.count_nonzero(geometry.source_model_mask > 0)),
            "model": int(np.count_nonzero(geometry.model_mask > 0)),
            "source_target": int(np.count_nonzero(geometry.source_target_mask > 0)),
            "target": int(np.count_nonzero(geometry.target_mask > 0)),
            "gt_valid_pixels": int(record.gt_valid_pixels),
            "mask_valid_pixels": int(record.mask_valid_pixels),
            "mask_only_pixels": int(record.mask_only_pixels),
        }
        alpha = [float(value) for value in normalized.alpha]
        scales = [float(value) for value in normalized.scales]
        metadata: dict[str, Any] = {
            "object_name": record.name,
            "object_relative_dir": record.relative_dir,
            "split": self.split,
            "epoch": effective_epoch,
            "selected_images": list(selected_names),
            "selected_image_sha256": list(selected_digests),
            "selected_image_digests": list(selected_digests),
            "normal_file": record.normal_file,
            "normal_sha256": sources["normal_digest"],
            "normal_digest": sources["normal_digest"],
            "mask_file": record.mask_file,
            "mask_sha256": sources["mask_digest"],
            "mask_digest": sources["mask_digest"],
            "mask_policy": self.config.mask_policy,
            "mask_source": record.mask_file,
            "target_mask_source": "normal_validity_mask",
            "model_mask_source": self.config.external_mask_filename,
            "source_geometry": dict(source_geometry),
            "roi": roi_values,
            "resized_geometry": resized_geometry,
            "mask_counts": mask_counts,
            "source_model_mask_count": mask_counts["source_model"],
            "model_mask_count": mask_counts["model"],
            "source_target_mask_count": mask_counts["source_target"],
            "target_mask_count": mask_counts["target"],
            "gt_valid_pixels": int(record.gt_valid_pixels),
            "mask_valid_pixels": int(record.mask_valid_pixels),
            "mask_only_pixels": int(record.mask_only_pixels),
            "normalization": {
                "alpha": alpha,
                "scales": scales,
            },
            "normalization_alpha": alpha,
            "normalization_scales": scales,
            "normalization_seed": int(normalized.seed),
            "seed": int(self.config.seed),
            "preprocessing_version": self.config.preprocessing_version,
            "manifest_version": int(self.manifest.version),
        }

        target_normal = geometry.target_normal
        target_mask = geometry.target_mask
        source_target_normal = geometry.source_target_normal
        source_target_mask = geometry.source_target_mask
        if (
            target_normal is None
            or target_mask is None
            or source_target_normal is None
            or source_target_mask is None
        ):
            raise RuntimeError("private EXR geometry did not produce complete target fields")
        sample: dict[str, Any] = {
            "imgs": torch.from_numpy(
                np.ascontiguousarray(normalized.images.transpose(2, 0, 1, 3), dtype=np.float32)
            ),
            "model_mask": torch.from_numpy(
                np.ascontiguousarray(geometry.model_mask[None], dtype=np.float32)
            ),
            "target_normal": torch.from_numpy(
                np.ascontiguousarray(target_normal.transpose(2, 0, 1), dtype=np.float32)
            ),
            "target_mask": torch.from_numpy(
                np.ascontiguousarray(target_mask[None], dtype=np.float32)
            ),
            "source_target_normal": torch.from_numpy(
                np.ascontiguousarray(source_target_normal.transpose(2, 0, 1), dtype=np.float32)
            ),
            "source_target_mask": torch.from_numpy(
                np.ascontiguousarray(source_target_mask[None], dtype=np.float32)
            ),
            "source_model_mask": torch.from_numpy(
                np.ascontiguousarray(geometry.source_model_mask[None], dtype=np.float32)
            ),
            "roi": torch.tensor(roi_values, dtype=torch.int64),
            "metadata": metadata,
        }
        if set(sample) != EXPECTED_FIELDS:  # pragma: no cover - defensive contract guard
            raise RuntimeError("internal private EXR sample contract violation")
        return sample


def collate_private_exr(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a nonempty batch while preserving metadata in sample order."""

    if not samples:
        raise ValueError("private EXR batch must be nonempty")
    if any(not isinstance(sample, Mapping) or set(sample) != EXPECTED_FIELDS for sample in samples):
        raise ValueError("private EXR sample fields do not match the training contract")
    batch: dict[str, Any] = {
        name: torch.stack([sample[name] for sample in samples], dim=0)
        for name in _TENSOR_FIELDS
    }
    batch["metadata"] = [sample["metadata"] for sample in samples]
    return batch


__all__ = ["EXPECTED_FIELDS", "PrivateExrTrainDataset", "collate_private_exr"]
