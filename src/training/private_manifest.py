"""Immutable split manifests and complete source preflight for private LINO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from src.comparison.exr_io import (
    read_file_bytes,
    read_mask_exr_bytes,
    read_rgb_exr_bytes,
)
from src.comparison.metrics import normal_validity_mask
from src.comparison.normal_contract import decode_ground_truth_normal
from src.training.config import PrivateTrainConfig


@dataclass(frozen=True)
class PrivateObjectRecord:
    """Immutable preflight metadata for one private-training object."""

    name: str
    relative_dir: str
    height: int
    width: int
    observation_files: tuple[str, ...]
    observation_sha256: tuple[str, ...]
    normal_file: str
    normal_sha256: str
    mask_file: str
    mask_sha256: str
    gt_valid_pixels: int
    mask_valid_pixels: int
    mask_only_pixels: int


@dataclass(frozen=True)
class PrivateSplitManifest:
    """Immutable, complete manifest for one configured train/test split."""

    version: int
    split: str
    data_root: str
    objects: tuple[PrivateObjectRecord, ...]


def _safe_basename(value: Any) -> bool:
    """Return whether ``value`` is one portable direct filename component."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if "\x00" in value or "\\" in value:
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
    """Resolve ``path`` and require it to remain below ``root``."""

    try:
        resolved_root = root.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} escapes resolved root: {path}") from exc
    return resolved_path


def _reject_symlink(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")


def _split_root(config: PrivateTrainConfig, split: str) -> tuple[str, Path, str]:
    if not isinstance(config, PrivateTrainConfig):
        raise TypeError("config must be a PrivateTrainConfig")
    if split not in {"train", "test"}:
        raise ValueError("split must be one of: train, test")
    configured_root = config.train_dir if split == "train" else config.test_dir
    root = Path(configured_root)
    try:
        resolved_root = root.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"failed to resolve {split} data root: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"{split} data root does not exist or is not a directory: {root}")
    return split, resolved_root, str(root)


def _object_directories(root: Path, config: PrivateTrainConfig) -> tuple[Path, ...]:
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ValueError(f"failed to inspect data root: {root}") from exc

    objects: list[Path] = []
    for entry in entries:
        if not entry.name.endswith(config.object_suffix):
            continue
        if not _safe_basename(entry.name):
            raise ValueError(f"object name is not a safe basename: {entry.name!r}")
        if entry.is_symlink():
            raise ValueError(f"object {entry.name} must not be a symlink: {entry}")
        _resolve_within(entry, root, label=f"object {entry.name}")
        if not entry.is_dir():
            raise ValueError(f"object {entry.name} must be a directory: {entry}")
        objects.append(entry)

    objects.sort(key=lambda path: path.name)
    names = [path.name for path in objects]
    if len(set(names)) != len(names):
        raise ValueError("object names must be unique")
    if not objects:
        raise ValueError(f"no object directories ending with {config.object_suffix!r}: {root}")
    return tuple(objects)


def _source_path(object_dir: Path, filename: str, *, label: str) -> Path:
    if not _safe_basename(filename):
        raise ValueError(f"{label} must be a safe basename: {filename!r}")
    candidate = object_dir / filename
    _reject_symlink(candidate, label=label)
    _resolve_within(candidate, object_dir, label=label)
    return candidate


def _observation_paths(object_dir: Path, config: PrivateTrainConfig) -> tuple[Path, ...]:
    try:
        entries = tuple(object_dir.iterdir())
    except OSError as exc:
        raise ValueError(f"failed to inspect object directory: {object_dir}") from exc

    observations: list[Path] = []
    for entry in entries:
        if not (
            entry.name.startswith(config.image_prefix)
            and entry.name.endswith(config.image_extension)
        ):
            continue
        if not _safe_basename(entry.name):
            raise ValueError(f"observation filename is not a safe basename: {entry.name!r}")
        if entry.is_symlink():
            raise ValueError(f"observation {entry.name} must not be a symlink: {entry}")
        _resolve_within(entry, object_dir, label=f"observation {entry.name}")
        if not entry.is_file():
            raise ValueError(f"observation {entry.name} must be a regular file: {entry}")
        observations.append(entry)

    observations.sort(key=lambda path: path.name)
    names = [path.name for path in observations]
    if len(set(names)) != len(names):
        raise ValueError(f"observation filenames must be unique for {object_dir.name}")
    return tuple(observations)


def _read_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    """Hash and decode callers' exact immutable byte snapshot."""

    payload = read_file_bytes(path, label=label)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} did not produce immutable file bytes: {path}")
    immutable = bytes(payload)
    return immutable, hashlib.sha256(immutable).hexdigest()


def _require_regular_file(path: Path, *, label: str, object_name: str) -> None:
    _resolve_within(path, path.parent, label=label)
    _reject_symlink(path, label=label)
    if not path.is_file():
        raise ValueError(f"{label} is missing or not a regular file for {object_name}: {path}")


def _normal_path(object_dir: Path, config: PrivateTrainConfig, *, object_name: str) -> Path:
    for filename in config.normal_filenames:
        candidate = _source_path(object_dir, filename, label=f"ground truth for {object_name}")
        if candidate.is_symlink():
            raise ValueError(f"ground truth for {object_name} must not be a symlink: {candidate}")
        if candidate.exists():
            _require_regular_file(candidate, label=f"ground truth for {object_name}", object_name=object_name)
            return candidate
    names = ", ".join(config.normal_filenames)
    raise ValueError(f"object {object_name} is missing a ground-truth normal file (tried: {names})")


def _required_path(object_dir: Path, filename: str, *, label: str, object_name: str) -> Path:
    candidate = _source_path(object_dir, filename, label=label)
    _reject_symlink(candidate, label=label)
    if not candidate.exists():
        raise ValueError(f"{label} is missing for {object_name}: {candidate}")
    _require_regular_file(candidate, label=label, object_name=object_name)
    return candidate


def _geometry_error(label: str, object_name: str, actual: tuple[int, ...], expected: tuple[int, int]) -> ValueError:
    return ValueError(
        f"{label} geometry mismatch for {object_name}: {actual[:2]}, expected {expected}"
    )


def _build_object_record(
    object_dir: Path,
    root: Path,
    config: PrivateTrainConfig,
) -> PrivateObjectRecord:
    object_name = object_dir.name
    _reject_symlink(object_dir, label=f"object {object_name}")
    resolved_object = _resolve_within(object_dir, root, label=f"object {object_name}")
    if not resolved_object.is_dir():
        raise ValueError(f"object {object_name} must be a directory: {object_dir}")
    expected = tuple(config.expected_source_geometry)

    observations = _observation_paths(object_dir, config)
    minimum = int(config.max_image_num)
    if len(observations) < minimum:
        raise ValueError(
            f"object {object_name} has only {len(observations)} observation(s); "
            f"requires {minimum}: {object_dir}"
        )

    observation_files: list[str] = []
    observation_hashes: list[str] = []
    for observation_path in observations:
        label = f"observation {observation_path.name} for {object_name}"
        payload, digest = _read_snapshot(observation_path, label=label)
        observation = read_rgb_exr_bytes(payload, label=label)
        shape = tuple(int(value) for value in observation.shape[:2])
        if shape != expected:
            raise _geometry_error(label, object_name, shape, expected)
        observation_files.append(observation_path.name)
        observation_hashes.append(digest)

    normal_path = _normal_path(object_dir, config, object_name=object_name)
    normal_label = f"ground truth for {object_name}"
    normal_payload, normal_digest = _read_snapshot(normal_path, label=normal_label)
    gt_encoded = read_rgb_exr_bytes(normal_payload, label=normal_label)
    gt = decode_ground_truth_normal(
        gt_encoded,
        config.normal_encoding,
        label=normal_label,
    )
    gt_shape = tuple(int(value) for value in gt.shape[:2])
    if gt_shape != expected:
        raise _geometry_error(normal_label, object_name, gt_shape, expected)
    gt_support = normal_validity_mask(gt)
    gt_valid_pixels = int(np.count_nonzero(gt_support))
    if gt_valid_pixels == 0:
        raise ValueError(f"empty GT-valid support for {object_name}")

    mask_path = _required_path(
        object_dir,
        config.external_mask_filename,
        label=f"external mask for {object_name}",
        object_name=object_name,
    )
    mask_label = f"external mask for {object_name}"
    mask_payload, mask_digest = _read_snapshot(mask_path, label=mask_label)
    external = read_mask_exr_bytes(mask_payload, label=mask_label) > 0
    mask_shape = tuple(int(value) for value in external.shape[:2])
    if mask_shape != expected:
        raise _geometry_error(mask_label, object_name, mask_shape, expected)
    mask_valid_pixels = int(np.count_nonzero(external))
    if mask_valid_pixels == 0:
        raise ValueError(f"empty external mask for {object_name}")

    outside = gt_support & ~external
    if np.any(outside):
        raise ValueError(
            f"{int(np.count_nonzero(outside))} GT-valid pixel(s) lie outside "
            f"the external mask for {object_name}"
        )
    mask_only_pixels = int(np.count_nonzero(external & ~gt_support))

    return PrivateObjectRecord(
        name=object_name,
        relative_dir=object_name,
        height=expected[0],
        width=expected[1],
        observation_files=tuple(observation_files),
        observation_sha256=tuple(observation_hashes),
        normal_file=normal_path.name,
        normal_sha256=normal_digest,
        mask_file=mask_path.name,
        mask_sha256=mask_digest,
        gt_valid_pixels=gt_valid_pixels,
        mask_valid_pixels=mask_valid_pixels,
        mask_only_pixels=mask_only_pixels,
    )


def build_private_split_manifest(
    config: PrivateTrainConfig,
    split: str,
) -> PrivateSplitManifest:
    """Preflight every source file and build one deterministic split manifest."""

    split_name, root, data_root = _split_root(config, split)
    object_dirs = _object_directories(root, config)
    records = tuple(_build_object_record(path, root, config) for path in object_dirs)
    names = [record.name for record in records]
    if len(set(names)) != len(names):
        raise ValueError("object names must be unique")
    return PrivateSplitManifest(
        version=1,
        split=split_name,
        data_root=data_root,
        objects=records,
    )


def private_manifest_bytes(manifest: PrivateSplitManifest) -> bytes:
    """Serialize one manifest using canonical compact JSON and a final newline."""

    if not isinstance(manifest, PrivateSplitManifest):
        raise TypeError("manifest must be a PrivateSplitManifest")
    return (
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def private_manifest_sha256(manifest: PrivateSplitManifest) -> str:
    """Return the digest of the exact canonical manifest bytes."""

    return hashlib.sha256(private_manifest_bytes(manifest)).hexdigest()


__all__ = [
    "PrivateObjectRecord",
    "PrivateSplitManifest",
    "build_private_split_manifest",
    "private_manifest_bytes",
    "private_manifest_sha256",
]
