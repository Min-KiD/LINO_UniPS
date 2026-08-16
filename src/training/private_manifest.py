"""Immutable split manifests and complete source preflight for private LINO."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from src.comparison.exr_io import read_mask_exr_bytes, read_rgb_exr_bytes
from src.comparison.metrics import normal_validity_mask
from src.comparison.normal_contract import decode_ground_truth_normal
from src.comparison.provenance import (
    _secure_directory_flags,
    assert_directory_path_identity,
    directory_identity,
    open_or_create_directory,
    read_regular_bytes_at_fd,
)
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


@dataclass(frozen=True)
class PrivateSourceIndexRecord:
    """Filename-only metadata for one lazily validated private object."""

    name: str
    relative_dir: str
    height: int
    width: int
    observation_files: tuple[str, ...]
    normal_file: str
    mask_file: str


@dataclass(frozen=True)
class PrivateSplitIndex:
    """Version-2 structural split index with no source-content claims."""

    version: int
    split: str
    data_root: str
    objects: tuple[PrivateSourceIndexRecord, ...]
    structural_index_version: str
    object_suffix: str
    image_prefix: str
    image_extension: str
    normal_encoding: str
    expected_source_geometry: tuple[int, int]
    mask_policy: str
    max_image_num: int
    light_selection: str
    seed: int


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


def _failure(split: str, object_name: str, filename: str, condition: str) -> ValueError:
    return ValueError(
        f"{condition} [split={split}, object={object_name}, file={filename}]"
    )


def _raise_failure(split: str, object_name: str, filename: str, condition: str) -> None:
    raise _failure(split, object_name, filename, condition)


def _contextualize(
    split: str,
    object_name: str,
    filename: str,
    exc: BaseException,
) -> ValueError:
    condition = str(exc).strip() or exc.__class__.__name__
    return _failure(split, object_name, filename, condition)


def _split_root(
    config: PrivateTrainConfig,
    split: str,
) -> tuple[str, Path, str, dict[str, int]]:
    if not isinstance(config, PrivateTrainConfig):
        raise TypeError("config must be a PrivateTrainConfig")
    if split not in {"train", "test"}:
        raise ValueError("split must be one of: train, test")
    configured_root = config.train_dir if split == "train" else config.test_dir
    root = Path(os.path.abspath(os.fspath(configured_root)))
    root_label = f"{split} data root directory"
    try:
        expected_root = directory_identity(root, label=root_label)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _contextualize(split, "<root>", "<root>", exc) from exc
    return split, root, str(configured_root), expected_root


def _fd_identity(descriptor: int, *, label: str) -> tuple[int, int]:
    try:
        info = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"failed to inspect {label} descriptor") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} descriptor is not a directory")
    return int(info.st_dev), int(info.st_ino)


def _identity_tuple(identity: dict[str, int]) -> tuple[int, int]:
    return int(identity["dev"]), int(identity["ino"])


def _assert_root_fd(
    descriptor: int,
    expected: tuple[int, int],
    *,
    split: str,
    object_name: str,
    filename: str,
) -> None:
    try:
        actual = _fd_identity(descriptor, label=f"{split} data root")
    except (OSError, RuntimeError, ValueError) as exc:
        raise _contextualize(split, object_name, filename, exc) from exc
    if actual != expected:
        _raise_failure(split, object_name, filename, "configured data root was replaced")


def _open_root(
    root: Path,
    expected: dict[str, int],
    *,
    split: str,
) -> tuple[int, tuple[int, int]]:
    descriptor: int | None = None
    root_expected = _identity_tuple(expected)
    try:
        descriptor, opened, _ = open_or_create_directory(root, label=f"{split} data root")
        actual = _identity_tuple(opened)
        if actual != root_expected:
            raise ValueError("configured data root was replaced while opening")
        _assert_root_fd(
            descriptor,
            root_expected,
            split=split,
            object_name="<root>",
            filename="<root>",
        )
        return descriptor, actual
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise _contextualize(split, "<root>", "<root>", exc) from exc


def _descriptor_names(
    descriptor: int,
    *,
    split: str,
    object_name: str,
    filename: str,
) -> tuple[str, ...]:
    try:
        names = tuple(os.listdir(descriptor))
    except (OSError, TypeError) as exc:
        raise _contextualize(split, object_name, filename, exc) from exc
    if any(not isinstance(name, str) for name in names):
        _raise_failure(split, object_name, filename, "directory listing contains a non-string name")
    return names


def _open_object(
    root_fd: int,
    root_expected: tuple[int, int],
    name: str,
    root: Path,
    *,
    split: str,
) -> tuple[int, tuple[int, int], Path]:
    object_path = root / name
    if not _safe_basename(name):
        _raise_failure(split, name, name, f"object name is not a safe basename: {name!r}")
    _assert_root_fd(
        root_fd,
        root_expected,
        split=split,
        object_name=name,
        filename="<object-directory>",
    )
    descriptor: int | None = None
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise _contextualize(split, name, "<object-directory>", exc) from exc
    except OSError as exc:
        raise _contextualize(split, name, "<object-directory>", exc) from exc
    if stat.S_ISLNK(info.st_mode):
        _raise_failure(split, name, "<object-directory>", "object directory must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        _raise_failure(split, name, "<object-directory>", "object entry must be a directory")
    expected = (int(info.st_dev), int(info.st_ino))
    try:
        flags = _secure_directory_flags()
        descriptor = os.open(name, flags, dir_fd=root_fd)
        actual = _fd_identity(descriptor, label=f"object {name}")
        if actual != expected:
            raise ValueError("object directory was replaced while opening")
        if actual != _identity_tuple(directory_identity(object_path, label=f"object {name}")):
            raise ValueError("object directory path identity changed")
        return descriptor, actual, object_path
    except (OSError, RuntimeError, ValueError) as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise _contextualize(split, name, "<object-directory>", exc) from exc


def _object_entries(
    root_fd: int,
    root: Path,
    root_expected: tuple[int, int],
    config: PrivateTrainConfig,
    *,
    split: str,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    names = _descriptor_names(
        root_fd,
        split=split,
        object_name="<root>",
        filename="<object-directory>",
    )
    entries: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()
    for name in names:
        if not name.endswith(config.object_suffix):
            continue
        if not _safe_basename(name):
            _raise_failure(split, name, name, f"object name is not a safe basename: {name!r}")
        if name in seen:
            _raise_failure(split, name, name, "object names must be unique")
        seen.add(name)
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as exc:
            raise _contextualize(split, name, "<object-directory>", exc) from exc
        if stat.S_ISLNK(info.st_mode):
            _raise_failure(split, name, "<object-directory>", "object directory must not be a symlink")
        if not stat.S_ISDIR(info.st_mode):
            _raise_failure(split, name, "<object-directory>", "object entry must be a directory")
        entries.append((name, (int(info.st_dev), int(info.st_ino))))
    if not entries:
        _raise_failure(
            split,
            "<root>",
            "<object-directory>",
            f"no object directories ending with {config.object_suffix!r}",
        )
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


def _source_entries(
    descriptor: int,
    *,
    split: str,
    object_name: str,
    config: PrivateTrainConfig,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    names = _descriptor_names(
        descriptor,
        split=split,
        object_name=object_name,
        filename="<observations>",
    )
    entries: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()
    for name in names:
        if not (name.startswith(config.image_prefix) and name.endswith(config.image_extension)):
            continue
        if not _safe_basename(name):
            _raise_failure(split, object_name, name, "observation filename is not a safe basename")
        if name in seen:
            _raise_failure(split, object_name, name, "observation filenames must be unique")
        seen.add(name)
        try:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise _contextualize(split, object_name, name, exc) from exc
        if stat.S_ISLNK(info.st_mode):
            _raise_failure(split, object_name, name, "observation must not be a symlink")
        if not stat.S_ISREG(info.st_mode):
            _raise_failure(split, object_name, name, "observation must be a regular file")
        entries.append((name, (int(info.st_dev), int(info.st_ino))))
    entries.sort(key=lambda item: item[0])
    return tuple(entries)


def _required_source(
    descriptor: int,
    filename: str,
    *,
    split: str,
    object_name: str,
    role: str,
) -> tuple[str, tuple[int, int]]:
    if not _safe_basename(filename):
        _raise_failure(split, object_name, filename, f"{role} filename is not a safe basename")
    try:
        info = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise _contextualize(split, object_name, filename, exc) from exc
    except OSError as exc:
        raise _contextualize(split, object_name, filename, exc) from exc
    if stat.S_ISLNK(info.st_mode):
        _raise_failure(split, object_name, filename, f"{role} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        _raise_failure(split, object_name, filename, f"{role} must be a regular file")
    return filename, (int(info.st_dev), int(info.st_ino))


def _snapshot(
    descriptor: int,
    filename: str,
    expected_file: tuple[int, int],
    expected_directory: tuple[int, int],
    object_path: Path,
    *,
    split: str,
    object_name: str,
) -> tuple[bytes, str]:
    try:
        before = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source must be a regular non-symlink file")
        before_identity = (int(before.st_dev), int(before.st_ino))
        if before_identity != expected_file:
            raise ValueError("source file identity changed before reading")
        payload = read_regular_bytes_at_fd(
            descriptor,
            filename,
            expected_directory_identity={"dev": expected_directory[0], "ino": expected_directory[1]},
            label=f"{split} object {object_name} file {filename}",
            directory_path=object_path,
        )
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("source reader did not return immutable bytes")
        after = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
        after_identity = (int(after.st_dev), int(after.st_ino))
        if after_identity != expected_file:
            raise ValueError("source file identity changed after reading")
        immutable = bytes(payload)
        return immutable, hashlib.sha256(immutable).hexdigest()
    except (OSError, RuntimeError, ValueError) as exc:
        raise _contextualize(split, object_name, filename, exc) from exc


class PrivateSourceSnapshot:
    """Read one manifest object's sources through pinned descriptors.

    The root and object descriptors remain open for the lifetime of the
    snapshot.  Every source read is descriptor-relative and revalidates both
    directory identities before and after consuming the immutable bytes.
    """

    def __init__(
        self,
        config: PrivateTrainConfig,
        split: str,
        record: PrivateObjectRecord | PrivateSourceIndexRecord,
    ) -> None:
        if not isinstance(config, PrivateTrainConfig):
            raise TypeError("config must be a PrivateTrainConfig")
        if split not in {"train", "test"}:
            raise ValueError("split must be one of: train, test")
        if not isinstance(record, (PrivateObjectRecord, PrivateSourceIndexRecord)):
            raise TypeError(
                "record must be a PrivateObjectRecord or PrivateSourceIndexRecord"
            )
        if not _safe_basename(record.name) or record.relative_dir != record.name:
            raise ValueError(f"manifest relative_dir is unsafe for {record.name}")

        self.config = config
        self.split = split
        self.record = record
        self._root_fd: int | None = None
        self._object_fd: int | None = None
        split_name, root, _, expected_root = _split_root(config, split)
        root_fd, root_identity = _open_root(root, expected_root, split=split_name)
        object_fd: int | None = None
        try:
            try:
                info = os.stat(record.name, dir_fd=root_fd, follow_symlinks=False)
            except OSError as exc:
                raise _contextualize(split, record.name, "<object-directory>", exc) from exc
            expected_object = (int(info.st_dev), int(info.st_ino))
            object_fd, object_identity, object_path = _open_object(
                root_fd,
                root_identity,
                record.name,
                root,
                split=split_name,
            )
            if object_identity != expected_object:
                _raise_failure(split, record.name, "<object-directory>", "object directory identity changed")
        except Exception:
            if object_fd is not None:
                try:
                    os.close(object_fd)
                except OSError:
                    pass
            try:
                os.close(root_fd)
            except OSError:
                pass
            raise
        self._root_fd = root_fd
        self._root_identity = root_identity
        self._root_path = root
        self._object_fd = object_fd
        self._object_identity = object_identity
        self._object_path = object_path
        self._closed = False

    def __enter__(self) -> "PrivateSourceSnapshot":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in (self._object_fd, self._root_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self._object_fd = None
        self._root_fd = None

    def read(self, filename: str, *, role: str) -> tuple[bytes, str]:
        if self._closed or self._root_fd is None or self._object_fd is None:
            raise ValueError("private source snapshot is closed")
        if not isinstance(role, str) or not role:
            raise ValueError("private source snapshot role must be a non-empty string")
        _assert_root_fd(
            self._root_fd,
            self._root_identity,
            split=self.split,
            object_name=self.record.name,
            filename=filename,
        )
        try:
            assert_directory_path_identity(
                self._root_path,
                {"dev": self._root_identity[0], "ino": self._root_identity[1]},
                label=f"{self.split} data root",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(self.split, self.record.name, filename, exc) from exc
        source_name, source_identity = _required_source(
            self._object_fd,
            filename,
            split=self.split,
            object_name=self.record.name,
            role=role,
        )
        payload, digest = _snapshot(
            self._object_fd,
            source_name,
            source_identity,
            self._object_identity,
            self._object_path,
            split=self.split,
            object_name=self.record.name,
        )
        _assert_root_fd(
            self._root_fd,
            self._root_identity,
            split=self.split,
            object_name=self.record.name,
            filename=filename,
        )
        try:
            assert_directory_path_identity(
                self._root_path,
                {"dev": self._root_identity[0], "ino": self._root_identity[1]},
                label=f"{self.split} data root",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(self.split, self.record.name, filename, exc) from exc
        return payload, digest


def open_private_source_snapshot(
    config: PrivateTrainConfig,
    split: str,
    record: PrivateObjectRecord | PrivateSourceIndexRecord,
) -> PrivateSourceSnapshot:
    """Open a descriptor-pinned source snapshot for one manifest record."""

    return PrivateSourceSnapshot(config, split, record)


def _build_source_index_record(
    root_fd: int,
    root_expected: tuple[int, int],
    object_name: str,
    object_expected: tuple[int, int],
    root: Path,
    config: PrivateTrainConfig,
    *,
    split: str,
) -> PrivateSourceIndexRecord:
    """Inspect names and entry types without opening any source file."""

    object_fd, object_identity, object_path = _open_object(
        root_fd,
        root_expected,
        object_name,
        root,
        split=split,
    )
    try:
        if object_identity != object_expected:
            _raise_failure(
                split,
                object_name,
                "<object-directory>",
                "object directory identity changed",
            )
        observations = _source_entries(
            object_fd,
            split=split,
            object_name=object_name,
            config=config,
        )
        minimum = int(config.max_image_num)
        if len(observations) < minimum:
            _raise_failure(
                split,
                object_name,
                "<observations>",
                f"only {len(observations)} observation(s); requires {minimum}",
            )
        normal_name, _ = _required_source(
            object_fd,
            config.normal_filenames[0],
            split=split,
            object_name=object_name,
            role="ground truth",
        )
        mask_name, _ = _required_source(
            object_fd,
            config.external_mask_filename,
            split=split,
            object_name=object_name,
            role="external mask",
        )
        final_sources = _source_entries(
            object_fd,
            split=split,
            object_name=object_name,
            config=config,
        )
        if final_sources != observations:
            _raise_failure(
                split,
                object_name,
                "<observations>",
                "observation allowlist changed while indexing",
            )
        try:
            assert_directory_path_identity(
                object_path,
                {"dev": object_identity[0], "ino": object_identity[1]},
                label=f"object {object_name}",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split, object_name, mask_name, exc) from exc
        height, width = config.expected_source_geometry
        return PrivateSourceIndexRecord(
            name=object_name,
            relative_dir=object_name,
            height=int(height),
            width=int(width),
            observation_files=tuple(name for name, _ in observations),
            normal_file=normal_name,
            mask_file=mask_name,
        )
    finally:
        try:
            os.close(object_fd)
        except OSError:
            pass


def _decoded_rgb(
    payload: bytes,
    *,
    split: str,
    object_name: str,
    filename: str,
    role: str,
) -> np.ndarray:
    label = f"{role} {filename} for {object_name} in {split}"
    try:
        decoded = read_rgb_exr_bytes(payload, label=label)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _contextualize(split, object_name, filename, exc) from exc
    shape = tuple(int(value) for value in decoded.shape)
    if len(shape) != 3 or shape[2] != 3:
        _raise_failure(split, object_name, filename, f"{role} EXR has invalid channel shape {shape}; expected H,W,3")
    return decoded


def _geometry_error(
    split: str,
    object_name: str,
    filename: str,
    role: str,
    actual: tuple[int, ...],
    expected: tuple[int, int],
) -> ValueError:
    return _failure(
        split,
        object_name,
        filename,
        f"{role} {filename} geometry mismatch: {actual[:2]}, expected {expected}",
    )


def _build_object_record(
    root_fd: int,
    root_expected: tuple[int, int],
    object_name: str,
    object_expected: tuple[int, int],
    root: Path,
    config: PrivateTrainConfig,
    *,
    split: str,
) -> PrivateObjectRecord:
    object_fd, object_identity, object_path = _open_object(
        root_fd,
        root_expected,
        object_name,
        root,
        split=split,
    )
    try:
        if object_identity != object_expected:
            _raise_failure(split, object_name, "<object-directory>", "object directory identity changed")
        observations = _source_entries(
            object_fd,
            split=split,
            object_name=object_name,
            config=config,
        )
        minimum = int(config.max_image_num)
        if len(observations) < minimum:
            _raise_failure(
                split,
                object_name,
                "<observations>",
                f"only {len(observations)} observation(s); requires {minimum}",
            )

        expected_geometry = tuple(config.expected_source_geometry)
        observation_files: list[str] = []
        observation_hashes: list[str] = []
        for filename, file_identity in observations:
            payload, digest = _snapshot(
                object_fd,
                filename,
                file_identity,
                object_identity,
                object_path,
                split=split,
                object_name=object_name,
            )
            observation = _decoded_rgb(
                payload,
                split=split,
                object_name=object_name,
                filename=filename,
                role="observation",
            )
            shape = tuple(int(value) for value in observation.shape)
            if shape[:2] != expected_geometry:
                raise _geometry_error(
                    split,
                    object_name,
                    filename,
                    "observation",
                    shape,
                    expected_geometry,
                )
            observation_files.append(filename)
            observation_hashes.append(digest)

        normal_filename = config.normal_filenames[0]
        normal_name, normal_identity = _required_source(
            object_fd,
            normal_filename,
            split=split,
            object_name=object_name,
            role="ground truth",
        )
        normal_payload, normal_digest = _snapshot(
            object_fd,
            normal_name,
            normal_identity,
            object_identity,
            object_path,
            split=split,
            object_name=object_name,
        )
        gt_encoded = _decoded_rgb(
            normal_payload,
            split=split,
            object_name=object_name,
            filename=normal_name,
            role="ground truth",
        )
        normal_label = f"ground truth {normal_name} for {object_name} in {split}"
        try:
            gt = decode_ground_truth_normal(
                gt_encoded,
                config.normal_encoding,
                label=normal_label,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split, object_name, normal_name, exc) from exc
        gt_shape = tuple(int(value) for value in gt.shape)
        if len(gt_shape) != 3 or gt_shape[2] != 3:
            _raise_failure(split, object_name, normal_name, f"ground truth has invalid channel shape {gt_shape}; expected H,W,3")
        if gt_shape[:2] != expected_geometry:
            raise _geometry_error(
                split,
                object_name,
                normal_name,
                "ground truth",
                gt_shape,
                expected_geometry,
            )
        gt_support = normal_validity_mask(gt)
        gt_valid_pixels = int(np.count_nonzero(gt_support))
        if gt_valid_pixels == 0:
            _raise_failure(split, object_name, normal_name, "empty GT-valid support")

        mask_name, mask_identity = _required_source(
            object_fd,
            config.external_mask_filename,
            split=split,
            object_name=object_name,
            role="external mask",
        )
        mask_payload, mask_digest = _snapshot(
            object_fd,
            mask_name,
            mask_identity,
            object_identity,
            object_path,
            split=split,
            object_name=object_name,
        )
        mask_label = f"external mask {mask_name} for {object_name} in {split}"
        try:
            external = read_mask_exr_bytes(mask_payload, label=mask_label)
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split, object_name, mask_name, exc) from exc
        mask_shape = tuple(int(value) for value in external.shape)
        if len(mask_shape) != 2:
            _raise_failure(split, object_name, mask_name, f"external mask has invalid channel shape {mask_shape}; expected H,W")
        if mask_shape != expected_geometry:
            raise _geometry_error(
                split,
                object_name,
                mask_name,
                "external mask",
                mask_shape,
                expected_geometry,
            )
        external = external > 0
        mask_valid_pixels = int(np.count_nonzero(external))
        if mask_valid_pixels == 0:
            _raise_failure(split, object_name, mask_name, "empty external mask")

        outside = gt_support & ~external
        if np.any(outside):
            _raise_failure(
                split,
                object_name,
                mask_name,
                f"{int(np.count_nonzero(outside))} GT-valid pixel(s) lie outside the external mask",
            )
        mask_only_pixels = int(np.count_nonzero(external & ~gt_support))

        try:
            assert_directory_path_identity(
                object_path,
                {"dev": object_identity[0], "ino": object_identity[1]},
                label=f"object {object_name}",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split, object_name, mask_name, exc) from exc

        return PrivateObjectRecord(
            name=object_name,
            relative_dir=object_name,
            height=expected_geometry[0],
            width=expected_geometry[1],
            observation_files=tuple(observation_files),
            observation_sha256=tuple(observation_hashes),
            normal_file=normal_name,
            normal_sha256=normal_digest,
            mask_file=mask_name,
            mask_sha256=mask_digest,
            gt_valid_pixels=gt_valid_pixels,
            mask_valid_pixels=mask_valid_pixels,
            mask_only_pixels=mask_only_pixels,
        )
    finally:
        try:
            os.close(object_fd)
        except OSError:
            pass


def build_private_split_manifest(
    config: PrivateTrainConfig,
    split: str,
) -> PrivateSplitManifest:
    """Preflight every source file and build one deterministic split manifest."""

    split_name, root, data_root, expected_root = _split_root(config, split)
    root_fd, root_identity = _open_root(root, expected_root, split=split_name)
    try:
        objects = _object_entries(
            root_fd,
            root,
            root_identity,
            config,
            split=split_name,
        )
        records = tuple(
            _build_object_record(
                root_fd,
                root_identity,
                object_name,
                object_identity,
                root,
                config,
                split=split_name,
            )
            for object_name, object_identity in objects
        )
        if len({record.name for record in records}) != len(records):
            _raise_failure(split_name, "<root>", "<object-directory>", "object names must be unique")
        _assert_root_fd(
            root_fd,
            root_identity,
            split=split_name,
            object_name="<root>",
            filename="<object-directory>",
        )
        final_names = _descriptor_names(
            root_fd,
            split=split_name,
            object_name="<root>",
            filename="<object-directory>",
        )
        final_objects = tuple(sorted(name for name in final_names if name.endswith(config.object_suffix)))
        expected_objects = tuple(name for name, _ in objects)
        if len(set(final_objects)) != len(final_objects):
            _raise_failure(split_name, "<root>", "<object-directory>", "object names must be unique")
        if final_objects != expected_objects:
            _raise_failure(split_name, "<root>", "<object-directory>", "object directory allowlist changed")
        for object_name, object_expected in objects:
            try:
                info = os.stat(object_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError as exc:
                raise _contextualize(split_name, object_name, "<object-directory>", exc) from exc
            actual_object = (int(info.st_dev), int(info.st_ino))
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _raise_failure(
                    split_name,
                    object_name,
                    "<object-directory>",
                    "object directory changed type",
                )
            if actual_object != object_expected:
                _raise_failure(
                    split_name,
                    object_name,
                    "<object-directory>",
                    "object directory identity changed",
                )
        try:
            assert_directory_path_identity(
                root,
                {"dev": root_identity[0], "ino": root_identity[1]},
                label=f"{split_name} data root",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split_name, "<root>", "<root>", exc) from exc
        return PrivateSplitManifest(
            version=1,
            split=split_name,
            data_root=data_root,
            objects=records,
        )
    finally:
        try:
            os.close(root_fd)
        except OSError:
            pass


def build_private_split_index(
    config: PrivateTrainConfig,
    split: str,
) -> PrivateSplitIndex:
    """Build a deterministic filename-only index without reading EXR bytes."""

    split_name, root, data_root, expected_root = _split_root(config, split)
    root_fd, root_identity = _open_root(root, expected_root, split=split_name)
    try:
        objects = _object_entries(
            root_fd,
            root,
            root_identity,
            config,
            split=split_name,
        )
        records: list[PrivateSourceIndexRecord] = []
        total = len(objects)
        interval = config.source_validation.progress_every_objects
        for completed, (object_name, object_identity) in enumerate(objects, start=1):
            records.append(
                _build_source_index_record(
                    root_fd,
                    root_identity,
                    object_name,
                    object_identity,
                    root,
                    config,
                    split=split_name,
                )
            )
            if completed % interval == 0 or completed == total:
                print(f"Indexed {completed}/{total} {split_name} objects")

        _assert_root_fd(
            root_fd,
            root_identity,
            split=split_name,
            object_name="<root>",
            filename="<object-directory>",
        )
        final_names = _descriptor_names(
            root_fd,
            split=split_name,
            object_name="<root>",
            filename="<object-directory>",
        )
        final_objects = tuple(
            sorted(name for name in final_names if name.endswith(config.object_suffix))
        )
        expected_objects = tuple(name for name, _ in objects)
        if len(set(final_objects)) != len(final_objects):
            _raise_failure(
                split_name,
                "<root>",
                "<object-directory>",
                "object names must be unique",
            )
        if final_objects != expected_objects:
            _raise_failure(
                split_name,
                "<root>",
                "<object-directory>",
                "object directory allowlist changed",
            )
        for object_name, object_expected in objects:
            try:
                info = os.stat(object_name, dir_fd=root_fd, follow_symlinks=False)
            except OSError as exc:
                raise _contextualize(
                    split_name, object_name, "<object-directory>", exc
                ) from exc
            actual = (int(info.st_dev), int(info.st_ino))
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                _raise_failure(
                    split_name,
                    object_name,
                    "<object-directory>",
                    "object directory changed type",
                )
            if actual != object_expected:
                _raise_failure(
                    split_name,
                    object_name,
                    "<object-directory>",
                    "object directory identity changed",
                )
        try:
            assert_directory_path_identity(
                root,
                {"dev": root_identity[0], "ino": root_identity[1]},
                label=f"{split_name} data root",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise _contextualize(split_name, "<root>", "<root>", exc) from exc

        return PrivateSplitIndex(
            version=2,
            split=split_name,
            data_root=data_root,
            objects=tuple(records),
            structural_index_version=config.source_validation.structural_index_version,
            object_suffix=config.object_suffix,
            image_prefix=config.image_prefix,
            image_extension=config.image_extension,
            normal_encoding=config.normal_encoding,
            expected_source_geometry=tuple(config.expected_source_geometry),
            mask_policy=config.mask_policy,
            max_image_num=int(config.max_image_num),
            light_selection=config.light_selection,
            seed=int(config.seed),
        )
    finally:
        try:
            os.close(root_fd)
        except OSError:
            pass


def private_index_bytes(index: PrivateSplitIndex) -> bytes:
    """Serialize one structural index as canonical compact JSON."""

    if not isinstance(index, PrivateSplitIndex):
        raise TypeError("index must be a PrivateSplitIndex")
    return (
        json.dumps(
            asdict(index),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def private_index_sha256(index: PrivateSplitIndex) -> str:
    """Return the digest of the exact canonical structural-index bytes."""

    return hashlib.sha256(private_index_bytes(index)).hexdigest()


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
    "PrivateSourceSnapshot",
    "PrivateSourceIndexRecord",
    "PrivateSplitIndex",
    "PrivateObjectRecord",
    "PrivateSplitManifest",
    "build_private_split_manifest",
    "build_private_split_index",
    "open_private_source_snapshot",
    "private_manifest_bytes",
    "private_manifest_sha256",
    "private_index_bytes",
    "private_index_sha256",
]
