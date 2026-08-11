"""Canonical, deterministic object/light manifests for SDM comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np

from .config import SdmExrInferenceConfig
from .exr_io import (
    read_file_bytes,
    read_mask_exr,
    read_rgb_exr,
    read_signed_normal_exr,
    sha256_file,
)
from .provenance import (
    assert_directory_path_identity,
    atomic_replace_bytes_at_fd,
    read_regular_bytes_at_fd,
    sha256_bytes,
)


@dataclass(frozen=True)
class ObjectRecord:
    """Immutable rich metadata for one object directory."""

    name: str
    relative_dir: str
    height: int
    width: int
    selected_images: tuple[str, ...]
    image_sha256: tuple[str, ...]
    normal_file: str
    normal_sha256: str
    mask_file: str | None
    mask_sha256: str | None


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable complete manifest shared by LINO and SDM runners."""

    version: int
    data_root: str
    seed: int
    max_image_num: int
    objects: tuple[ObjectRecord, ...]


def stable_seed(base_seed: int, object_name: str, purpose: str) -> int:
    """Derive a deterministic object- and purpose-specific unsigned seed."""

    payload = f"{base_seed}\0{object_name}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _object_directories(config: SdmExrInferenceConfig) -> tuple[Path, ...]:
    root = Path(config.data_root)
    if not root.is_dir():
        raise ValueError(f"data_root does not exist or is not a directory: {root}")
    objects = tuple(
        sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.endswith(config.object_suffix)
            ),
            key=lambda path: path.name,
        )
    )
    if not objects:
        raise ValueError(f"no object directories ending with {config.object_suffix!r}: {root}")
    return objects


def _observation_paths(object_dir: Path, config: SdmExrInferenceConfig) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in object_dir.iterdir()
                if path.is_file()
                and path.name.startswith(config.image_prefix)
                and path.name.endswith(config.image_extension)
            ),
            key=lambda path: path.name,
        )
    )


def _selection_manifest(
    config: SdmExrInferenceConfig,
    selection_manifest_bytes: bytes | None = None,
) -> Mapping[str, Any]:
    source = Path(config.effective_selection_manifest_path)
    try:
        if selection_manifest_bytes is None:
            with source.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        else:
            raw = json.loads(selection_manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read selection manifest: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"selection manifest must be a JSON object: {source}")
    return raw


def _is_basename(value: str) -> bool:
    if not isinstance(value, str) or "\\" in value:
        # A backslash is a path separator on Windows even when this process is
        # running on POSIX.  Rejecting it everywhere keeps canonical manifests
        # portable and prevents a literal backslash from smuggling a component.
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    # PureWindowsPath catches drive/UNC absolute paths even when running on
    # POSIX.  Requiring an exact basename also rejects traversal and nested
    # paths, which cannot be represented by the SDM selection file.
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and path.name == value
        and value not in {"", ".", ".."}
    )


def _manifest_selections(
    config: SdmExrInferenceConfig,
    object_dirs: tuple[Path, ...],
    observations: Mapping[str, tuple[Path, ...]],
    selection_manifest_bytes: bytes | None = None,
) -> dict[str, tuple[str, ...]]:
    raw = _selection_manifest(config, selection_manifest_bytes)
    expected = {path.name for path in object_dirs}
    actual = set(raw)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"selection manifest contains unknown object(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"selection manifest is missing object(s): {', '.join(missing)}")

    selections: dict[str, tuple[str, ...]] = {}
    for object_dir in object_dirs:
        object_name = object_dir.name
        values = raw[object_name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"selection for {object_dir} must be a nonempty filename list")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"selection for {object_dir} must contain nonempty filenames")
        if any(not _is_basename(value) for value in values):
            raise ValueError(f"selection for {object_dir} contains an out-of-object path")
        if len(set(values)) != len(values):
            raise ValueError(f"selection for {object_dir} contains duplicate filenames")
        available = {path.name for path in observations[object_name]}
        unknown_files = [value for value in values if value not in available]
        if unknown_files:
            raise ValueError(
                f"selection for {object_dir} contains missing file(s): {', '.join(unknown_files)}"
            )
        # Deliberately retain the JSON list order; SDM consumes that canonical
        # order and seeded mode is the only mode that permutes observations.
        selections[object_name] = tuple(values)
    return selections


def _select_images(
    config: SdmExrInferenceConfig,
    object_dir: Path,
    observations: tuple[Path, ...],
    manifest_selections: Mapping[str, tuple[str, ...]] | None,
) -> tuple[Path, ...]:
    if len(observations) < config.max_image_num:
        raise ValueError(
            f"object {object_dir} has only {len(observations)} image(s); "
            f"requires {config.max_image_num}: {object_dir}"
        )
    if manifest_selections is not None:
        by_name = {path.name: path for path in observations}
        return tuple(by_name[name] for name in manifest_selections[object_dir.name])
    permutation = np.random.default_rng(
        stable_seed(config.seed, object_dir.name, "light_selection")
    ).permutation(len(observations))
    return tuple(observations[int(index)] for index in permutation[: config.max_image_num])


def _normal_path(object_dir: Path, config: SdmExrInferenceConfig) -> Path:
    for filename in config.normal_filenames:
        if not _is_basename(filename):
            raise ValueError(f"normal filename must be an object-relative basename: {object_dir / filename}")
        candidate = object_dir / filename
        if candidate.is_file():
            return candidate
    names = ", ".join(config.normal_filenames)
    raise ValueError(f"object {object_dir} is missing a normal file (tried: {names})")


def build_dataset_manifest(
    config: SdmExrInferenceConfig,
    *,
    selection_manifest_bytes: bytes | None = None,
) -> DatasetManifest:
    """Validate the dataset and derive a stable ordered-light manifest."""

    object_dirs = _object_directories(config)
    observations = {
        object_dir.name: _observation_paths(object_dir, config) for object_dir in object_dirs
    }
    manifest_selections: Mapping[str, tuple[str, ...]] | None = None
    if config.light_selection == "manifest":
        manifest_selections = _manifest_selections(
            config,
            object_dirs,
            observations,
            selection_manifest_bytes,
        )

    records: list[ObjectRecord] = []
    data_root = Path(config.data_root)
    for object_dir in object_dirs:
        object_name = object_dir.name
        available = observations[object_name]
        selected = _select_images(config, object_dir, available, manifest_selections)
        if not selected:
            raise ValueError(f"object {object_dir} has no selected observations")

        first_image = read_rgb_exr(selected[0])
        height, width = first_image.shape[:2]
        image_hashes: list[str] = [sha256_file(selected[0])]
        for image_path in selected[1:]:
            image = read_rgb_exr(image_path)
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"observation dimensions are not aligned for {object_dir}: {image_path}"
                )
            image_hashes.append(sha256_file(image_path))

        normal_path = _normal_path(object_dir, config)
        normal = read_signed_normal_exr(normal_path)
        if normal.shape[:2] != (height, width):
            raise ValueError(
                f"normal dimensions are not aligned with observations for {object_dir}: {normal_path}"
            )

        mask_path: Path | None = None
        mask_hash: str | None = None
        if config.mask_policy == "external":
            if not _is_basename(config.external_mask_filename):
                raise ValueError(
                    "external_mask_filename must be an object-relative basename: "
                    f"{config.external_mask_filename}"
                )
            mask_path = object_dir / config.external_mask_filename
            if not mask_path.is_file():
                raise ValueError(f"external mask is missing: {mask_path}")
            mask = read_mask_exr(mask_path)
            if mask.shape != (height, width):
                raise ValueError(
                    f"external mask dimensions are not aligned for {object_dir}: {mask_path}"
                )
            if not np.any(mask > 0):
                raise ValueError(f"external mask is empty for {object_dir}: {mask_path}")
            mask_hash = sha256_file(mask_path)

        records.append(
            ObjectRecord(
                name=object_name,
                relative_dir=object_dir.relative_to(data_root).as_posix(),
                height=int(height),
                width=int(width),
                selected_images=tuple(path.name for path in selected),
                image_sha256=tuple(image_hashes),
                normal_file=normal_path.name,
                normal_sha256=sha256_file(normal_path),
                mask_file=mask_path.name if mask_path is not None else None,
                mask_sha256=mask_hash,
            )
        )

    return DatasetManifest(
        version=1,
        data_root=str(config.data_root),
        seed=int(config.seed),
        max_image_num=int(config.max_image_num),
        objects=tuple(records),
    )


def _manifest_dict(manifest: DatasetManifest) -> dict[str, Any]:
    return asdict(manifest)


def dataset_manifest_bytes(manifest: DatasetManifest) -> bytes:
    """Serialize the rich manifest exactly as persisted by comparison runners."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    return (
        json.dumps(
            _manifest_dict(manifest),
            indent=2,
            sort_keys=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sdm_selection_manifest_bytes(manifest: DatasetManifest) -> bytes:
    """Serialize SDM's ordered object-to-observation mapping."""

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    payload = {record.name: list(record.selected_images) for record in manifest.objects}
    return (
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _policy_child(path: str | Path, policy_path: Path, *, label: str) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(str(path)))
    if absolute.parent != policy_path:
        raise ValueError(f"{label} must be a direct child of policy output: {absolute}")
    name = absolute.name
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError(f"{label} must use one safe basename: {absolute}")
    return absolute, name


def persist_comparison_manifests_at_fd(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
    *,
    policy_fd: int,
    policy_identity: Mapping[str, int],
    policy_path: Path,
    selection_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Publish canonical manifests through one pinned policy directory."""

    source_selection = Path(config.effective_selection_manifest_path)
    canonical_selection = (
        policy_path / "selected_lights.json"
        if config.light_selection == "manifest"
        else source_selection
    )
    input_manifest, input_name = _policy_child(
        config.input_manifest_path,
        policy_path,
        label="rich input manifest",
    )
    canonical_manifest, canonical_name = _policy_child(
        canonical_selection,
        policy_path,
        label="effective selection manifest",
    )

    same_selection_path = (
        canonical_selection.resolve(strict=False) == source_selection.resolve(strict=False)
    )
    if config.light_selection == "manifest" and (
        source_selection.resolve(strict=False) == input_manifest.resolve(strict=False)
    ):
        raise ValueError("selection manifest cannot share the rich input manifest destination")
    if config.light_selection == "manifest" and selection_source_bytes is None:
        selection_source_bytes = read_file_bytes(
            source_selection,
            label="selection manifest",
        )
    if config.light_selection == "manifest" and same_selection_path:
        assert selection_source_bytes is not None
        current_selection = read_regular_bytes_at_fd(
            policy_fd,
            canonical_name,
            expected_directory_identity=policy_identity,
            directory_path=policy_path,
            label="selection manifest",
        )
        if current_selection != selection_source_bytes:
            raise ValueError("selection manifest changed after its immutable snapshot")

    rich_bytes = dataset_manifest_bytes(manifest)
    selection_bytes = sdm_selection_manifest_bytes(manifest)
    atomic_replace_bytes_at_fd(
        policy_fd,
        input_name,
        rich_bytes,
        expected_directory_identity=policy_identity,
        directory_path=policy_path,
        label="rich input manifest",
    )
    if config.light_selection != "manifest" or not same_selection_path:
        atomic_replace_bytes_at_fd(
            policy_fd,
            canonical_name,
            selection_bytes,
            expected_directory_identity=policy_identity,
            directory_path=policy_path,
            label="effective selection manifest",
        )

    assert_directory_path_identity(
        policy_path,
        policy_identity,
        label="policy output directory",
    )
    input_digest = sha256_bytes(rich_bytes)
    if config.light_selection == "manifest":
        assert selection_source_bytes is not None
        source_digest = sha256_bytes(selection_source_bytes)
    else:
        source_digest = sha256_bytes(selection_bytes)
    effective_digest = (
        source_digest if same_selection_path else sha256_bytes(selection_bytes)
    )
    input_result_path = input_manifest.resolve(strict=True)
    source_result_path = source_selection.resolve(strict=True)
    effective_result_path = canonical_manifest.resolve(strict=True)
    assert_directory_path_identity(
        policy_path,
        policy_identity,
        label="policy output directory",
    )
    return {
        "input_manifest_path": input_result_path,
        "selection_manifest_path": source_result_path,
        "effective_selection_manifest_path": effective_result_path,
        "input_manifest_sha256": input_digest,
        "selection_manifest_sha256": source_digest,
        "effective_selection_manifest_sha256": effective_digest,
    }


def _atomic_json_write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = path if isinstance(path, Path) else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    return destination


def save_dataset_manifest(
    manifest: DatasetManifest | SdmExrInferenceConfig,
    path: str | Path | DatasetManifest | SdmExrInferenceConfig,
) -> Path:
    """Atomically write the rich JSON manifest and return its destination.

    The canonical call is ``save_dataset_manifest(manifest, path)``.  For
    callers that already have the validated config, the equivalent
    ``save_dataset_manifest(config, manifest)`` and
    ``save_dataset_manifest(manifest, config)`` forms derive
    ``config.input_manifest_path`` without duplicating path policy logic.
    """

    if isinstance(manifest, DatasetManifest):
        record = manifest
        if isinstance(path, SdmExrInferenceConfig):
            destination: str | Path = path.input_manifest_path
        elif isinstance(path, (str, Path)):
            destination = path
        else:
            raise TypeError("path must be a path or SdmExrInferenceConfig")
    elif isinstance(path, DatasetManifest) and isinstance(manifest, SdmExrInferenceConfig):
        record = path
        destination = manifest.input_manifest_path
    else:
        raise TypeError("manifest must be a DatasetManifest")
    return _atomic_json_write(destination, _manifest_dict(record))


def _manifest_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"manifest field {field_name} must be positive")
    return value


def _manifest_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest field {field_name} must be a nonempty string")
    return value


def _manifest_string_list(
    value: Any,
    field_name: str,
    *,
    basenames: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"manifest field {field_name} must be a list of strings")
    if not value:
        raise ValueError(f"manifest field {field_name} must not be empty")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _manifest_text(item, f"{field_name}[{index}]")
        if basenames and not _is_basename(text):
            raise ValueError(
                f"manifest field {field_name}[{index}] must be an object-relative basename"
            )
        result.append(text)
    if len(set(result)) != len(result):
        raise ValueError(f"manifest field {field_name} must not contain duplicates")
    return tuple(result)


def _required(item: Mapping[str, Any], field_name: str) -> Any:
    try:
        return item[field_name]
    except KeyError as exc:
        raise ValueError(f"manifest object is missing required field {field_name}") from exc


def _load_object_record(item: Any) -> ObjectRecord:
    if not isinstance(item, Mapping):
        raise ValueError("manifest object records must be JSON objects")

    name = _manifest_text(_required(item, "name"), "name")
    relative_dir = _manifest_text(_required(item, "relative_dir"), "relative_dir")
    if not _is_basename(relative_dir) or relative_dir != name:
        raise ValueError(
            "manifest field relative_dir must be the canonical direct basename "
            "matching name"
        )
    height = _manifest_int(_required(item, "height"), "height", positive=True)
    width = _manifest_int(_required(item, "width"), "width", positive=True)
    selected_images = _manifest_string_list(
        _required(item, "selected_images"), "selected_images", basenames=True
    )
    image_hashes = _manifest_string_list(_required(item, "image_sha256"), "image_sha256")
    if len(selected_images) != len(image_hashes):
        raise ValueError("manifest fields selected_images and image_sha256 must have equal length")

    normal_file = _manifest_text(_required(item, "normal_file"), "normal_file")
    if not _is_basename(normal_file):
        raise ValueError("manifest field normal_file must be an object-relative basename")
    normal_hash = _manifest_text(_required(item, "normal_sha256"), "normal_sha256")

    mask_file_raw = _required(item, "mask_file")
    mask_hash_raw = _required(item, "mask_sha256")
    if mask_file_raw is None:
        if mask_hash_raw is not None:
            raise ValueError("manifest mask_file and mask_sha256 must be null together")
        mask_file = None
        mask_hash = None
    else:
        mask_file = _manifest_text(mask_file_raw, "mask_file")
        if not _is_basename(mask_file):
            raise ValueError("manifest field mask_file must be an object-relative basename")
        mask_hash = _manifest_text(mask_hash_raw, "mask_sha256")

    return ObjectRecord(
        name=name,
        relative_dir=relative_dir,
        height=height,
        width=width,
        selected_images=selected_images,
        image_sha256=image_hashes,
        normal_file=normal_file,
        normal_sha256=normal_hash,
        mask_file=mask_file,
        mask_sha256=mask_hash,
    )


def load_dataset_manifest(path: str | Path | SdmExrInferenceConfig) -> DatasetManifest:
    """Load a rich manifest, restoring immutable tuple fields."""

    if isinstance(path, SdmExrInferenceConfig):
        source = path.input_manifest_path
    else:
        source = path if isinstance(path, Path) else Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read dataset manifest: {source}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset manifest must be a JSON object: {source}")
    try:
        version = _manifest_int(raw["version"], "version", positive=True)
        data_root = _manifest_text(raw["data_root"], "data_root")
        seed = _manifest_int(raw["seed"], "seed")
        max_image_num = _manifest_int(raw["max_image_num"], "max_image_num", positive=True)
        objects_raw = raw["objects"]
        if not isinstance(objects_raw, list):
            raise ValueError("manifest field objects must be a list")
        if not objects_raw:
            raise ValueError("manifest field objects must not be empty")
        objects = [_load_object_record(item) for item in objects_raw]
        names = [record.name for record in objects]
        if len(set(names)) != len(names):
            raise ValueError("manifest object names must be unique")
        return DatasetManifest(
            version=version,
            data_root=data_root,
            seed=seed,
            max_image_num=max_image_num,
            objects=tuple(objects),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid dataset manifest: {source}: {exc}") from exc


def save_sdm_selection_manifest(
    manifest: DatasetManifest | SdmExrInferenceConfig,
    path: str | Path | DatasetManifest | SdmExrInferenceConfig,
) -> Path:
    """Atomically write SDM's object-name to ordered-basename mapping.

    As with :func:`save_dataset_manifest`, callers may pass a config first or
    second to derive ``effective_selection_manifest_path``.
    """

    if isinstance(manifest, DatasetManifest):
        record = manifest
        if isinstance(path, SdmExrInferenceConfig):
            destination: str | Path = path.effective_selection_manifest_path
        elif isinstance(path, (str, Path)):
            destination = path
        else:
            raise TypeError("path must be a path or SdmExrInferenceConfig")
    elif isinstance(path, DatasetManifest) and isinstance(manifest, SdmExrInferenceConfig):
        record = path
        destination = manifest.effective_selection_manifest_path
    else:
        raise TypeError("manifest must be a DatasetManifest")
    payload = {record.name: list(record.selected_images) for record in record.objects}
    return _atomic_json_write(destination, payload)


__all__ = [
    "ObjectRecord",
    "DatasetManifest",
    "stable_seed",
    "build_dataset_manifest",
    "dataset_manifest_bytes",
    "save_dataset_manifest",
    "load_dataset_manifest",
    "persist_comparison_manifests_at_fd",
    "save_sdm_selection_manifest",
    "sdm_selection_manifest_bytes",
]
