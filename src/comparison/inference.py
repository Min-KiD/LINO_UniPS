"""CPU-testable released-checkpoint inference and provenance for LINO.

The comparison runner intentionally keeps model construction behind a local
import and accepts injected model/dataset factories.  This makes configuration
and contract tests independent of Lightning, large checkpoints, and CUDA while
leaving the released model implementation untouched.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import SdmExrInferenceConfig
from .exr_io import (
    encode_normal_exr,
    encode_normal_png,
    read_file_bytes,
    read_signed_normal_exr_bytes,
)
from .manifest import (
    DatasetManifest,
    build_dataset_manifest,
    persist_comparison_manifests_at_fd,
    stable_seed,
)
from .provenance import (
    atomic_replace_bytes_at_fd,
    canonical_json_bytes,
    config_runtime_fingerprint,
    file_identity,
    lino_preprocessing_snapshot,
    sha256_bytes,
)
from .metrics import angular_metrics, load_source_gt, normal_validity_mask
from .reporting import (
    estimate_eta_seconds,
    format_clock_duration,
    format_optional_gib,
    should_report_progress,
)
from .transfer_gate import (
    constant_front_facing_mae,
    coordinate_transform_maes,
    preflight_transfer_sources,
    summarize_transfer_metrics,
)


ModelLoader = Callable[[SdmExrInferenceConfig, torch.device], Any]
DatasetFactory = Callable[[SdmExrInferenceConfig, DatasetManifest], Any]


_PRIVATE_SOURCE_REVISION = "lino-private-exr-training-v1"


def _precision_dtype(config: SdmExrInferenceConfig) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config.precision]


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _geometry_pair(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a [height, width] pair")
    if any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{label} must be a positive [height, width] pair")
    return int(value[0]), int(value[1])


def _load_checkpoint_state_dict(
    checkpoint_bytes: bytes,
    *,
    raw_only: bool,
) -> Mapping[str, torch.Tensor]:
    """Load a checkpoint payload, optionally enforcing the private raw boundary."""

    try:
        payload = torch.load(
            io.BytesIO(checkpoint_bytes),
            weights_only=False,
            map_location="cpu",
        )
    except Exception as exc:
        raise ValueError("LINO checkpoint cannot be parsed for architecture validation") from exc
    state_dict: Any = payload
    if raw_only:
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError(
                "strict trained LINO inference requires a non-empty raw tensor-only .pth state dict"
            )
        if any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in payload.items()
        ):
            raise ValueError(
                "strict trained LINO inference requires a raw tensor-only .pth state dict; "
                "checkpoint wrappers and artifact payloads are not accepted"
            )
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    state_dict = _mapping(state_dict, label="LINO checkpoint state_dict")
    if not state_dict:
        raise ValueError("LINO checkpoint state_dict must be non-empty")
    if raw_only and any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("LINO checkpoint state_dict must contain named tensors")
    return state_dict


def _checkpoint_schema_fingerprint(checkpoint_bytes: bytes) -> str:
    """Fingerprint a raw model-only checkpoint without constructing the model."""

    state_dict = _load_checkpoint_state_dict(checkpoint_bytes, raw_only=True)
    entries: list[list[Any]] = []
    for name, value in state_dict.items():
        entries.append([name, [int(dimension) for dimension in value.shape], str(value.dtype)])
    return sha256_bytes(canonical_json_bytes(entries))


def _validate_trained_export_sidecar(
    config: SdmExrInferenceConfig,
    checkpoint: Path,
    checkpoint_bytes: bytes,
    checkpoint_digest: str,
    *,
    selection_source_bytes: bytes,
) -> dict[str, Any]:
    """Validate the immutable sidecar paired with a private trained export.

    The sidecar is intentionally checked before the model constructor is
    reached.  This makes a wrong export/data contract fail without importing
    or allocating the released model.
    """

    if checkpoint.suffix != ".pth":
        raise ValueError(
            "strict trained LINO inference requires a regular .pth model-only checkpoint"
        )
    sidecar = checkpoint.with_suffix(".json")
    try:
        sidecar_identity_before = file_identity(sidecar, label="LINO checkpoint sidecar")
        sidecar_bytes = read_file_bytes(sidecar, label="LINO checkpoint sidecar")
        sidecar_identity = file_identity(sidecar, label="LINO checkpoint sidecar")
    except (OSError, ValueError) as exc:
        raise ValueError(
            "trained LINO checkpoint requires an adjacent regular JSON sidecar: "
            f"{sidecar}"
        ) from exc
    if sidecar_identity != sidecar_identity_before:
        raise ValueError("LINO checkpoint sidecar was replaced while reading its immutable snapshot")
    if not sidecar_bytes:
        raise ValueError("LINO checkpoint sidecar is empty")
    sidecar_digest = sha256_bytes(sidecar_bytes)
    try:
        metadata = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("LINO checkpoint sidecar is not valid JSON") from exc
    metadata = _mapping(metadata, label="LINO checkpoint sidecar")

    if metadata.get("artifact_kind") != "lino_private_inference_weights":
        raise ValueError("LINO checkpoint sidecar artifact_kind is invalid")
    if metadata.get("checkpoint_sha256") != checkpoint_digest:
        raise ValueError("LINO checkpoint sidecar checkpoint SHA-256 does not match checkpoint bytes")
    architecture = _nonempty_text(
        metadata.get("architecture_schema_sha256"),
        label="LINO checkpoint sidecar architecture_schema_sha256",
    )
    if architecture != _checkpoint_schema_fingerprint(checkpoint_bytes):
        raise ValueError("LINO checkpoint architecture schema fingerprint does not match sidecar")
    preprocessing = metadata.get("preprocessing_version")
    if preprocessing != config.preprocessing_version:
        raise ValueError(
            "LINO checkpoint sidecar preprocessing_version does not match inference config"
        )
    run_kind = metadata.get("run_kind")
    comparable = metadata.get("comparable")
    accepted_smoke = (
        config.allow_non_comparable_checkpoint
        and run_kind == "smoke"
        and comparable is False
    )
    if not (run_kind == "experiment" and comparable is True) and not accepted_smoke:
        raise ValueError(
            "trained LINO inference requires a comparable experiment export, or an "
            "explicitly opted-in non-comparable smoke export"
        )

    contract = _mapping(metadata.get("data_contract"), label="LINO sidecar data_contract")
    if contract.get("artifact_kind") != "lino_private_training_contract":
        raise ValueError("LINO sidecar data_contract artifact_kind is invalid")
    contract_run_kind = contract.get("run_kind")
    contract_comparable = contract.get("comparable")
    if accepted_smoke:
        if contract_run_kind != "smoke" or contract_comparable is not False:
            raise ValueError("LINO sidecar smoke data_contract is inconsistent")
    elif contract_run_kind != "experiment" or contract_comparable is not True:
        raise ValueError("LINO sidecar data_contract is not a comparable experiment")
    contract_architecture = _nonempty_text(
        contract.get("architecture_schema_sha256"),
        label="LINO sidecar data_contract architecture_schema_sha256",
    )
    if contract_architecture != architecture:
        raise ValueError("LINO sidecar architecture schema fingerprint mismatch")
    source_revision = _nonempty_text(
        metadata.get("source_revision"),
        label="LINO checkpoint sidecar source_revision",
    )
    contract_source_revision = _nonempty_text(
        contract.get("source_revision"),
        label="LINO sidecar data_contract source_revision",
    )
    if (
        source_revision != contract_source_revision
        or source_revision != _PRIVATE_SOURCE_REVISION
    ):
        raise ValueError(
            "LINO sidecar source_revision does not match the approved private training revision"
        )
    snapshot = _mapping(contract.get("config_snapshot"), label="LINO sidecar config_snapshot")

    expected_geometry = config.expected_source_geometry
    if expected_geometry is None:
        raise ValueError("trained LINO inference requires expected_source_geometry")
    if _geometry_pair(snapshot.get("expected_source_geometry"), label="LINO sidecar source geometry") != expected_geometry:
        raise ValueError("LINO sidecar source geometry does not match inference config")
    expected_snapshot = {
        "max_image_resolution": config.max_image_resolution,
        "canonical_resolution": 256,
        "mask_policy": "external",
        "normal_encoding": "unsigned",
        "external_mask_filename": config.external_mask_filename,
        "mask_margin": config.mask_margin,
        "pixel_samples": config.pixel_samples,
        "preprocessing_version": config.preprocessing_version,
        "object_suffix": config.object_suffix,
        "image_prefix": config.image_prefix,
        "image_extension": config.image_extension,
        "normal_filenames": list(config.normal_filenames),
        "seed": config.seed,
    }
    for field, expected in expected_snapshot.items():
        if snapshot.get(field) != expected:
            raise ValueError(f"LINO sidecar {field} does not match inference contract")
    if snapshot.get("max_image_num") != 6 or snapshot.get("light_selection") != "seeded":
        raise ValueError("LINO sidecar training light-selection contract is invalid")

    final_selection_digest = _nonempty_text(
        contract.get("final_selection_manifest_sha256"),
        label="LINO sidecar final_selection_manifest_sha256",
    )
    if final_selection_digest != sha256_bytes(selection_source_bytes):
        raise ValueError("LINO sidecar final selection manifest digest does not match manifest bytes")
    return {
        "path": str(sidecar.resolve(strict=True)),
        "identity": sidecar_identity,
        "sha256": sidecar_digest,
        "architecture_schema_sha256": architecture,
        "source_revision": source_revision,
        "run_kind": str(run_kind),
        "comparable": bool(comparable),
    }


def _validate_trained_selection_manifest(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
) -> None:
    """Require the final private comparison manifest's exact 16-light shape."""

    if config.light_selection != "manifest" or config.max_image_num != 16:
        raise ValueError("trained private inference requires manifest mode with exactly 16 lights")
    expected_names = {record.name for record in manifest.objects}
    if len(expected_names) != len(manifest.objects):
        raise ValueError("trained private inference manifest contains duplicate objects")
    for record in manifest.objects:
        names = tuple(record.selected_images)
        if len(names) != 16 or len(set(names)) != 16:
            raise ValueError(
                f"trained private inference requires exactly 16 unique selected images for {record.name}"
            )


def _resolve_device(config: SdmExrInferenceConfig) -> torch.device:
    """Resolve ``auto`` and reject unsupported device/precision combinations."""

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU inference supports only fp32 precision")
    # The released normal graph contains explicit bfloat16 conversions in its
    # decoder.  A true CUDA fp32/fp16 route would therefore be misleading
    # without architecture changes; keep the supported real path honest.
    if device.type == "cuda" and config.precision != "bf16":
        raise ValueError("released CUDA LINO inference requires bf16 precision")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported inference device: {device}")
    return device


def load_local_lino_checkpoint(
    config: SdmExrInferenceConfig,
    device: torch.device | str,
    *,
    checkpoint_bytes: bytes | None = None,
) -> Any:
    """Construct and strictly validate the released normal LINO checkpoint.

    The normal model import is deliberately function-local.  Importing
    ``src.models`` at module import time would pull optional Lightning and
    TorchMetrics dependencies into configuration-only workflows.
    """

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if resolved_device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU inference supports only fp32 precision")
    if resolved_device.type != "cuda":
        raise RuntimeError("released LINO checkpoint inference requires CUDA")
    if config.precision != "bf16":
        raise ValueError("released CUDA LINO inference requires bf16 precision")

    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LINO checkpoint does not exist: {checkpoint}")

    raw = checkpoint_bytes
    if raw is None:
        raw = read_file_bytes(checkpoint, label="LINO checkpoint")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("LINO checkpoint snapshot must be nonempty bytes")

    strict_route = bool(config.require_checkpoint_data_contract)
    if strict_route and checkpoint.suffix != ".pth":
        raise ValueError(
            "strict trained LINO inference requires a regular .pth model-only checkpoint"
        )
    state_dict = _load_checkpoint_state_dict(raw, raw_only=strict_route)

    # Keep this import local and import only the released normal architecture;
    # no PBR model, optimizer, trainer, or network-backed hub loader is used.
    # In the strict route the raw payload is parsed above, before construction
    # or device transfer, so an artifact checkpoint cannot allocate the model.
    from src.models.Net_module import LiNo_UniPS

    model = LiNo_UniPS(pixel_samples=config.pixel_samples, task_name="SDM_EXR")

    # Match the author's released loaders in ``hubconf.py`` and
    # ``LiNo_UniPS.from_pretrained`` for the default route.  The paired private
    # export route opts into strict loading only after its sidecar/data gate
    # has been validated by ``run_lino_inference``.
    model.load_state_dict(state_dict, strict=strict_route)

    # Preserve checkpoint parameter storage precision.  Autocast in the
    # inference loop controls operation precision without a permanent cast.
    model.to(resolved_device)
    model.eval()
    return model


def _default_dataset_factory(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
) -> Any:
    # Kept local to avoid importing data/model optional dependencies when this
    # module is imported only for CLI parsing or checkpoint helper tests.
    from src.data.sdm_exr_data import SdmExrDataset

    return SdmExrDataset(config, manifest)


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
            json.dump(payload, stream, indent=2, sort_keys=False, allow_nan=False)
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


def _secure_directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("secure LINO output requires O_DIRECTORY/O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _open_or_create_directory(path: Path, *, label: str) -> tuple[int, dict[str, int], Path]:
    """Walk/create a directory tree using no-follow descriptor-relative calls."""

    absolute = Path(os.path.abspath(str(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ValueError(f"{label} must resolve to an absolute path: {path}")
    flags = _secure_directory_flags()
    descriptor = os.open(os.sep, flags)
    try:
        for component in parts[1:]:
            try:
                info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    raise ValueError(f"{label} appeared during secure creation: {absolute}")
                info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"{label} must contain only real directories: {absolute}")
            expected = (int(info.st_dev), int(info.st_ino))
            child_fd = os.open(component, flags, dir_fd=descriptor)
            child_info = os.fstat(child_fd)
            actual = (int(child_info.st_dev), int(child_info.st_ino))
            if actual != expected:
                os.close(child_fd)
                raise ValueError(f"{label} was replaced during secure creation: {absolute}")
            os.close(descriptor)
            descriptor = child_fd
        info = os.fstat(descriptor)
        identity = {"dev": int(info.st_dev), "ino": int(info.st_ino)}
        return descriptor, identity, absolute
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_or_create_child_directory(
    parent_fd: int,
    parent_path: Path,
    name: str,
    *,
    label: str,
) -> tuple[int, dict[str, int], Path]:
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError(f"{label} name must be one safe path component")
    flags = _secure_directory_flags()
    child_fd: int | None = None
    try:
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} must be a real directory: {parent_path / name}")
        expected = (int(info.st_dev), int(info.st_ino))
        child_fd = os.open(name, flags, dir_fd=parent_fd)
        child_info = os.fstat(child_fd)
        actual = (int(child_info.st_dev), int(child_info.st_ino))
        if actual != expected:
            raise ValueError(f"{label} was replaced during secure creation: {parent_path / name}")
        result = child_fd, {"dev": actual[0], "ino": actual[1]}, parent_path / name
        child_fd = None
        return result
    except OSError as exc:
        raise ValueError(f"failed to securely open {label}: {parent_path / name}") from exc
    finally:
        if child_fd is not None:
            try:
                os.close(child_fd)
            except OSError:
                pass


def _write_bytes_at_fd(
    directory_fd: int,
    basename: str,
    payload: bytes,
    *,
    expected_directory_identity: dict[str, int],
    label: str,
    directory_path: Path | None = None,
) -> tuple[int, int]:
    """Atomically publish an owned artifact through a pinned directory FD."""

    return atomic_replace_bytes_at_fd(
        directory_fd,
        basename,
        payload,
        expected_directory_identity=expected_directory_identity,
        label=label,
        directory_path=directory_path,
    )


def _assert_directory_path_identity(
    path: Path,
    expected: dict[str, int],
    *,
    label: str,
) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} disappeared during publication: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} is not a real directory: {path}")
    actual = {"dev": int(info.st_dev), "ino": int(info.st_ino)}
    if actual != expected:
        raise ValueError(f"{label} was replaced during publication: {path}")


def _open_lino_output_tree(
    config: SdmExrInferenceConfig,
) -> tuple[int, dict[str, int], Path, int, dict[str, int], Path]:
    """Open the policy/LINO tree and close the parent if child opening fails."""

    policy_fd, policy_identity, policy_path = _open_or_create_directory(
        config.policy_root,
        label="LINO policy output directory",
    )
    try:
        lino_fd, lino_identity, lino_path = _open_or_create_child_directory(
            policy_fd,
            policy_path,
            "lino",
            label="LINO output directory",
        )
    except Exception:
        os.close(policy_fd)
        raise
    return (
        policy_fd,
        policy_identity,
        policy_path,
        lino_fd,
        lino_identity,
        lino_path,
    )


@contextlib.contextmanager
def _pinned_lino_output_tree(config: SdmExrInferenceConfig):
    tree = _open_lino_output_tree(config)
    policy_fd, _, _, lino_fd, _, _ = tree
    try:
        yield tree
    finally:
        try:
            os.close(lino_fd)
        finally:
            os.close(policy_fd)


def _validate_lino_output_tree(
    lino_fd: int,
    manifest: DatasetManifest,
    *,
    save_png: bool,
    require_all_objects: bool,
    require_run: bool,
    allow_run: bool = True,
    expected_run_identity: tuple[int, int] | None = None,
) -> None:
    """Validate exact root/object artifact sets through the pinned LINO FD."""

    if require_run and not allow_run:
        raise ValueError("require_run and allow_run=False are incompatible")
    if require_run and expected_run_identity is None:
        raise ValueError("required LINO run provenance identity is missing")
    if not require_run and expected_run_identity is not None:
        raise ValueError("unexpected LINO run provenance identity")
    expected_objects = {record.name for record in manifest.objects}
    expected_root = expected_objects | ({"run.json"} if require_run else set())
    actual_root = set(os.listdir(lino_fd))
    allowed_root = expected_objects | ({"run.json"} if allow_run else set())
    extras = actual_root - allowed_root
    if extras:
        raise ValueError("extra LINO prediction artifact(s): " + ", ".join(sorted(extras)))
    if require_all_objects:
        missing = expected_objects - actual_root
        if missing:
            raise ValueError(
                "missing LINO object output(s): " + ", ".join(sorted(missing))
            )
    if require_run and actual_root != expected_root:
        missing = expected_root - actual_root
        raise ValueError("missing LINO artifact(s): " + ", ".join(sorted(missing)))

    if "run.json" in actual_root:
        run_info = os.stat("run.json", dir_fd=lino_fd, follow_symlinks=False)
        if not stat.S_ISREG(run_info.st_mode):
            raise ValueError("LINO run provenance must be a regular non-symlink file")
        run_identity = (int(run_info.st_dev), int(run_info.st_ino))
        if expected_run_identity is not None and run_identity != expected_run_identity:
            raise ValueError("LINO run provenance was replaced after publication")

    directory_flags = _secure_directory_flags()
    expected_files = {"normal_pred.exr"}
    if save_png:
        expected_files.add("normal_pred.png")
    for object_name in sorted(expected_objects & actual_root):
        before = os.stat(object_name, dir_fd=lino_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"LINO object output must be a real directory: {object_name}")
        object_fd = os.open(object_name, directory_flags, dir_fd=lino_fd)
        try:
            opened = os.fstat(object_fd)
            if (int(opened.st_dev), int(opened.st_ino)) != (
                int(before.st_dev),
                int(before.st_ino),
            ):
                raise ValueError(f"LINO object output was replaced: {object_name}")
            actual_files = set(os.listdir(object_fd))
            extra_files = actual_files - expected_files
            if extra_files:
                raise ValueError(
                    f"extra LINO prediction artifact(s) for {object_name}: "
                    + ", ".join(sorted(extra_files))
                )
            if require_all_objects:
                missing_files = expected_files - actual_files
                if missing_files:
                    raise ValueError(
                        f"missing LINO prediction artifact(s) for {object_name}: "
                        + ", ".join(sorted(missing_files))
                    )
            for basename in sorted(expected_files & actual_files):
                info = os.stat(basename, dir_fd=object_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(
                        f"LINO prediction must be a regular non-symlink file: "
                        f"{object_name}/{basename}"
                    )
        finally:
            os.close(object_fd)


def _invalidate_lino_run(lino_fd: int) -> None:
    """Remove stale success provenance before any rerun artifact is mutated."""

    try:
        info = os.stat("run.json", dir_fd=lino_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("LINO run provenance must be a regular non-symlink file")
    os.unlink("run.json", dir_fd=lino_fd)
    os.fsync(lino_fd)


def _remove_published_lino_run(
    lino_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    """Roll back this invocation's success marker without following replacements."""

    try:
        info = os.stat("run.json", dir_fd=lino_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode):
        return
    if (int(info.st_dev), int(info.st_ino)) != expected_identity:
        return
    os.unlink("run.json", dir_fd=lino_fd)
    os.fsync(lino_fd)


def _json_safe(value: Any) -> Any:
    """Convert metadata/tensor scalar values into strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _repository_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _selection_paths(config: SdmExrInferenceConfig) -> tuple[Path, Path]:
    """Return input source and canonical persisted selection paths.

    ``SdmExrInferenceConfig.effective_selection_manifest_path`` intentionally
    points to a user-supplied manifest in manifest mode.  The runner therefore
    persists a separate canonical copy under the policy output and never
    replaces that input file.
    """

    source = Path(config.effective_selection_manifest_path)
    if config.light_selection == "manifest" and config.selection_manifest is not None:
        canonical = config.policy_root / "selected_lights.json"
    else:
        canonical = source
    return source, canonical


def _persist_manifests(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
    *,
    policy_fd: int | None = None,
    policy_identity: Mapping[str, int] | None = None,
    policy_path: Path | None = None,
    selection_source_bytes: bytes | None = None,
) -> tuple[Path, Path, str, str, str | None]:
    """Persist rich/effective manifests and return their exact digests."""

    owns_descriptor = policy_fd is None
    if owns_descriptor:
        if policy_identity is not None or policy_path is not None:
            raise ValueError("unpinned manifest publication received partial policy metadata")
        policy_fd, opened_identity, opened_path = _open_or_create_directory(
            config.policy_root,
            label="LINO policy output directory",
        )
        policy_identity = opened_identity
        policy_path = opened_path
    elif policy_identity is None or policy_path is None:
        raise ValueError("pinned manifest publication requires policy identity and path")
    assert policy_fd is not None
    assert policy_identity is not None
    assert policy_path is not None
    try:
        persisted = persist_comparison_manifests_at_fd(
            config,
            manifest,
            policy_fd=policy_fd,
            policy_identity=policy_identity,
            policy_path=policy_path,
            selection_source_bytes=selection_source_bytes,
        )
        return (
            Path(persisted["input_manifest_path"]),
            Path(persisted["effective_selection_manifest_path"]),
            str(persisted["input_manifest_sha256"]),
            str(persisted["selection_manifest_sha256"]),
            str(persisted["effective_selection_manifest_sha256"]),
        )
    finally:
        if owns_descriptor:
            try:
                os.close(policy_fd)
            except OSError:
                pass


def _seed_for_object(config: SdmExrInferenceConfig, object_name: str) -> int:
    # NumPy accepts unsigned 32-bit seeds while torch accepts signed 64-bit
    # seeds.  Derive both deterministically from the same stable object seed.
    return stable_seed(config.seed, object_name, "lino_inference")


def _seed_object_rngs(seed: int, *, cuda: bool) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))
    if cuda:
        torch.cuda.manual_seed_all(seed % (2**63 - 1))


def _collate_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise TypeError("dataset samples must be mappings")
    if "imgs" not in sample or "mask" not in sample:
        raise ValueError("dataset sample is missing imgs or mask")

    imgs = sample["imgs"]
    if not torch.is_tensor(imgs):
        raise TypeError("dataset imgs must be a torch.Tensor")
    if imgs.ndim == 4:
        from src.data.sdm_exr_data import collate_single_sdm_exr

        batch = collate_single_sdm_exr([dict(sample)])
    elif imgs.ndim == 5:
        batch = dict(sample)
    else:
        raise ValueError(f"dataset imgs must be [C,H,W,N] or [B,C,H,W,N], got {imgs.shape}")

    required = {"imgs", "mask", "mask_original", "roi", "metadata"}
    if set(batch) != required:
        raise ValueError(f"dataset batch fields must be {sorted(required)}")
    return batch


def _model_batch_on_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move only model image/mask tensors; preserve CPU ROI/mask metadata."""

    result = dict(batch)
    for field in ("imgs", "mask"):
        value = result[field]
        if not torch.is_tensor(value):
            raise TypeError(f"dataset field {field} must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise ValueError(f"dataset field {field} must remain float32")
        result[field] = value.to(device=device)
    for field in ("roi", "mask_original"):
        value = result[field]
        if not torch.is_tensor(value):
            raise TypeError(f"dataset field {field} must be a torch.Tensor")
        if value.device.type != "cpu":
            raise ValueError(f"dataset field {field} must remain on CPU")
        if field == "roi" and value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("dataset field roi must remain an integer CPU tensor")
        if field == "mask_original" and value.dtype != torch.float32:
            raise ValueError("dataset field mask_original must remain a float32 CPU tensor")
    return result


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _normalise_prediction(raw: Any, *, height: int, width: int) -> np.ndarray:
    if torch.is_tensor(raw):
        array = raw.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        try:
            array = np.asarray(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("LINO forward must return a numeric normal array") from exc
    if array.ndim != 3 or array.shape != (height, width, 3):
        raise ValueError(
            "LINO forward must return source-resolution [H0,W0,3] output: "
            f"got {array.shape}, expected {(height, width, 3)}"
        )
    try:
        array = np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("LINO forward must return a numeric normal array") from exc
    if not np.isfinite(array).all():
        raise ValueError("LINO forward returned non-finite normals")

    # Normalize each nonzero vector and preserve exact zero support.  This is
    # intentionally the only postprocessing: no model/input mask is reapplied.
    lengths = np.linalg.norm(array.astype(np.float64), axis=2, keepdims=True)
    normalized = np.zeros_like(array, dtype=np.float32)
    np.divide(array, lengths, out=normalized, where=lengths > 0)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized LINO normals are non-finite")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _run_lino_inference_pinned(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
    output_tree: tuple[int, dict[str, int], Path, int, dict[str, int], Path],
    *,
    model_loader: ModelLoader | None,
    dataset_factory: DatasetFactory | None,
    device: torch.device,
    dtype: torch.dtype,
    checkpoint: Path,
    checkpoint_bytes: bytes,
    checkpoint_digest: str,
    checkpoint_identity: Mapping[str, int],
    config_file_path: Path | None,
    config_digest: str | None,
    selection_source_bytes: bytes | None,
    trained_export: Mapping[str, Any] | None,
    clock: Callable[[], float],
    total_started: float,
) -> dict[str, Any]:
    (
        policy_fd,
        policy_identity,
        policy_path,
        lino_fd,
        lino_identity,
        lino_path,
    ) = output_tree
    _assert_directory_path_identity(
        policy_path,
        policy_identity,
        label="LINO policy output directory",
    )
    _assert_directory_path_identity(
        lino_path,
        lino_identity,
        label="LINO output directory",
    )
    _validate_lino_output_tree(
        lino_fd,
        manifest,
        save_png=config.save_png,
        require_all_objects=False,
        require_run=False,
    )
    _invalidate_lino_run(lino_fd)
    if config.require_checkpoint_data_contract:
        _validate_trained_selection_manifest(config, manifest)
    preflight_report = preflight_transfer_sources(config, manifest)
    if tuple(preflight_report) != tuple(record.name for record in manifest.objects):
        raise ValueError("preflight object order does not match the manifest")
    _, canonical_selection, input_digest, selection_digest, canonical_digest = _persist_manifests(
        config,
        manifest,
        policy_fd=policy_fd,
        policy_identity=policy_identity,
        policy_path=policy_path,
        selection_source_bytes=selection_source_bytes,
    )
    source_selection, _ = _selection_paths(config)

    factory = dataset_factory or _default_dataset_factory
    dataset = factory(config, manifest)
    if dataset is None:
        raise ValueError("dataset_factory returned None")
    try:
        object_count = len(dataset)
    except (TypeError, AttributeError) as exc:
        raise TypeError("dataset_factory result must be a sized sequence") from exc
    if object_count != len(manifest.objects):
        raise ValueError(
            f"dataset length {object_count} does not match manifest object count "
            f"{len(manifest.objects)}"
        )

    if model_loader is None:
        model = load_local_lino_checkpoint(
            config,
            device,
            checkpoint_bytes=checkpoint_bytes,
        )
    else:
        model = model_loader(config, device)
    if model is None:
        raise ValueError("model_loader returned None")
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = clock()
    last_object_finished = started
    object_records: list[dict[str, Any]] = []
    identity_maes: list[float] = []
    baseline_maes: list[float] = []
    coordinate_sweeps: list[dict[str, float]] = []

    for index, record in enumerate(manifest.objects):
        object_started = clock()
        seed = _seed_for_object(config, record.name)
        _seed_object_rngs(seed, cuda=device.type == "cuda")
        sample = dataset[index]
        batch = _collate_sample(sample)
        metadata = batch["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"dataset metadata must be a mapping for {record.name}")
        object_name = str(metadata.get("object_name", record.name))
        if object_name != record.name:
            raise ValueError(
                f"dataset object order/name mismatch: expected {record.name}, got {object_name}"
            )
        model_geometry = metadata.get("resized_geometry")
        if (
            not isinstance(model_geometry, Mapping)
            or set(model_geometry) != {"height", "width"}
            or type(model_geometry["height"]) is not int
            or type(model_geometry["width"]) is not int
            or model_geometry["height"] <= 0
            or model_geometry["width"] <= 0
        ):
            raise ValueError(f"model geometry is invalid for {record.name}")

        model_batch = _model_batch_on_device(batch, device)
        with torch.inference_mode():
            with _autocast_context(device, dtype):
                raw_prediction = model(model_batch)
        prediction = _normalise_prediction(
            raw_prediction,
            height=int(record.height),
            width=int(record.width),
        )

        _assert_directory_path_identity(
            policy_path,
            policy_identity,
            label="LINO policy output directory",
        )
        _assert_directory_path_identity(
            lino_path,
            lino_identity,
            label="LINO output directory",
        )
        object_fd: int | None = None
        try:
            object_fd, object_identity, object_path = _open_or_create_child_directory(
                lino_fd,
                lino_path,
                record.name,
                label=f"LINO object output for {record.name}",
            )
            existing_names = set(os.listdir(object_fd))
            allowed_names = {"normal_pred.exr"}
            if config.save_png:
                allowed_names.add("normal_pred.png")
            extras = existing_names - allowed_names
            if extras:
                raise ValueError(
                    f"extra LINO prediction artifact(s) for {record.name}: "
                    + ", ".join(sorted(extras))
                )
            exr_bytes = encode_normal_exr(prediction)
            scored_prediction = read_signed_normal_exr_bytes(
                exr_bytes,
                label=f"LINO prediction for {record.name}",
            )
            source_gt, _ = load_source_gt(config, record)
            support = normal_validity_mask(source_gt)
            object_metrics = angular_metrics(
                source_gt,
                scored_prediction,
                support,
            )
            identity_mae = float(object_metrics["mae"])
            baseline_mae = constant_front_facing_mae(source_gt, support)
            coordinate_sweep = coordinate_transform_maes(
                source_gt,
                scored_prediction,
                support,
            )
            object_coordinate_summary = summarize_transfer_metrics(
                [identity_mae],
                [baseline_mae],
                [coordinate_sweep],
            )
            identity_maes.append(identity_mae)
            baseline_maes.append(baseline_mae)
            coordinate_sweeps.append(coordinate_sweep)
            source_report = dict(preflight_report[record.name])
            transfer_diagnostics = {
                **source_report,
                "model_geometry": dict(model_geometry),
                "identity_mae": identity_mae,
                "constant_normal_mae": baseline_mae,
                "best_coordinate_mae": object_coordinate_summary[
                    "best_coordinate_macro_mae"
                ],
                "best_coordinate_transform": object_coordinate_summary[
                    "best_coordinate_transform"
                ],
            }
            _write_bytes_at_fd(
                object_fd,
                "normal_pred.exr",
                exr_bytes,
                expected_directory_identity=object_identity,
                directory_path=object_path,
                label=f"LINO prediction for {record.name}",
            )
            if config.save_png:
                png_bytes = encode_normal_png(prediction)
                _write_bytes_at_fd(
                    object_fd,
                    "normal_pred.png",
                    png_bytes,
                    expected_directory_identity=object_identity,
                    directory_path=object_path,
                    label=f"LINO preview for {record.name}",
                )
            _assert_directory_path_identity(
                object_path,
                object_identity,
                label=f"LINO object output for {record.name}",
            )
            _assert_directory_path_identity(
                lino_path,
                lino_identity,
                label="LINO output directory",
            )
            _assert_directory_path_identity(
                policy_path,
                policy_identity,
                label="LINO policy output directory",
            )
            exr_path = object_path / "normal_pred.exr"
            png_path = object_path / "normal_pred.png"
            output_digest = sha256_bytes(exr_bytes)
        finally:
            if object_fd is not None:
                os.close(object_fd)

        object_finished = clock()
        last_object_finished = object_finished
        object_records.append(
            {
                "object_name": record.name,
                "selected_images": list(record.selected_images),
                "source_geometry": {
                    "height": int(record.height),
                    "width": int(record.width),
                },
                "metadata": _json_safe(metadata),
                "seed": int(seed),
                "runtime_seconds": object_finished - object_started,
                "output_path": str(exr_path) if config.save_exr else None,
                "preview_path": str(png_path) if config.save_png else None,
                "output_sha256": output_digest,
                "mae": identity_mae,
                "valid_pixel_count": int(object_metrics["valid_pixel_count"]),
                "transfer_diagnostics": _json_safe(transfer_diagnostics),
            }
        )
        completed = index + 1
        elapsed = object_finished - started
        if should_report_progress(completed, object_count):
            eta = estimate_eta_seconds(elapsed, completed, object_count)
            print(
                f"LINO progress: {completed}/{object_count} | "
                f"elapsed {format_clock_duration(elapsed)} | "
                f"ETA {format_clock_duration(eta)}"
            )
        del model_batch, batch, sample, raw_prediction, prediction

    runtime_seconds = last_object_finished - started
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = None
        peak_reserved = None

    repo_root = Path(__file__).resolve().parents[2]
    transfer_summary = summarize_transfer_metrics(
        identity_maes,
        baseline_maes,
        coordinate_sweeps,
    )
    mean_mae = float(sum(identity_maes) / len(identity_maes))
    if not np.isclose(
        mean_mae,
        transfer_summary["identity_macro_mae"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("official MAE and transfer identity summary diverged")
    total_runtime_seconds = max(0.0, float(clock() - total_started))
    provenance: dict[str, Any] = {
        "model": "LINO-UniPS",
        "config_path": str(config_file_path.resolve(strict=True)) if config_file_path else None,
        "config_sha256": config_digest,
        "config_runtime_fingerprint": config_runtime_fingerprint(config),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_path": str(checkpoint.resolve(strict=True)),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_sidecar_path": (
            trained_export["path"] if trained_export is not None else None
        ),
        "checkpoint_sidecar_sha256": (
            trained_export["sha256"] if trained_export is not None else None
        ),
        "preprocessing": lino_preprocessing_snapshot(config),
        "preprocessing_version": config.preprocessing_version,
        "require_checkpoint_data_contract": config.require_checkpoint_data_contract,
        "allow_non_comparable_checkpoint": config.allow_non_comparable_checkpoint,
        "source_revision": (
            trained_export["source_revision"] if trained_export is not None else None
        ),
        "architecture_schema_sha256": (
            trained_export["architecture_schema_sha256"]
            if trained_export is not None
            else None
        ),
        "selected_light_count": int(config.max_image_num),
        "run_kind": (
            trained_export["run_kind"] if trained_export is not None else "released_transfer"
        ),
        "comparable": (
            bool(trained_export["comparable"]) if trained_export is not None else False
        ),
        "repository_commit": _repository_commit(repo_root),
        "mask_policy": config.mask_policy,
        "normal_encoding": config.normal_encoding,
        "input_manifest_sha256": input_digest,
        "selection_manifest_sha256": selection_digest,
        "effective_selection_manifest_sha256": canonical_digest,
        "selection_manifest_path": str(source_selection),
        "effective_selection_manifest_path": str(canonical_selection),
        "device": str(device),
        "precision": config.precision,
        "requested_precision": config.precision,
        "effective_precision": config.precision,
        "objects": object_records,
        "mean_mae": mean_mae,
        "mae_objects": len(object_records),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "transfer_summary": transfer_summary,
        "total_runtime_seconds": total_runtime_seconds,
    }
    serialized_provenance = (
        json.dumps(
            provenance,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _validate_lino_output_tree(
        lino_fd,
        manifest,
        save_png=config.save_png,
        require_all_objects=True,
        require_run=False,
        allow_run=False,
    )
    _assert_directory_path_identity(
        policy_path,
        policy_identity,
        label="LINO policy output directory",
    )
    _assert_directory_path_identity(
        lino_path,
        lino_identity,
        label="LINO output directory",
    )
    run_identity = _write_bytes_at_fd(
        lino_fd,
        "run.json",
        serialized_provenance,
        expected_directory_identity=lino_identity,
        directory_path=lino_path,
        label="LINO run provenance",
    )
    try:
        _validate_lino_output_tree(
            lino_fd,
            manifest,
            save_png=config.save_png,
            require_all_objects=True,
            require_run=True,
            expected_run_identity=run_identity,
        )
        _assert_directory_path_identity(
            lino_path,
            lino_identity,
            label="LINO output directory",
        )
        _assert_directory_path_identity(
            policy_path,
            policy_identity,
            label="LINO policy output directory",
        )
    except Exception:
        _remove_published_lino_run(lino_fd, run_identity)
        raise
    return provenance


def run_lino_inference(
    config: SdmExrInferenceConfig,
    *,
    model_loader: ModelLoader | None = None,
    dataset_factory: DatasetFactory | None = None,
    config_path: str | Path | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run one deterministic sequential LINO inference and persist provenance."""

    clock = time.perf_counter if clock is None else clock
    total_started = clock()
    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    with _pinned_lino_output_tree(config) as output_tree:
        _, _, _, lino_fd, _, _ = output_tree
        _invalidate_lino_run(lino_fd)
        if not config.save_exr:
            raise ValueError("save_exr must be true for authoritative LINO output")

        checkpoint = Path(config.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"LINO checkpoint does not exist: {checkpoint}")
        checkpoint_identity_before = file_identity(checkpoint, label="LINO checkpoint")
        checkpoint_bytes = read_file_bytes(checkpoint, label="LINO checkpoint")
        checkpoint_digest = sha256_bytes(checkpoint_bytes)
        checkpoint_identity = file_identity(checkpoint, label="LINO checkpoint")
        if checkpoint_identity != checkpoint_identity_before:
            raise ValueError("LINO checkpoint was replaced while reading its immutable snapshot")

        config_file_path: Path | None = None
        config_digest: str | None = None
        if config_path is not None:
            config_file_path = Path(config_path)
            config_raw = read_file_bytes(config_file_path, label="LINO config")
            config_digest = sha256_bytes(config_raw)

        selection_source_bytes: bytes | None = None
        if config.light_selection == "manifest":
            selection_source_bytes = read_file_bytes(
                config.effective_selection_manifest_path,
                label="selection manifest",
            )
        trained_export: Mapping[str, Any] | None = None
        if config.require_checkpoint_data_contract:
            if selection_source_bytes is None:
                raise ValueError(
                    "trained private inference requires an immutable selection manifest snapshot"
                )
            trained_export = _validate_trained_export_sidecar(
                config,
                checkpoint,
                checkpoint_bytes,
                checkpoint_digest,
                selection_source_bytes=selection_source_bytes,
            )
        device = _resolve_device(config)
        dtype = _precision_dtype(config)
        if model_loader is None and device.type != "cuda":
            raise RuntimeError("released LINO checkpoint inference requires CUDA bf16")
        print(f"Exploring {config.data_root}")
        manifest = build_dataset_manifest(
            config,
            selection_manifest_bytes=selection_source_bytes,
        )
        print(f"Found {len(manifest.objects)} objects!\n")
        print(f"Using device: {device}")
        result = _run_lino_inference_pinned(
            config,
            manifest,
            output_tree,
            model_loader=model_loader,
            dataset_factory=dataset_factory,
            device=device,
            dtype=dtype,
            checkpoint=checkpoint,
            checkpoint_bytes=checkpoint_bytes,
            checkpoint_digest=checkpoint_digest,
            checkpoint_identity=checkpoint_identity,
            config_file_path=config_file_path,
            config_digest=config_digest,
            selection_source_bytes=selection_source_bytes,
            trained_export=trained_export,
            clock=clock,
            total_started=total_started,
        )
    print(
        f"Inference complete: {len(manifest.objects)} objects -> "
        f"{config.lino_output_dir}"
    )
    print(f"Mean MAE ({result['mae_objects']} objects): {result['mean_mae']:.4f}")
    transfer = result["transfer_summary"]
    print(
        "Constant [0,0,1] baseline MAE: "
        f"{transfer['constant_normal_macro_mae']:.4f}"
    )
    print(
        "Best coordinate diagnostic: "
        f"{transfer['best_coordinate_transform']['label']} | "
        f"MAE {transfer['best_coordinate_macro_mae']:.4f}"
    )
    print(
        "Peak CUDA memory: allocated "
        f"{format_optional_gib(result['peak_cuda_allocated_bytes'])} | "
        "reserved "
        f"{format_optional_gib(result['peak_cuda_reserved_bytes'])}"
    )
    total_runtime_seconds = float(result["total_runtime_seconds"])
    print(f"Total inference time: {format_clock_duration(total_runtime_seconds)}")
    return result


__all__ = ["load_local_lino_checkpoint", "run_lino_inference"]
