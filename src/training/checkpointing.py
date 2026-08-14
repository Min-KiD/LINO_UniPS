"""Persistence contracts for private LINO training.

This module deliberately keeps three artifact classes separate:

* ``.pth`` files are raw model-only initialization/inference weights;
* ``.ckpt`` files contain the complete state needed to resume training; and
* an epoch publication is a small transaction over a checkpoint, export, and
  metrics files.

Checkpoint bytes are always parsed from one immutable file snapshot before a
live model or CUDA device is touched.  The implementation uses the secure
descriptor-relative publication helpers already used by the comparison path.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import random
import secrets
import stat
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import numpy as np
import torch

from src.comparison.provenance import (
    _secure_directory_flags,
    atomic_replace_bytes_at_fd,
    canonical_json_bytes,
    directory_identity,
    ensure_real_directory,
    open_or_create_directory,
    read_regular_bytes_at_fd,
    sha256_bytes,
)


ALLOWED_AUTHOR_EXTRAS = frozenset(
    {
        "image_encoder.backbone.aggregator.light_tokens_proj.conv_transpose.weight",
        "image_encoder.backbone.aggregator.light_tokens_proj.conv_transpose.bias",
        "image_encoder.backbone.aggregator.register_tokens_proj.conv_transpose.weight",
        "image_encoder.backbone.aggregator.register_tokens_proj.conv_transpose.bias",
    }
)

_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_ARTIFACT = "lino_private_training_checkpoint"
_EXPORT_ARTIFACT = "lino_private_inference_weights"
_CONTRACT_ARTIFACT = "lino_private_training_contract"
_SAFE_NAME = re.compile(r"^[^/\\]+$")
_EPOCH_NAME = re.compile(r"(?:lino_|private_|)?epoch[_-](\d+)", re.IGNORECASE)


class _FrozenList(tuple):
    """Tuple-backed list marker so thawing preserves PyTorch state shapes."""


@dataclass(frozen=True)
class TrainingProgress:
    """Progress at an epoch boundary."""

    completed_epoch: int
    next_epoch: int
    global_step: int


@dataclass(frozen=True)
class BestMetrics:
    """Best validation values observed so far."""

    mae: float
    loss: float
    epoch: int


@dataclass(frozen=True)
class StartupCheckpointPreflight:
    """Immutable CPU-only result of startup artifact validation."""

    mode: str
    path: Path | None
    sha256: str | None
    payload: Mapping[str, object] | None
    load_report: Mapping[str, object]


@dataclass(frozen=True)
class ResumeState:
    """State restored by a successful resume."""

    progress: TrainingProgress
    best: BestMetrics
    run_contract: Mapping[str, object]


@dataclass(frozen=True)
class ExportResult:
    """Paths and digest of one raw inference export."""

    weights_path: Path
    metadata_path: Path
    checkpoint_sha256: str


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("contract contains a non-finite value")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _schema_entries(expected_schema: object) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Normalize a model, state mapping, or schema iterable into a tuple."""

    if isinstance(expected_schema, torch.nn.Module):
        entries = [
            (name, tuple(int(dimension) for dimension in tensor.shape), str(tensor.dtype))
            for name, tensor in expected_schema.state_dict().items()
        ]
    elif isinstance(expected_schema, Mapping):
        entries = [
            (str(name), tuple(int(dimension) for dimension in tensor.shape), str(tensor.dtype))
            for name, tensor in expected_schema.items()
            if isinstance(tensor, torch.Tensor)
        ]
        if len(entries) != len(expected_schema):
            raise ValueError("expected schema mapping values must be tensors")
    else:
        try:
            raw_entries = list(expected_schema)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("expected_schema must be a model, state mapping, or iterable") from exc
        entries = []
        for entry in raw_entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                raise ValueError("expected schema entries must be (name, shape, dtype)")
            name, shape, dtype = entry
            if not isinstance(name, str) or not isinstance(dtype, str):
                raise ValueError("expected schema names and dtypes must be strings")
            if not isinstance(shape, (tuple, list)) or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            ):
                raise ValueError("expected schema shapes must contain non-negative integers")
            entries.append((name, tuple(shape), dtype))
    names = [name for name, _shape, _dtype in entries]
    if len(names) != len(set(names)):
        raise ValueError("expected schema contains duplicate parameter names")
    return tuple(entries)


def schema_fingerprint(schema: object) -> str:
    """Hash parameter names, shapes, and dtypes in their registered order."""

    entries = _schema_entries(schema)
    serializable = [[name, list(shape), dtype] for name, shape, dtype in entries]
    return sha256_bytes(canonical_json_bytes(serializable))


def _schema_check(
    state: Mapping[str, object],
    expected_schema: object,
    *,
    allow_author_extras: bool = False,
) -> dict[str, object]:
    expected = _schema_entries(expected_schema)
    expected_map = {name: (shape, dtype) for name, shape, dtype in expected}
    actual_keys = set(state)
    expected_keys = set(expected_map)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    disallowed = [name for name in unexpected if name not in ALLOWED_AUTHOR_EXTRAS]
    if missing:
        raise ValueError(f"checkpoint missing model keys: {', '.join(missing[:8])}")
    if disallowed or (unexpected and not allow_author_extras):
        raise ValueError(
            "checkpoint has unexpected model keys: "
            + ", ".join(disallowed or unexpected)
        )
    for name, (shape, _dtype) in expected_map.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint model key {name} is not a tensor")
        if tuple(int(dimension) for dimension in value.shape) != shape:
            raise ValueError(
                f"checkpoint model key {name} shape mismatch: "
                f"got {tuple(value.shape)}, expected {shape}"
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"checkpoint model key {name} contains non-finite values")
    return {
        "expected_schema_sha256": schema_fingerprint(expected),
        "actual_schema_sha256": schema_fingerprint(
            tuple(
                (name, tuple(value.shape), str(value.dtype))
                for name, value in state.items()
                if isinstance(value, torch.Tensor)
            )
        ),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "author_extras": sorted(name for name in unexpected if name in ALLOWED_AUTHOR_EXTRAS),
    }


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        # Optimizer state dictionaries intentionally use integer parameter
        # identifiers.  Preserve those keys; only JSON-facing contracts need
        # string normalization.
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    if isinstance(value, list):
        return [_thaw(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _read_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    parent = absolute.parent
    try:
        parent_identity = directory_identity(parent, label=f"{label} parent directory")
        flags = _secure_directory_flags()
        descriptor = os.open(str(parent), flags)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} is missing or unreadable: {path}") from exc
    try:
        # The provenance helper pins the parent directory and verifies the
        # final regular-file inode before and after reading its bytes.
        raw = read_regular_bytes_at_fd(
            descriptor,
            absolute.name,
            expected_directory_identity=parent_identity,
            label=label,
            directory_path=parent,
        )
    except OSError as exc:
        raise ValueError(f"failed to read {label}: {path}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if not raw:
        raise ValueError(f"{label} is empty: {path}")
    return raw, hashlib.sha256(raw).hexdigest()


def _load_torch_bytes(raw: bytes, *, label: str) -> object:
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch versions before ``weights_only`` are still supported.
        try:
            return torch.load(io.BytesIO(raw), map_location="cpu")
        except Exception as exc:  # pragma: no cover - version-specific fallback
            raise ValueError(f"failed to parse {label}") from exc
    except Exception as exc:
        raise ValueError(f"failed to parse {label}") from exc


def _validate_progress(value: object) -> TrainingProgress:
    mapping = _as_mapping(value, "progress")
    expected = {"completed_epoch", "next_epoch", "global_step"}
    if set(mapping) != expected:
        raise ValueError("progress must contain completed_epoch, next_epoch, and global_step")
    for key in sorted(expected):
        item = mapping[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"progress.{key} must be a non-negative integer")
    completed = int(mapping["completed_epoch"])
    next_epoch = int(mapping["next_epoch"])
    if next_epoch != completed + 1:
        raise ValueError("progress.next_epoch must equal completed_epoch + 1")
    return TrainingProgress(completed, next_epoch, int(mapping["global_step"]))


def _validate_best(value: object) -> BestMetrics:
    mapping = _as_mapping(value, "best_metrics")
    if set(mapping) != {"mae", "loss", "epoch"}:
        raise ValueError("best_metrics must contain mae, loss, and epoch")
    mae, loss = mapping["mae"], mapping["loss"]
    # Positive infinity is the intentional sentinel for a run with no best
    # validation result yet.  NaN and negative infinity are never meaningful.
    if (
        isinstance(mae, bool)
        or not isinstance(mae, (int, float))
        or np.isnan(mae)
        or mae == -float("inf")
    ):
        raise ValueError("best_metrics.mae must be numeric and not NaN")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or np.isnan(loss)
        or loss == -float("inf")
    ):
        raise ValueError("best_metrics.loss must be numeric and not NaN")
    epoch = mapping["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("best_metrics.epoch must be a non-negative integer")
    return BestMetrics(float(mae), float(loss), int(epoch))


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, CPU Torch, and available CUDA RNG state."""

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().detach().cpu().clone(),
        "cuda": cuda_states,
    }


def restore_rng_state(state: object) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    mapping = _as_mapping(state, "rng_state")
    if set(mapping) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("rng_state has an invalid key set")
    random.setstate(_thaw(mapping["python"]))  # type: ignore[arg-type]
    numpy_state = _thaw(mapping["numpy"])
    if not isinstance(numpy_state, tuple):
        raise ValueError("rng_state.numpy must be a tuple")
    np.random.set_state(numpy_state)
    torch_state = _thaw(mapping["torch"])
    if not isinstance(torch_state, torch.Tensor):
        raise ValueError("rng_state.torch must be a tensor")
    torch.set_rng_state(torch_state.to(device="cpu", dtype=torch.uint8))
    cuda_states = _thaw(mapping["cuda"])
    if torch.cuda.is_available():
        if not isinstance(cuda_states, (tuple, list)):
            raise ValueError("rng_state.cuda must be a sequence")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _config_snapshot(config: object) -> dict[str, object]:
    if isinstance(config, Mapping):
        raw = dict(config)
    elif is_dataclass(config):
        raw = {field.name: getattr(config, field.name) for field in fields(config)}
    elif hasattr(config, "__dict__"):
        raw = dict(vars(config))
    else:
        raw = {}
    return {str(key): _json_safe(value) for key, value in raw.items()}


def build_run_contract(
    config: object | None = None,
    *,
    architecture_schema: object | None = None,
    train_manifest_sha256: str | None = None,
    test_manifest_sha256: str | None = None,
    final_selection_manifest_sha256: str | None = None,
    source_revision: str | None = None,
    runtime_versions: Mapping[str, object] | None = None,
    run_kind: str = "experiment",
    comparable: bool | None = None,
    base_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the immutable data/architecture contract recorded in a checkpoint."""

    if run_kind not in {"experiment", "smoke"}:
        raise ValueError("run_kind must be experiment or smoke")
    contract: dict[str, object] = (
        {str(key): _json_safe(value) for key, value in base_contract.items()}
        if base_contract is not None
        else {}
    )
    contract.setdefault("schema_version", _CHECKPOINT_SCHEMA_VERSION)
    contract.setdefault("artifact_kind", _CONTRACT_ARTIFACT)
    contract["run_kind"] = run_kind
    contract["comparable"] = bool(run_kind == "experiment" if comparable is None else comparable)
    if run_kind == "smoke":
        contract["comparable"] = False

    snapshot = _config_snapshot(config) if config is not None else {}
    if snapshot:
        epochs = snapshot.get("epochs", snapshot.get("total_epochs"))
        # Startup mode, checkpoint paths, output destination, and total epoch
        # count are operational knobs.  They may change when resuming; every
        # data/model/optimization field remains in this snapshot and therefore
        # remains part of the compatibility contract.
        strict_snapshot = dict(snapshot)
        for operational in (
            "startup_mode",
            "init_checkpoint",
            "resume_checkpoint",
            "save_dir",
            "epochs",
        ):
            strict_snapshot.pop(operational, None)
        contract["config_snapshot"] = strict_snapshot
        if isinstance(epochs, int) and not isinstance(epochs, bool):
            contract["total_epochs"] = int(epochs)
    if architecture_schema is not None:
        normalized = _schema_entries(architecture_schema)
        contract["architecture_schema_sha256"] = schema_fingerprint(normalized)
    if train_manifest_sha256 is not None:
        contract["train_manifest_sha256"] = str(train_manifest_sha256)
    if test_manifest_sha256 is not None:
        contract["test_manifest_sha256"] = str(test_manifest_sha256)
    if final_selection_manifest_sha256 is not None:
        contract["final_selection_manifest_sha256"] = str(final_selection_manifest_sha256)
    if source_revision is not None:
        contract["source_revision"] = str(source_revision)
    if runtime_versions is not None:
        contract["runtime_versions"] = _json_safe(runtime_versions)
    return {str(key): _json_safe(value) for key, value in contract.items()}


def _contract_compatible(saved: Mapping[str, object], expected: Mapping[str, object]) -> None:
    """Require exact contract equality except for an increased total epoch count."""

    saved_keys, expected_keys = set(saved), set(expected)
    if saved_keys != expected_keys:
        missing = sorted(expected_keys - saved_keys)
        extra = sorted(saved_keys - expected_keys)
        raise ValueError(f"run contract fingerprint mismatch (missing={missing}, extra={extra})")
    for key in sorted(saved_keys):
        if key == "total_epochs":
            old, new = saved[key], expected[key]
            if (
                isinstance(old, bool)
                or isinstance(new, bool)
                or not isinstance(old, int)
                or not isinstance(new, int)
                or new < old
            ):
                raise ValueError("run contract total_epochs may only increase")
            continue
        if saved[key] != expected[key]:
            raise ValueError(f"run contract fingerprint mismatch at {key}")


def _expected_schema_from_args(expected_schema: object | None, model: object | None) -> object:
    if expected_schema is not None:
        return expected_schema
    if isinstance(model, torch.nn.Module):
        return model
    raise ValueError("expected_schema or model is required for checkpoint preflight")


def preflight_startup_checkpoint(
    mode: str,
    path: str | Path | None,
    *,
    expected_schema: object | None = None,
    expected_contract: Mapping[str, object] | None = None,
    model: torch.nn.Module | None = None,
) -> StartupCheckpointPreflight:
    """Validate startup bytes and schema without constructing or mutating a model."""

    if mode not in {"cold_start", "init_checkpoint", "resume"}:
        raise ValueError("startup mode must be cold_start, init_checkpoint, or resume")
    if mode == "cold_start":
        if path is not None:
            raise ValueError("cold_start cannot receive a checkpoint path")
        return StartupCheckpointPreflight("cold_start", None, None, None, MappingProxyType({}))
    if path is None:
        raise ValueError(f"{mode} requires a checkpoint path")
    checkpoint = Path(path)
    required_suffix = ".pth" if mode == "init_checkpoint" else ".ckpt"
    if checkpoint.suffix != required_suffix:
        if mode == "resume":
            raise ValueError("resume requires a full private-training .ckpt artifact")
        raise ValueError("init_checkpoint requires a model-only .pth artifact")
    schema = _expected_schema_from_args(expected_schema, model)
    normalized_schema = _schema_entries(schema)
    raw, digest = _read_snapshot(checkpoint, label=f"private training {mode} checkpoint")
    payload = _load_torch_bytes(raw, label=f"private training {mode} checkpoint")
    if mode == "init_checkpoint":
        payload_mapping = _as_mapping(payload, "initial checkpoint")
        if "state_dict" in payload_mapping:
            if set(payload_mapping) != {"state_dict"}:
                raise ValueError("initial checkpoint wrapper may contain only state_dict")
            state = _as_mapping(payload_mapping["state_dict"], "initial checkpoint state_dict")
        else:
            state = payload_mapping
        report = _schema_check(state, normalized_schema, allow_author_extras=True)
        report.update({"artifact_kind": "lino_private_initial_weights", "checkpoint_sha256": digest})
        frozen = _freeze({"state_dict": state})
        assert isinstance(frozen, Mapping)
        return StartupCheckpointPreflight(
            mode,
            checkpoint,
            digest,
            frozen,
            MappingProxyType(dict(report)),
        )

    payload_mapping = _as_mapping(payload, "resume checkpoint")
    required_keys = {
        "schema_version",
        "artifact_kind",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "progress",
        "best_metrics",
        "rng_state",
        "run_contract",
    }
    if set(payload_mapping) != required_keys:
        raise ValueError("resume checkpoint must be a full private-training .ckpt artifact")
    if payload_mapping["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported private-training checkpoint schema_version")
    if payload_mapping["artifact_kind"] != _CHECKPOINT_ARTIFACT:
        raise ValueError("resume checkpoint artifact_kind is not full private-training .ckpt")
    state = _as_mapping(payload_mapping["model_state_dict"], "model_state_dict")
    report = _schema_check(state, normalized_schema, allow_author_extras=False)
    progress = _validate_progress(payload_mapping["progress"])
    best = _validate_best(payload_mapping["best_metrics"])
    rng = _as_mapping(payload_mapping["rng_state"], "rng_state")
    if set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("resume checkpoint rng_state is incomplete")
    contract = _as_mapping(payload_mapping["run_contract"], "run_contract")
    if expected_contract is not None:
        _contract_compatible(contract, expected_contract)
    report.update(
        {
            "artifact_kind": _CHECKPOINT_ARTIFACT,
            "checkpoint_sha256": digest,
            "progress": asdict(progress),
            "best_metrics": asdict(best),
            "expected_contract": _json_safe(expected_contract) if expected_contract is not None else None,
        }
    )
    frozen = _freeze(payload_mapping)
    assert isinstance(frozen, Mapping)
    return StartupCheckpointPreflight(
        mode,
        checkpoint,
        digest,
        frozen,
        MappingProxyType(dict(report)),
    )


def _cpu_copy(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _torch_bytes(payload: object) -> bytes:
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _atomic_replace(path: Path, payload: bytes, *, label: str) -> None:
    destination = Path(path)
    parent = ensure_real_directory(destination.parent, label=f"{label} directory")
    descriptor: int | None = None
    try:
        descriptor, identity, absolute = open_or_create_directory(parent, label=f"{label} directory")
        atomic_replace_bytes_at_fd(
            descriptor,
            destination.name,
            payload,
            expected_directory_identity=identity,
            label=label,
            directory_path=absolute,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"failed to atomically write {label}: {destination}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def save_resume_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    progress: TrainingProgress,
    best: BestMetrics,
    contract: Mapping[str, object],
) -> Path:
    """Atomically save a complete CPU-portable private-training checkpoint."""

    checkpoint = Path(path)
    if checkpoint.suffix != ".ckpt":
        raise ValueError("resume checkpoint must use the .ckpt suffix")
    progress = _validate_progress(asdict(progress))
    best = _validate_best(asdict(best))
    contract = _as_mapping(contract, "run_contract")
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": _CHECKPOINT_ARTIFACT,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_copy(scheduler.state_dict()),
        "progress": asdict(progress),
        "best_metrics": asdict(best),
        "rng_state": _cpu_copy(capture_rng_state()),
        "run_contract": _json_safe(contract),
    }
    _atomic_replace(checkpoint, _torch_bytes(payload), label="private-training checkpoint")
    return checkpoint


def _move_value_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_parameters(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state is not None:
                optimizer.state[parameter] = _move_value_to_device(state, parameter.device)  # type: ignore[assignment]


def _model_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return _schema_entries(model)


def _preflight_for_load(
    source: str | Path | StartupCheckpointPreflight,
    *,
    mode: str,
    model: torch.nn.Module,
    expected_contract: Mapping[str, object] | None,
) -> StartupCheckpointPreflight:
    if isinstance(source, StartupCheckpointPreflight):
        preflight = source
        if preflight.mode != mode:
            raise ValueError(f"checkpoint preflight mode must be {mode}")
        return preflight
    return preflight_startup_checkpoint(
        mode,
        source,
        expected_schema=_model_schema(model),
        expected_contract=expected_contract,
    )


def load_initial_weights(
    source: str | Path | StartupCheckpointPreflight,
    *,
    model: torch.nn.Module,
    expected_contract: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Load model-only initialization weights after strict preflight."""

    preflight = _preflight_for_load(
        source,
        mode="init_checkpoint",
        model=model,
        expected_contract=expected_contract,
    )
    if preflight.payload is None:
        raise ValueError("initial checkpoint preflight has no payload")
    frozen_mapping = _as_mapping(preflight.payload, "initial checkpoint preflight")
    state = _as_mapping(frozen_mapping["state_dict"], "initial checkpoint state_dict")
    live_report = _schema_check(state, _model_schema(model), allow_author_extras=True)
    thawed = _thaw(state)
    result = model.load_state_dict(thawed, strict=False)  # type: ignore[arg-type]
    missing = sorted(result.missing_keys)
    unexpected = sorted(result.unexpected_keys)
    if missing or any(name not in ALLOWED_AUTHOR_EXTRAS for name in unexpected):
        raise ValueError("initial checkpoint load changed the released model schema")
    merged = dict(preflight.load_report)
    merged.update(live_report)
    return MappingProxyType(merged)


def load_resume_checkpoint(
    source: str | Path | StartupCheckpointPreflight,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expected_contract: Mapping[str, object] | None = None,
) -> ResumeState:
    """Restore model/optimizer/scheduler/RNG transactionally from one preflight."""

    preflight = _preflight_for_load(
        source,
        mode="resume",
        model=model,
        expected_contract=expected_contract,
    )
    if preflight.payload is None:
        raise ValueError("resume checkpoint preflight has no payload")
    payload = _as_mapping(preflight.payload, "resume checkpoint preflight")
    live_schema = _model_schema(model)
    expected_fingerprint = preflight.load_report.get("expected_schema_sha256")
    if expected_fingerprint != schema_fingerprint(live_schema):
        raise ValueError("live model schema changed after checkpoint preflight")
    if expected_contract is not None:
        _contract_compatible(
            _as_mapping(payload["run_contract"], "run_contract"),
            expected_contract,
        )

    old_model = copy.deepcopy(model.state_dict())
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_scheduler = copy.deepcopy(scheduler.state_dict())
    old_rng = capture_rng_state()
    progress = _validate_progress(payload["progress"])
    best = _validate_best(payload["best_metrics"])
    try:
        model.load_state_dict(_thaw(payload["model_state_dict"]), strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(_thaw(payload["optimizer_state_dict"]))  # type: ignore[arg-type]
        scheduler.load_state_dict(_thaw(payload["scheduler_state_dict"]))  # type: ignore[attr-defined]
        _move_optimizer_state_to_parameters(optimizer)
        restore_rng_state(payload["rng_state"])
    except Exception as exc:
        try:
            model.load_state_dict(old_model, strict=True)
            optimizer.load_state_dict(old_optimizer)
            scheduler.load_state_dict(old_scheduler)  # type: ignore[attr-defined]
            _move_optimizer_state_to_parameters(optimizer)
            restore_rng_state(old_rng)
        except Exception as rollback_exc:  # pragma: no cover - catastrophic runtime failure
            raise RuntimeError("resume failed and rollback also failed") from rollback_exc
        raise ValueError("failed to restore private-training checkpoint transactionally") from exc
    return ResumeState(
        progress=progress,
        best=best,
        run_contract=MappingProxyType(dict(_thaw(payload["run_contract"]))),  # type: ignore[arg-type]
    )


def _validate_export_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model state {name} is not a tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"model state {name} contains non-finite values")
        state[name] = tensor.detach().cpu().clone()
    _schema_check(state, _model_schema(model), allow_author_extras=False)
    return state


def export_inference_weights(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_factory: Callable[[], torch.nn.Module],
    metadata: Mapping[str, object],
) -> ExportResult:
    """Export raw model weights and strictly round-trip them through a fresh CPU model."""

    weights_path = Path(path)
    if weights_path.suffix != ".pth":
        raise ValueError("inference export must use the .pth suffix")
    state = _validate_export_state(model)
    raw = _torch_bytes(state)
    try:
        with torch.device("cpu"):
            fresh_model = model_factory()
        if not isinstance(fresh_model, torch.nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module")
        fresh_model = fresh_model.to(device="cpu")
        fresh_model.load_state_dict(state, strict=True)
        if schema_fingerprint(fresh_model) != schema_fingerprint(model):
            raise ValueError("fresh export model schema differs from live model")
    except Exception as exc:
        if isinstance(exc, (TypeError, ValueError)):
            raise
        raise ValueError("failed to validate raw inference export") from exc
    digest = sha256_bytes(raw)
    sidecar_data = {str(key): _json_safe(value) for key, value in metadata.items()}
    sidecar_data.setdefault("artifact_kind", _EXPORT_ARTIFACT)
    sidecar_data["artifact_kind"] = _EXPORT_ARTIFACT
    sidecar_data["checkpoint_sha256"] = digest
    sidecar_data.setdefault("epoch", None)
    sidecar_data.setdefault("preprocessing_version", None)
    sidecar_data.setdefault("data_contract", None)
    sidecar_data.setdefault("source_revision", None)
    sidecar_data.setdefault("run_kind", "experiment")
    sidecar_data.setdefault("comparable", sidecar_data["run_kind"] == "experiment")
    sidecar_data.setdefault("architecture_schema_sha256", schema_fingerprint(model))
    sidecar = weights_path.with_suffix(".json")
    sidecar_raw = (json.dumps(sidecar_data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    _atomic_replace(weights_path, raw, label="raw LINO inference weights")
    _atomic_replace(sidecar, sidecar_raw, label="raw LINO inference sidecar")
    return ExportResult(weights_path, sidecar, digest)


def _safe_artifact_name(name: str) -> str:
    if not isinstance(name, str) or not name or not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"artifact name must be one safe path component: {name!r}")
    return name


def _stage_file(directory_fd: int, name: str, payload: bytes) -> str:
    safe_name = _safe_artifact_name(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    temporary: str | None = None
    for _ in range(16):
        temporary = f".{safe_name}.{secrets.token_hex(12)}.tmp"
        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError("short artifact write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return temporary
        except FileExistsError:
            continue
        except OSError as exc:
            try:
                if temporary is not None:
                    os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
            raise ValueError(f"failed to stage artifact {name}") from exc
    raise ValueError(f"failed to allocate temporary artifact {name}")


def _validate_staged_artifact(name: str, payload: bytes) -> None:
    suffix = Path(name).suffix.lower()
    if suffix in {".ckpt", ".pth"}:
        parsed = _load_torch_bytes(payload, label=f"staged artifact {name}")
        if not isinstance(parsed, Mapping):
            raise ValueError(f"staged artifact {name} must contain a mapping")
        if suffix == ".ckpt" and parsed.get("artifact_kind") != _CHECKPOINT_ARTIFACT:
            raise ValueError(f"staged checkpoint {name} is not a private-training checkpoint")
        if suffix == ".pth" and "state_dict" in parsed:
            raise ValueError(f"staged inference export {name} must be raw model weights")
    elif suffix == ".json":
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"staged artifact {name} is not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError(f"staged artifact {name} must contain a JSON mapping")
        if parsed.get("artifact_kind") not in {None, _EXPORT_ARTIFACT}:
            raise ValueError(f"staged sidecar {name} has an invalid artifact_kind")
    elif not isinstance(payload, bytes) or not payload:
        raise ValueError(f"staged artifact {name} is empty")


def publish_epoch_artifacts(
    run_dir: str | Path,
    *,
    artifacts: Mapping[str, bytes] | None = None,
    epoch: int | None = None,
    checkpoint_bytes: bytes | None = None,
    metrics_csv_bytes: bytes | None = None,
    export: ExportResult | None = None,
) -> Mapping[str, Path]:
    """Publish a validated artifact bundle with rollback on any replacement failure.

    ``artifacts`` is the explicit API: keys are basenames and values are exact
    bytes.  The convenience arguments are useful to the trainer and are merged
    into that mapping before staging.
    """

    destination = Path(run_dir)
    parent = ensure_real_directory(destination, label="epoch artifact directory")
    merged: dict[str, bytes] = {}
    if artifacts is not None:
        for name, payload in artifacts.items():
            if not isinstance(payload, bytes):
                raise TypeError(f"artifact {name} payload must be bytes")
            merged[_safe_artifact_name(name)] = payload
    if checkpoint_bytes is not None:
        merged["last.ckpt"] = checkpoint_bytes
        if epoch is not None:
            merged[f"lino_epoch_{int(epoch):03d}.ckpt"] = checkpoint_bytes
    if metrics_csv_bytes is not None:
        merged["metrics.csv"] = metrics_csv_bytes
    if export is not None:
        weights, weights_digest = _read_snapshot(export.weights_path, label="raw LINO export")
        sidecar, _ = _read_snapshot(export.metadata_path, label="raw LINO export sidecar")
        if weights_digest != export.checkpoint_sha256:
            raise ValueError("raw LINO export digest changed before publication")
        merged[export.weights_path.name] = weights
        merged[export.metadata_path.name] = sidecar
    if not merged:
        raise ValueError("epoch artifact bundle must not be empty")
    for name, payload in merged.items():
        _validate_staged_artifact(name, payload)

    directory_fd: int | None = None
    staged: dict[str, str] = {}
    backups: dict[str, str] = {}
    old_exists: dict[str, bool] = {}
    replaced: list[str] = []
    try:
        directory_fd, identity, absolute = open_or_create_directory(parent, label="epoch artifact directory")
        for name, payload in merged.items():
            staged[name] = _stage_file(directory_fd, name, payload)
        for name in merged:
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                old_exists[name] = False
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"epoch artifact destination must be a regular file: {name}")
            old_exists[name] = True
            backup = f".{name}.{secrets.token_hex(12)}.bak"
            os.link(name, backup, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            backups[name] = backup
        directory_info = os.fstat(directory_fd)
        if (int(directory_info.st_dev), int(directory_info.st_ino)) != (int(identity["dev"]), int(identity["ino"])):
            raise ValueError("epoch artifact directory was replaced before publication")
        for name, temporary in staged.items():
            os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            replaced.append(name)
            staged[name] = ""
        os.fsync(directory_fd)
        for backup in backups.values():
            os.unlink(backup, dir_fd=directory_fd)
        return {name: destination / name for name in merged}
    except Exception as exc:
        # Restore every public name, including names replaced before a later
        # os.replace failure.  A best-effort rollback still removes staged
        # files, and the original exception remains the useful diagnostic.
        if directory_fd is not None:
            for name in reversed(replaced):
                try:
                    if old_exists.get(name, False) and name in backups:
                        os.replace(backups[name], name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                        backups.pop(name, None)
                    elif not old_exists.get(name, False):
                        os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    pass
            for name, backup in list(backups.items()):
                try:
                    os.unlink(backup, dir_fd=directory_fd)
                except OSError:
                    pass
            for temporary in staged.values():
                if temporary:
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except OSError:
                        pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        if isinstance(exc, (TypeError, ValueError)):
            raise
        raise ValueError("failed to publish epoch artifact bundle; previous bundle restored") from exc
    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def apply_artifact_retention(
    run_dir: str | Path,
    *,
    keep_milestone_epochs: Sequence[int] = (),
    latest_epoch: int | None = None,
) -> tuple[Path, ...]:
    """Delete non-milestone epoch artifacts while retaining aliases and best files."""

    root = Path(run_dir)
    keep = {int(epoch) for epoch in keep_milestone_epochs}
    if any(epoch <= 0 for epoch in keep) or (latest_epoch is not None and latest_epoch <= 0):
        raise ValueError("retention epochs must be positive")
    removed: list[Path] = []
    directories = [root]
    for child in (root / "checkpoints", root / "exports"):
        if child.is_dir() and not child.is_symlink():
            directories.append(child)
    for directory in directories:
        for candidate in directory.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                continue
            match = _EPOCH_NAME.search(candidate.stem)
            if match is None:
                continue
            epoch = int(match.group(1))
            if epoch in keep or (latest_epoch is not None and epoch == latest_epoch):
                continue
            candidate.unlink()
            removed.append(candidate)
    return tuple(sorted(removed))


__all__ = [
    "ALLOWED_AUTHOR_EXTRAS",
    "BestMetrics",
    "ExportResult",
    "ResumeState",
    "StartupCheckpointPreflight",
    "TrainingProgress",
    "apply_artifact_retention",
    "build_run_contract",
    "capture_rng_state",
    "export_inference_weights",
    "load_initial_weights",
    "load_resume_checkpoint",
    "preflight_startup_checkpoint",
    "publish_epoch_artifacts",
    "restore_rng_state",
    "save_resume_checkpoint",
    "schema_fingerprint",
]
