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
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import numpy as np
import torch

from src.comparison.provenance import (
    _secure_directory_flags,
    assert_directory_path_identity,
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
    # The parsed payload above is diagnostic only.  Loaders use this private,
    # immutable byte snapshot and reparse it into fresh CPU objects so callers
    # cannot mutate an already validated tensor into an authoritative state.
    _raw_bytes: bytes | None = field(default=None, repr=False, compare=False)


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
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
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
    for name, (shape, dtype) in expected_map.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint model key {name} is not a tensor")
        if tuple(int(dimension) for dimension in value.shape) != shape:
            raise ValueError(
                f"checkpoint model key {name} shape mismatch: "
                f"got {tuple(value.shape)}, expected {shape}"
            )
        if str(value.dtype) != dtype:
            raise ValueError(
                f"checkpoint model key {name} dtype mismatch: "
                f"got {value.dtype}, expected {dtype}"
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


def _open_existing_directory_path(path: Path, *, label: str) -> tuple[int, dict[str, int], Path]:
    """Open every directory component with no-follow descriptor checks."""

    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ValueError(f"{label} must be absolute")
    flags = _secure_directory_flags()
    descriptor = os.open(os.sep, flags)
    try:
        for component in parts[1:]:
            info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"{label} contains a symlink or non-directory component")
            expected = (int(info.st_dev), int(info.st_ino))
            child = os.open(component, flags, dir_fd=descriptor)
            child_info = os.fstat(child)
            actual = (int(child_info.st_dev), int(child_info.st_ino))
            if actual != expected:
                os.close(child)
                raise ValueError(f"{label} was replaced while opening")
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        return descriptor, {"dev": int(info.st_dev), "ino": int(info.st_ino)}, absolute
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    parent = absolute.parent
    try:
        descriptor, parent_identity, parent_absolute = _open_existing_directory_path(
            parent,
            label=f"{label} parent directory",
        )
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
            directory_path=parent_absolute,
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


def _validate_cpu_structure(value: object, label: str, *, _depth: int = 0) -> None:
    """Reject arbitrary objects, non-CPU tensors, and non-finite values."""

    if _depth > 32:
        raise ValueError(f"{label} is nested too deeply")
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{label} tensor must be on CPU")
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{label} contains non-finite tensor values")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{label} contains non-finite array values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool, type(None))):
                _validate_cpu_structure(item, f"{label}[{key!r}]", _depth=_depth + 1)
            else:
                raise ValueError(f"{label} contains an unsupported mapping key")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_cpu_structure(item, f"{label}[{index}]", _depth=_depth + 1)
        return
    if isinstance(value, (str, bool, int)) or value is None:
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    raise ValueError(f"{label} contains an unsupported value type: {type(value).__name__}")


def _validate_optimizer_state(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("optimizer_state_dict must be a mapping")
    if set(value) != {"state", "param_groups"}:
        raise ValueError("optimizer_state_dict must contain state and param_groups")
    states = value["state"]
    groups = value["param_groups"]
    if not isinstance(states, Mapping):
        raise ValueError("optimizer_state_dict.state must be a mapping")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("optimizer_state_dict.param_groups must be a nonempty list")
    parameter_ids: list[int] = []
    for parameter_id, state in states.items():
        if isinstance(parameter_id, bool) or not isinstance(parameter_id, int) or parameter_id < 0:
            raise ValueError("optimizer_state_dict.state keys must be non-negative integers")
        if not isinstance(state, Mapping):
            raise ValueError("optimizer_state_dict state entries must be mappings")
        _validate_cpu_structure(state, f"optimizer_state_dict.state[{parameter_id}]")
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or "params" not in group:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}] is invalid")
        params = group["params"]
        if not isinstance(params, (list, tuple)) or not params:
            raise ValueError(f"optimizer_state_dict.param_groups[{index}].params is invalid")
        for parameter_id in params:
            if isinstance(parameter_id, bool) or not isinstance(parameter_id, int) or parameter_id < 0:
                raise ValueError("optimizer parameter ids must be non-negative integers")
            parameter_ids.append(parameter_id)
        _validate_cpu_structure(group, f"optimizer_state_dict.param_groups[{index}]")
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("optimizer parameter ids must be unique")
    if not set(states).issubset(set(parameter_ids)):
        raise ValueError("optimizer state contains an id absent from param_groups")


def _validate_scheduler_state(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("scheduler_state_dict must be a mapping")
    required = {
        "step_size",
        "gamma",
        "base_lrs",
        "last_epoch",
        "verbose",
        "_step_count",
        "_get_lr_called_within_step",
        "_last_lr",
    }
    if set(value) != required:
        raise ValueError("scheduler_state_dict is not the exact StepLR state schema")
    if (
        isinstance(value["step_size"], bool)
        or not isinstance(value["step_size"], int)
        or value["step_size"] <= 0
        or isinstance(value["last_epoch"], bool)
        or not isinstance(value["last_epoch"], int)
        or value["last_epoch"] < -1
        or isinstance(value["_step_count"], bool)
        or not isinstance(value["_step_count"], int)
        or value["_step_count"] < 1
        or not isinstance(value["verbose"], bool)
        or not isinstance(value["_get_lr_called_within_step"], bool)
    ):
        raise ValueError("scheduler_state_dict has invalid StepLR scalar fields")
    if (
        isinstance(value["gamma"], bool)
        or not isinstance(value["gamma"], (int, float))
        or not np.isfinite(value["gamma"])
        or value["gamma"] <= 0
    ):
        raise ValueError("scheduler_state_dict.gamma must be a positive finite number")
    base_lrs, last_lrs = value["base_lrs"], value["_last_lr"]
    if (
        not isinstance(base_lrs, (list, tuple))
        or not base_lrs
        or not isinstance(last_lrs, (list, tuple))
        or len(last_lrs) != len(base_lrs)
        or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not np.isfinite(rate)
            for rate in (*base_lrs, *last_lrs)
        )
    ):
        raise ValueError("scheduler_state_dict learning-rate lists are invalid")
    _validate_cpu_structure(value, "scheduler_state_dict")


def _validate_scheduler_against_contract(
    value: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    groups = optimizer_state["param_groups"]
    if len(value["base_lrs"]) != len(groups):  # type: ignore[arg-type]
        raise ValueError("scheduler_state_dict does not match optimizer param_groups")
    snapshot = contract["config_snapshot"]
    if not isinstance(snapshot, Mapping):
        raise ValueError("run_contract config_snapshot must be a mapping")
    configured_step = snapshot.get("scheduler_step_size")
    if configured_step is not None and value["step_size"] != configured_step:
        raise ValueError("scheduler_state_dict.step_size conflicts with run contract")
    configured_gamma = snapshot.get("scheduler_gamma")
    if configured_gamma is not None and value["gamma"] != configured_gamma:
        raise ValueError("scheduler_state_dict.gamma conflicts with run contract")


def _validate_rng_state(value: object, contract: Mapping[str, object] | None = None) -> None:
    if not isinstance(value, Mapping) or set(value) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("rng_state must contain python, numpy, torch, and cuda")
    python_state = value["python"]
    try:
        random.Random().setstate(copy.deepcopy(python_state))
    except (TypeError, ValueError) as exc:
        raise ValueError("rng_state.python is not accepted by random.Random.setstate") from exc
    if (
        not isinstance(python_state, tuple)
        or len(python_state) != 3
        or isinstance(python_state[0], bool)
        or not isinstance(python_state[0], int)
        or not isinstance(python_state[1], tuple)
        or len(python_state[1]) < 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in python_state[1])
        or (python_state[2] is not None and not isinstance(python_state[2], float))
    ):
        raise ValueError("rng_state.python has an invalid random.Random state")
    if python_state[2] is not None and not np.isfinite(python_state[2]):
        raise ValueError("rng_state.python gaussian cache is non-finite")
    numpy_state = value["numpy"]
    if (
        not isinstance(numpy_state, tuple)
        or len(numpy_state) != 5
        or not isinstance(numpy_state[0], str)
        or not isinstance(numpy_state[1], np.ndarray)
        or numpy_state[1].dtype.kind not in "ui"
        or numpy_state[1].ndim != 1
        or numpy_state[1].size == 0
        or isinstance(numpy_state[2], bool)
        or not isinstance(numpy_state[2], int)
        or isinstance(numpy_state[3], bool)
        or not isinstance(numpy_state[3], (int, np.integer))
        or isinstance(numpy_state[4], bool)
        or not isinstance(numpy_state[4], (int, float, np.integer, np.floating))
    ):
        raise ValueError("rng_state.numpy has an invalid NumPy state")
    if not np.isfinite(float(numpy_state[4])):
        raise ValueError("rng_state.numpy cache is non-finite")
    try:
        np.random.RandomState().set_state(
            (
                numpy_state[0],
                numpy_state[1].copy(),
                numpy_state[2],
                numpy_state[3],
                numpy_state[4],
            )
        )
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("rng_state.numpy is not accepted by RandomState.set_state") from exc
    torch_state = value["torch"]
    expected_torch_bytes = int(torch.Generator(device="cpu").get_state().numel())
    if (
        not isinstance(torch_state, torch.Tensor)
        or torch_state.device.type != "cpu"
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
        or torch_state.numel() == 0
        or torch_state.numel() != expected_torch_bytes
    ):
        raise ValueError("rng_state.torch has an invalid CPU uint8 state")
    try:
        torch.Generator(device="cpu").set_state(torch_state.detach().clone())
    except (RuntimeError, TypeError) as exc:
        raise ValueError("rng_state.torch is not accepted by a CPU generator") from exc
    cuda_states = value["cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise ValueError("rng_state.cuda must be a list of CPU uint8 states")
    expected_cuda_count: int | None = None
    if contract is not None:
        raw_count = contract.get("cuda_device_count")
        runtime_versions = contract.get("runtime_versions")
        if raw_count is None and isinstance(runtime_versions, Mapping):
            raw_count = runtime_versions.get("cuda_device_count")
        if raw_count is not None:
            if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
                raise ValueError("run_contract cuda_device_count is invalid")
            expected_cuda_count = int(raw_count)
    if expected_cuda_count is not None and len(cuda_states) != expected_cuda_count:
        raise ValueError("rng_state.cuda count conflicts with run contract")
    for index, state in enumerate(cuda_states):
        if (
            not isinstance(state, torch.Tensor)
            or state.device.type != "cpu"
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
            or state.numel() != expected_torch_bytes
        ):
            raise ValueError(f"rng_state.cuda[{index}] has an invalid CPU uint8 state")


_REQUIRED_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "run_kind",
        "comparable",
        "total_epochs",
        "architecture_schema_sha256",
        "train_manifest_sha256",
        "test_manifest_sha256",
        "final_selection_manifest_sha256",
        "source_revision",
        "gt_validity_policy",
        "runtime_versions",
        "config_snapshot",
    }
)


def _validate_run_contract(value: object) -> Mapping[str, object]:
    contract = _as_mapping(value, "run_contract")
    missing = sorted(_REQUIRED_CONTRACT_KEYS - set(contract))
    if missing:
        raise ValueError(f"run_contract is missing required fields: {', '.join(missing)}")
    if contract["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("run_contract schema_version is invalid")
    if contract["artifact_kind"] != _CONTRACT_ARTIFACT:
        raise ValueError("run_contract artifact_kind is invalid")
    if contract["run_kind"] not in {"experiment", "smoke"}:
        raise ValueError("run_contract run_kind is invalid")
    if not isinstance(contract["comparable"], bool):
        raise ValueError("run_contract comparable must be boolean")
    if contract["run_kind"] == "smoke" and contract["comparable"]:
        raise ValueError("smoke run_contract cannot be comparable")
    total_epochs = contract["total_epochs"]
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int) or total_epochs <= 0:
        raise ValueError("run_contract total_epochs must be positive")
    for key in (
        "architecture_schema_sha256",
        "train_manifest_sha256",
        "test_manifest_sha256",
        "final_selection_manifest_sha256",
        "source_revision",
        "gt_validity_policy",
    ):
        if not isinstance(contract[key], str) or not contract[key]:
            raise ValueError(f"run_contract {key} must be a non-empty string")
    if not isinstance(contract["runtime_versions"], Mapping):
        raise ValueError("run_contract runtime_versions must be a mapping")
    if not isinstance(contract["config_snapshot"], Mapping):
        raise ValueError("run_contract config_snapshot must be a mapping")
    return contract


def _fresh_preflight_payload(preflight: StartupCheckpointPreflight) -> Mapping[str, object]:
    """Reparse private preflight bytes into fresh CPU objects at load time."""

    if preflight._raw_bytes is None:
        raise ValueError("checkpoint preflight has no private byte snapshot")
    digest = sha256_bytes(preflight._raw_bytes)
    if preflight.sha256 != digest:
        raise ValueError("checkpoint preflight byte snapshot digest changed")
    payload = _load_torch_bytes(preflight._raw_bytes, label="private training checkpoint snapshot")
    return _as_mapping(payload, "checkpoint preflight payload")

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
    gt_validity_policy: str | None = None,
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
    # These fields are owned by this module and cannot be overridden by a
    # caller-provided base mapping.
    contract["schema_version"] = _CHECKPOINT_SCHEMA_VERSION
    contract["artifact_kind"] = _CONTRACT_ARTIFACT
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
    if gt_validity_policy is not None:
        contract["gt_validity_policy"] = str(gt_validity_policy)
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
        return StartupCheckpointPreflight(
            "cold_start", None, None, None, MappingProxyType({}), None
        )
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
            raw,
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
    optimizer_state = payload_mapping["optimizer_state_dict"]
    scheduler_state = payload_mapping["scheduler_state_dict"]
    _validate_optimizer_state(optimizer_state)
    _validate_scheduler_state(scheduler_state)
    progress = _validate_progress(payload_mapping["progress"])
    best = _validate_best(payload_mapping["best_metrics"])
    contract = _validate_run_contract(payload_mapping["run_contract"])
    if progress.completed_epoch > int(contract["total_epochs"]):
        raise ValueError(
            "progress.completed_epoch exceeds run_contract.total_epochs"
        )
    _validate_scheduler_against_contract(scheduler_state, optimizer_state, contract)  # type: ignore[arg-type]
    _validate_rng_state(payload_mapping["rng_state"], contract)
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
        raw,
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
    contract = _validate_run_contract(contract)
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
    _validate_optimizer_state(payload["optimizer_state_dict"])
    _validate_scheduler_state(payload["scheduler_state_dict"])
    _validate_scheduler_against_contract(
        payload["scheduler_state_dict"],  # type: ignore[arg-type]
        payload["optimizer_state_dict"],  # type: ignore[arg-type]
        contract,
    )
    _validate_rng_state(payload["rng_state"], contract)
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
    fresh_mapping = _fresh_preflight_payload(preflight)
    if "state_dict" in fresh_mapping:
        state = _as_mapping(fresh_mapping["state_dict"], "initial checkpoint state_dict")
    else:
        state = fresh_mapping
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
    payload = _fresh_preflight_payload(preflight)
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
    # Publish the two files through one staged bundle.  If either replacement
    # fails, the bundle helper restores both previous files (or removes both
    # newly-created files), so a raw weight can never appear without its
    # provenance sidecar.
    publish_epoch_artifacts(
        weights_path.parent,
        artifacts={weights_path.name: raw, sidecar.name: sidecar_raw},
    )
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
        # The public bundle is committed after the replacement loop and
        # directory fsync.  Backup cleanup is post-commit housekeeping: if a
        # backup cannot be removed, retain it as a hidden rollback copy and do
        # not enter the rollback path (which could no longer restore a backup
        # that was already deleted).
        for name, backup in list(backups.items()):
            try:
                os.unlink(backup, dir_fd=directory_fd)
            except OSError:
                continue
            backups.pop(name, None)
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


def _safe_relative_artifact_parts(name: str) -> tuple[str, ...]:
    """Validate a safe relative artifact path without traversal."""

    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError(f"artifact path must be safe and relative: {name!r}")
    parts = tuple(name.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"artifact path must be safe and relative: {name!r}")
    for part in parts:
        _safe_artifact_name(part)
    return parts


def _open_tree_child_directory(
    parent_fd: int,
    parent_identity: tuple[int, int],
    name: str,
    *,
    label: str,
) -> tuple[int, tuple[int, int]]:
    """Open/create one no-follow child while retaining its parent FD."""

    info = os.fstat(parent_fd)
    if (int(info.st_dev), int(info.st_ino)) != parent_identity:
        raise ValueError(f"{label} parent directory was replaced")
    try:
        child_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise ValueError(f"{label} appeared during secure creation") from exc
        child_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} cannot be inspected") from exc
    if stat.S_ISLNK(child_info.st_mode) or not stat.S_ISDIR(child_info.st_mode):
        raise ValueError(f"{label} must be a real directory, not a symlink")
    expected = (int(child_info.st_dev), int(child_info.st_ino))
    try:
        child_fd = os.open(name, _secure_directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened") from exc
    try:
        actual_info = os.fstat(child_fd)
        actual = (int(actual_info.st_dev), int(actual_info.st_ino))
        if actual != expected:
            raise ValueError(f"{label} was replaced while opening")
        return child_fd, actual
    except Exception:
        try:
            os.close(child_fd)
        except OSError:
            pass
        raise


def publish_tree_artifacts(
    run_dir: str | Path,
    *,
    artifacts: Mapping[str, bytes],
) -> Mapping[str, Path]:
    """Publish one rollback-safe bundle spanning nested artifact paths."""

    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("tree artifact bundle must not be empty")
    merged: dict[str, bytes] = {}
    parts_by_name: dict[str, tuple[str, ...]] = {}
    for raw_name, payload in artifacts.items():
        parts = _safe_relative_artifact_parts(raw_name)
        normalized = "/".join(parts)
        if normalized in merged:
            raise ValueError(f"duplicate tree artifact path: {normalized}")
        if not isinstance(payload, bytes):
            raise TypeError(f"artifact {normalized} payload must be bytes")
        _validate_staged_artifact(parts[-1], payload)
        merged[normalized] = payload
        parts_by_name[normalized] = parts

    destination = Path(run_dir)
    ensure_real_directory(destination, label="tree artifact directory")
    root_fd: int | None = None
    handles: dict[tuple[str, ...], tuple[int, tuple[int, int], Path]] = {}
    staged: dict[str, tuple[int, str]] = {}
    backups: dict[str, tuple[int, str]] = {}
    old_exists: dict[str, bool] = {}
    replaced: list[str] = []

    def get_handle(parts: tuple[str, ...]) -> tuple[int, tuple[int, int], Path]:
        if parts in handles:
            return handles[parts]
        if not parts:
            if root_fd is None:
                raise ValueError("tree artifact root is not open")
            path_identity = directory_identity(destination, label="tree artifact directory")
            identity = (int(path_identity["dev"]), int(path_identity["ino"]))
            handle = (root_fd, (int(identity[0]), int(identity[1])), destination.resolve())
            handles[parts] = handle
            return handle
        parent_fd, parent_identity, _parent_path = get_handle(parts[:-1])
        child_fd, child_identity = _open_tree_child_directory(
            parent_fd,
            parent_identity,
            parts[-1],
            label=f"tree artifact directory {'/'.join(parts)}",
        )
        handle = (child_fd, child_identity, destination / Path(*parts))
        handles[parts] = handle
        return handle

    try:
        root_fd, root_opened, root_path = open_or_create_directory(
            destination,
            label="tree artifact directory",
        )
        root_identity = (int(root_opened["dev"]), int(root_opened["ino"]))
        handles[()] = (root_fd, root_identity, root_path)
        for normalized in sorted(merged):
            get_handle(parts_by_name[normalized][:-1])

        for normalized in sorted(merged):
            parts = parts_by_name[normalized]
            fd = handles[parts[:-1]][0]
            staged[normalized] = (fd, _stage_file(fd, parts[-1], merged[normalized]))

        for normalized in sorted(merged):
            parts = parts_by_name[normalized]
            fd = handles[parts[:-1]][0]
            name = parts[-1]
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                old_exists[normalized] = False
                continue
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValueError(f"tree artifact destination must be a regular file: {normalized}")
            old_exists[normalized] = True
            backup_name = f".{name}.{secrets.token_hex(12)}.bak"
            os.link(name, backup_name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
            backups[normalized] = (fd, backup_name)

        for _parts, (fd, identity, path) in handles.items():
            current = os.fstat(fd)
            if (int(current.st_dev), int(current.st_ino)) != identity:
                raise ValueError(f"tree artifact directory was replaced: {path}")
            assert_directory_path_identity(
                path,
                {"dev": identity[0], "ino": identity[1]},
                label=f"tree artifact directory {path}",
            )

        for normalized in sorted(merged):
            parts = parts_by_name[normalized]
            fd, temporary = staged[normalized]
            os.replace(temporary, parts[-1], src_dir_fd=fd, dst_dir_fd=fd)
            staged[normalized] = (fd, "")
            replaced.append(normalized)
        for fd, _identity, _path in handles.values():
            os.fsync(fd)
        for _parts, (_fd, identity, path) in handles.items():
            assert_directory_path_identity(
                path,
                {"dev": identity[0], "ino": identity[1]},
                label=f"tree artifact directory {path}",
            )

        # Cleanup is post-commit.  A failed unlink leaves a hidden backup but
        # never re-enters rollback after the public bundle is committed.
        for normalized, (fd, backup_name) in list(backups.items()):
            try:
                os.unlink(backup_name, dir_fd=fd)
            except OSError:
                continue
            backups.pop(normalized, None)
        return {
            normalized: destination / Path(*parts_by_name[normalized])
            for normalized in merged
        }
    except Exception as exc:
        for normalized in reversed(replaced):
            parts = parts_by_name[normalized]
            fd = handles[parts[:-1]][0]
            try:
                if old_exists.get(normalized, False) and normalized in backups:
                    os.replace(backups[normalized][1], parts[-1], src_dir_fd=fd, dst_dir_fd=fd)
                    backups.pop(normalized, None)
                elif not old_exists.get(normalized, False):
                    os.unlink(parts[-1], dir_fd=fd)
            except OSError:
                pass
        for normalized, (fd, backup_name) in list(backups.items()):
            try:
                os.unlink(backup_name, dir_fd=fd)
            except OSError:
                pass
        for fd, temporary in staged.values():
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=fd)
                except OSError:
                    pass
        for fd, _identity, _path in handles.values():
            try:
                os.fsync(fd)
            except OSError:
                pass
        if isinstance(exc, (TypeError, ValueError)):
            raise
        raise ValueError("failed to publish tree artifact bundle; previous bundle restored") from exc
    finally:
        closed: set[int] = set()
        for fd, _identity, _path in reversed(tuple(handles.values())):
            if fd in closed:
                continue
            closed.add(fd)
            try:
                os.close(fd)
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
    directories = [root, root / "checkpoints", root / "exports"]
    for directory in directories:
        try:
            descriptor, _identity, absolute = _open_existing_directory_path(
                directory,
                label="artifact retention directory",
            )
        except FileNotFoundError:
            continue
        try:
            for name in os.listdir(descriptor):
                try:
                    info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    continue
                match = _EPOCH_NAME.search(Path(name).stem)
                if match is None:
                    continue
                epoch = int(match.group(1))
                if epoch in keep or (latest_epoch is not None and epoch == latest_epoch):
                    continue
                os.unlink(name, dir_fd=descriptor)
                removed.append(absolute / name)
            os.fsync(descriptor)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
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
    "publish_tree_artifacts",
    "restore_rng_state",
    "save_resume_checkpoint",
    "schema_fingerprint",
]
