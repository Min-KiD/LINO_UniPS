"""Deterministic source-resolution angular metrics and paired scoring.

The comparison boundary intentionally has no model, CUDA, network, or inference
dependencies.  It reads only signed EXR predictions and the original source
normal recorded by the canonical rich manifest.
"""

from __future__ import annotations

import csv
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .config import SdmExrInferenceConfig, load_sdm_exr_config
from .exr_io import read_signed_normal_exr_bytes, sha256_file
from .manifest import (
    DatasetManifest,
    ObjectRecord,
    build_dataset_manifest,
    load_dataset_manifest,
)
from .normal_contract import decode_ground_truth_normal
from .provenance import (
    atomic_create_json,
    directory_identity,
    python_runtime_version,
    read_json_mapping,
    same_path,
    same_directory_identity,
    sha256_bytes,
    sha256_json,
    config_runtime_fingerprint,
    file_identity,
    lino_preprocessing_snapshot,
)
from .sdm_view import build_sdm_command, validate_sdm_view


_NORMAL_EPS = 1.0e-12
_METRIC_NAMES = (
    "mae",
    "median",
    "p90",
    "p95",
    "accuracy_11_25",
    "accuracy_30",
    "accuracy_45",
)
_LINEAR_METRIC_NAMES = (
    "mae",
    "accuracy_11_25",
    "accuracy_30",
    "accuracy_45",
)


def _normal_array(value: Any, *, label: str) -> np.ndarray:
    """Validate a finite numeric source-resolution normal array."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric H,W,3 array") from exc
    if (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"{label} must be a numeric H,W,3 array")
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must have nonempty shape H,W,3; got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    # The contract is evaluated in float64.  A finite extended-precision input
    # can overflow while being converted; reject it before any metric is made.
    try:
        converted = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} cannot be represented in float64") from exc
    if not np.isfinite(converted).all():
        raise ValueError(f"{label} cannot be represented as finite float64")
    return array


def _boolean_mask(value: Any, *, shape: tuple[int, int]) -> np.ndarray:
    try:
        mask = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("mask must be a boolean H,W array") from exc
    if mask.dtype != np.dtype(bool) or mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"mask must be a boolean array with shape {shape}")
    return mask


def _stable_components(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values, per-vector max-abs scale, and scaled vector norm."""

    values = np.asarray(array, dtype=np.float64)
    scale = np.max(np.abs(values), axis=2, keepdims=True)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    scaled = values / safe_scale
    scaled_norm = np.sqrt(np.sum(scaled * scaled, axis=2, keepdims=True))
    return values, scale, scaled_norm


def _stable_normal_validity(array: np.ndarray) -> np.ndarray:
    _, scale, scaled_norm = _stable_components(array)
    # Any scale above eps is necessarily non-negligible without multiplying
    # by the norm (which could overflow for finite 1e200-scale vectors).
    valid = scale[..., 0] > _NORMAL_EPS
    small = (scale[..., 0] > 0.0) & (scale[..., 0] <= _NORMAL_EPS)
    small_norm = scale[..., 0] * scaled_norm[..., 0]
    valid |= small & (small_norm > _NORMAL_EPS)
    return np.asarray(valid, dtype=bool)


def _stable_normalize(array: np.ndarray) -> np.ndarray:
    """Normalize vectors stably while retaining denominator clipping semantics."""

    values, scale, scaled_norm = _stable_components(array)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    scaled = values / safe_scale
    regular = (scale > 0.0) & (scaled_norm > 0.0)
    # For vectors whose true norm is below eps, denominator clipping means
    # values / eps rather than a unit vector.  This branch is safe because all
    # components are at most eps in magnitude when scale <= eps.
    tiny_norm = (scale > 0.0) & (scale <= _NORMAL_EPS)
    tiny_norm &= (scale * scaled_norm < _NORMAL_EPS)
    denominator = np.where(regular, scaled_norm, 1.0)
    normalized = np.divide(scaled, denominator, out=np.zeros_like(values), where=regular)
    normalized = np.where(tiny_norm, values / _NORMAL_EPS, normalized)
    if not np.isfinite(normalized).all():
        raise ValueError("normalization produced non-finite values")
    return np.asarray(normalized, dtype=np.float64)


def normal_validity_mask(normal: Any) -> np.ndarray:
    """Return support from finite, non-negligible source GT vectors only.

    This mask deliberately has no knowledge of either model's input mask.  A
    caller may choose another explicit boolean mask for :func:`angular_metrics`,
    but paired scoring always uses this exact source-GT-derived result.
    """

    array = _normal_array(normal, label="normal")
    return _stable_normal_validity(array)


def angular_metrics(gt: Any, pred: Any, mask: Any) -> dict[str, float | int]:
    """Compute angular error statistics over an explicit boolean support mask."""

    gt_array = _normal_array(gt, label="ground-truth normal")
    pred_array = _normal_array(pred, label="prediction normal")
    if gt_array.shape != pred_array.shape:
        raise ValueError(
            "ground-truth and prediction normal shapes must match: "
            f"{gt_array.shape} != {pred_array.shape}"
        )
    support = _boolean_mask(mask, shape=gt_array.shape[:2])
    valid_count = int(np.count_nonzero(support))
    if valid_count == 0:
        raise ValueError("angular metrics cannot be computed for empty support")

    # Clipping the denominator rather than dropping tiny vectors keeps this
    # primitive well-defined for a caller that deliberately supplies them;
    # source scoring excludes tiny GT vectors through normal_validity_mask.
    gt_unit = _stable_normalize(gt_array)
    pred_unit = _stable_normalize(pred_array)
    cosine = np.sum(gt_unit * pred_unit, axis=2)
    cosine = np.clip(cosine, -1.0, 1.0)
    errors = np.degrees(np.arccos(cosine[support]))
    if not np.isfinite(errors).all():
        raise ValueError("angular metrics produced non-finite results")

    return {
        "mae": float(np.mean(errors)),
        "median": float(np.percentile(errors, 50.0)),
        "p90": float(np.percentile(errors, 90.0)),
        "p95": float(np.percentile(errors, 95.0)),
        "accuracy_11_25": float(np.mean(errors < 11.25)),
        "accuracy_30": float(np.mean(errors < 30.0)),
        "accuracy_45": float(np.mean(errors < 45.0)),
        "valid_pixel_count": valid_count,
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


def _atomic_csv_write(path: str | Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> Path:
    destination = path if isinstance(path, Path) else Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
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


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _selection_paths(config: SdmExrInferenceConfig) -> tuple[Path, Path]:
    source = Path(config.effective_selection_manifest_path)
    if config.light_selection == "manifest" and config.selection_manifest is not None:
        return source, Path(config.policy_root) / "selected_lights.json"
    return source, source


def _current_provenance(
    config: SdmExrInferenceConfig,
) -> tuple[DatasetManifest, dict[str, str], tuple[Path, Path]]:
    input_manifest = Path(config.input_manifest_path)
    try:
        if input_manifest.is_symlink() or not input_manifest.is_file():
            raise ValueError("rich manifest must be a regular non-symlink file")
        manifest = load_dataset_manifest(input_manifest)
        input_digest = sha256_file(input_manifest)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid current rich manifest: {input_manifest}") from exc

    if Path(manifest.data_root).resolve(strict=False) != Path(config.data_root).resolve(
        strict=False
    ):
        raise ValueError("rich manifest data_root does not match comparison config")
    if manifest.seed != config.seed or manifest.max_image_num != config.max_image_num:
        raise ValueError("rich manifest seed/max_image_num does not match comparison config")
    try:
        rebuilt = build_dataset_manifest(config)
    except (OSError, ValueError) as exc:
        raise ValueError("failed to rebuild current source manifest") from exc
    if rebuilt != manifest:
        raise ValueError("current source files do not match the prepared rich manifest")

    source_selection, effective_selection = _selection_paths(config)
    try:
        source_digest = sha256_file(source_selection)
        effective_digest = sha256_file(effective_selection)
    except OSError as exc:
        raise ValueError("current selection manifest is missing or unreadable") from exc
    return (
        manifest,
        {
            "input_manifest_sha256": input_digest,
            "selection_manifest_sha256": source_digest,
            "effective_selection_manifest_sha256": effective_digest,
        },
        (source_selection, effective_selection),
    )


def _request_output_path(
    config: SdmExrInferenceConfig,
    request: Mapping[str, Any],
) -> Path:
    request_id = request.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id
        or Path(request_id).name != request_id
        or request_id in {".", ".."}
        or "\\" in request_id
    ):
        raise ValueError("SDM run request is missing a valid request_id")
    output_raw = request.get("output_path")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise ValueError("SDM run request is missing output_path")
    output = Path(output_raw)
    base = Path(config.sdm_output_dir).resolve(strict=False)
    try:
        resolved = output.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"SDM request output directory is missing: {output}") from exc
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"SDM request output must be a real directory: {output}")
    if resolved.parent != base or resolved.name != request_id:
        raise ValueError("SDM request output_path is not its fresh per-request directory")
    configured = request.get("configured_output_path")
    if not same_path(configured, base):
        raise ValueError("SDM run request configured_output_path does not match config")
    return resolved


def _validate_request_location(
    config: SdmExrInferenceConfig,
    request_path: Path,
    request: Mapping[str, Any],
) -> tuple[Path, Path]:
    request_id = request.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id
        or Path(request_id).name != request_id
        or request_id in {".", ".."}
        or "\\" in request_id
    ):
        raise ValueError("SDM run request is missing a valid request_id")
    expected_request = (
        Path(config.policy_root).resolve(strict=False)
        / "sdm_requests"
        / f"{request_id}.json"
    )
    if request_path.resolve(strict=True) != expected_request or not same_path(
        request.get("request_path"), expected_request
    ):
        raise ValueError("SDM run request path does not match its request_id")
    completion = request.get("completion_path")
    expected_completion = (
        Path(config.policy_root).resolve(strict=False)
        / "sdm_completions"
        / f"{request_id}.json"
    )
    if not same_path(completion, expected_completion):
        raise ValueError("SDM run request completion_path does not match its request_id")
    return expected_request, expected_completion


def _validate_sdm_request(
    config: SdmExrInferenceConfig,
    request_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> tuple[
    dict[str, Any],
    bytes,
    DatasetManifest,
    dict[str, str],
    tuple[Path, Path],
    Path,
    Path,
    dict[str, object],
]:
    exact_request = Path(request_path)
    request, request_bytes = read_json_mapping(exact_request, label="SDM run request")
    if request.get("schema_version") != 2 or request.get("model") != "SDM-UniPS":
        raise ValueError("unsupported SDM run request schema or model")
    _, completion_path = _validate_request_location(config, exact_request, request)

    configured_path_raw = request.get("config_path")
    if not isinstance(configured_path_raw, str) or not configured_path_raw.strip():
        raise ValueError("SDM run request is missing config_path")
    configured_path = Path(configured_path_raw)
    if config_path is not None and not same_path(config_path, configured_path):
        raise ValueError("explicit config path does not match SDM run request")
    if configured_path.is_symlink() or not configured_path.is_file():
        raise ValueError("comparison config must be a regular non-symlink file")
    if request.get("config_sha256") != sha256_file(configured_path):
        raise ValueError("comparison config digest changed after SDM preparation")
    if load_sdm_exr_config(configured_path) != config:
        raise ValueError("comparison config object does not match requested config")

    manifest, digests, selection_paths = _current_provenance(config)
    view_attestation = validate_sdm_view(config, manifest)
    _validate_provenance(
        request,
        label="SDM run request",
        config=config,
        digests=digests,
        paths=selection_paths,
        required_fields=(
            "input_manifest_path",
            "selection_manifest_path",
            "effective_selection_manifest_path",
            "output_path",
        ),
    )
    output_path = _request_output_path(config, request)

    expected_view_fields = {
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
    }
    for field, expected in expected_view_fields.items():
        if request.get(field) != expected:
            raise ValueError(f"SDM run request {field} does not match current SDM view")
    if request.get("view_attestation") != view_attestation:
        raise ValueError("SDM run request view attestation does not match current SDM view")

    request_identity_specs = (
        ("sdm_output_base_identity", Path(config.sdm_output_dir), "SDM output base"),
        ("sdm_output_identity", output_path, "SDM output directory"),
        ("sdm_request_dir_identity", exact_request.parent, "SDM request directory"),
        ("sdm_completion_dir_identity", completion_path.parent, "SDM completion directory"),
    )
    for field, path, label in request_identity_specs:
        directory_identity(path, label=label)
        if field not in request:
            raise ValueError(f"SDM run request is missing {field}")
        if not same_directory_identity(path, request[field], label=label):
            raise ValueError(f"SDM run request {field} does not match current {label}")

    repo_raw = request.get("sdm_repo")
    if not isinstance(repo_raw, str) or not repo_raw.strip():
        raise ValueError("SDM run request is missing sdm_repo")
    repo = Path(repo_raw)
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError("SDM repository must remain a real directory")
    expected_main = repo / "main.py"
    expected_config = repo / "configs" / "baseline_optimized_infer.yaml"
    if not same_path(request.get("sdm_main"), expected_main):
        raise ValueError("SDM run request main path does not match its repository")
    if not same_path(request.get("sdm_config"), expected_config):
        raise ValueError("SDM run request optimized config path does not match its repository")
    if not same_path(request.get("view_path"), config.sdm_view_dir):
        raise ValueError("SDM run request view_path does not match current config")

    runtime_fields = (
        ("sdm_main", "sdm_main_sha256"),
        ("sdm_config", "sdm_config_sha256"),
        ("sdm_checkpoint", "sdm_checkpoint_sha256"),
        ("sdm_python", "sdm_python_sha256"),
    )
    for path_field, digest_field in runtime_fields:
        raw = request.get(path_field)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"SDM run request is missing {path_field}")
        runtime_path = Path(raw)
        if not runtime_path.is_file():
            raise ValueError(f"SDM runtime file is missing: {runtime_path}")
        if request.get(digest_field) != sha256_file(runtime_path):
            raise ValueError(f"SDM runtime digest changed for {path_field}")

    expected_argv = build_sdm_command(
        config,
        repo,
        Path(str(request["sdm_checkpoint"])),
        Path(str(request["sdm_python"])),
        output_dir=output_path,
    )
    if request.get("argv") != expected_argv:
        raise ValueError("SDM run request argv does not match current exact command")
    python_version = python_runtime_version(Path(str(request["sdm_python"])))
    if request.get("sdm_python_version") != python_version:
        raise ValueError("SDM Python version changed after preparation")
    expected_fingerprint_inputs: dict[str, Any] = {
        "request_id": request["request_id"],
        "config_sha256": request["config_sha256"],
        "mask_policy": config.mask_policy,
        "input_manifest_sha256": digests["input_manifest_sha256"],
        "selection_manifest_sha256": digests["selection_manifest_sha256"],
        "effective_selection_manifest_sha256": digests[
            "effective_selection_manifest_sha256"
        ],
        "sdm_main_sha256": request["sdm_main_sha256"],
        "sdm_config_sha256": request["sdm_config_sha256"],
        "sdm_checkpoint_sha256": request["sdm_checkpoint_sha256"],
        "sdm_python_sha256": request["sdm_python_sha256"],
        "sdm_python_version": python_version,
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
        "sdm_output_base_identity": request["sdm_output_base_identity"],
        "sdm_output_identity": request["sdm_output_identity"],
        "sdm_request_dir_identity": request["sdm_request_dir_identity"],
        "sdm_completion_dir_identity": request["sdm_completion_dir_identity"],
        "argv": expected_argv,
    }
    if request.get("fingerprint_inputs") != expected_fingerprint_inputs:
        raise ValueError("SDM run request fingerprint inputs do not match current run")
    expected_fingerprint = sha256_json(expected_fingerprint_inputs)
    if request.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("SDM run request fingerprint is invalid")
    return (
        request,
        request_bytes,
        manifest,
        digests,
        selection_paths,
        output_path,
        completion_path,
        view_attestation,
    )


def _same_path(left: Any, right: Path) -> bool:
    if not isinstance(left, str) or not left.strip():
        return False
    try:
        return Path(left).resolve(strict=False) == right.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def _validate_provenance(
    payload: Mapping[str, Any],
    *,
    label: str,
    config: SdmExrInferenceConfig,
    digests: Mapping[str, str],
    paths: tuple[Path, Path],
    sdm_output_dir: Path | None = None,
    required_fields: tuple[str, ...] = (),
) -> None:
    if payload.get("mask_policy") != config.mask_policy:
        raise ValueError(f"{label} mask_policy does not match current policy")
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"{label} is missing required field {field}")
    for field, expected in digests.items():
        if field not in payload:
            raise ValueError(f"{label} is missing required field {field}")
        if payload[field] != expected:
            raise ValueError(f"{label} {field} does not match current manifest digest")

    source_selection, effective_selection = paths
    for field, expected in (
        ("selection_manifest_path", source_selection),
        ("effective_selection_manifest_path", effective_selection),
    ):
        if field in payload and not _same_path(payload[field], expected):
            raise ValueError(f"{label} {field} does not match current selection manifest")
    if "input_manifest_path" in payload and not _same_path(
        payload["input_manifest_path"], Path(config.input_manifest_path)
    ):
        raise ValueError(f"{label} input_manifest_path does not match current manifest")
    if sdm_output_dir is not None and "output_path" in payload and not _same_path(
        payload["output_path"], sdm_output_dir
    ):
        raise ValueError(f"{label} output_path does not match explicit SDM output directory")


def _validate_lino_runtime_provenance(
    payload: Mapping[str, Any],
    *,
    config: SdmExrInferenceConfig,
    config_path: str | Path | None,
) -> None:
    """Require the LINO run to pair with the exact current runtime inputs."""

    required = (
        "config_path",
        "config_sha256",
        "config_runtime_fingerprint",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_identity",
        "preprocessing",
    )
    for field in required:
        if field not in payload:
            raise ValueError(f"LINO run provenance is missing required field {field}")
    if payload.get("config_runtime_fingerprint") != config_runtime_fingerprint(config):
        raise ValueError("LINO run provenance config runtime fingerprint is stale")

    if config_path is None:
        if payload.get("config_path") is not None or payload.get("config_sha256") is not None:
            raise ValueError("LINO run provenance config path must be explicit for pairing")
    else:
        expected_config_path = Path(config_path)
        if not _same_path(payload.get("config_path"), expected_config_path):
            raise ValueError("LINO run provenance config_path does not match current config")
        config_raw = _read_regular_file_once(expected_config_path, label="LINO config")
        expected_config_digest = sha256_bytes(config_raw)
        if payload.get("config_sha256") != expected_config_digest:
            raise ValueError("LINO run provenance config digest is stale")

    checkpoint = Path(config.checkpoint)
    checkpoint_identity_before = file_identity(checkpoint, label="LINO checkpoint")
    checkpoint_raw = _read_regular_file_once(checkpoint, label="LINO checkpoint")
    expected_checkpoint_digest = sha256_bytes(checkpoint_raw)
    checkpoint_identity_after = file_identity(checkpoint, label="LINO checkpoint")
    if checkpoint_identity_after != checkpoint_identity_before:
        raise ValueError("LINO checkpoint was replaced while validating provenance")
    if payload.get("checkpoint_sha256") != expected_checkpoint_digest:
        raise ValueError("LINO run provenance checkpoint digest is stale")
    if not _same_path(payload.get("checkpoint_path"), checkpoint):
        raise ValueError("LINO run provenance checkpoint_path does not match current checkpoint")
    if payload.get("checkpoint_identity") != checkpoint_identity_after:
        raise ValueError("LINO run provenance checkpoint identity is stale")
    if payload.get("preprocessing") != lino_preprocessing_snapshot(config):
        raise ValueError("LINO run provenance preprocessing knobs are stale")


def _expected_lino_paths(
    config: SdmExrInferenceConfig, manifest: DatasetManifest
) -> dict[str, Path]:
    root = Path(config.lino_output_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"LINO output directory is missing or invalid: {root}")
    expected_names = {record.name for record in manifest.objects}
    entries = tuple(root.iterdir())
    actual_names = {entry.name for entry in entries if entry.is_dir() and not entry.is_symlink()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        detail = []
        if missing:
            detail.append(f"missing object prediction(s): {', '.join(missing)}")
        if extra:
            detail.append(f"extra object prediction(s): {', '.join(extra)}")
        raise ValueError("LINO prediction object set mismatch: " + "; ".join(detail))
    unexpected_root = [entry.name for entry in entries if entry.name != "run.json" and entry.name not in expected_names]
    if unexpected_root:
        raise ValueError(f"extra LINO prediction artifact(s): {', '.join(sorted(unexpected_root))}")

    paths: dict[str, Path] = {}
    for record in manifest.objects:
        object_dir = root / record.name
        children = tuple(object_dir.iterdir())
        prediction = object_dir / "normal_pred.exr"
        if not prediction.is_file() or prediction.is_symlink():
            raise ValueError(f"missing LINO prediction for {record.name}: {prediction}")
        allowed = {"normal_pred.exr", "normal_pred.png"}
        extras = [child.name for child in children if child.name not in allowed]
        if extras:
            raise ValueError(
                f"extra LINO prediction artifact(s) for {record.name}: {', '.join(sorted(extras))}"
            )
        paths[record.name] = prediction
    return paths


def _expected_sdm_paths(sdm_output_dir: Path, manifest: DatasetManifest) -> dict[str, Path]:
    root = Path(sdm_output_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"SDM output directory is missing or invalid: {root}")
    expected: dict[str, Path] = {}
    generated_by_name: dict[str, str] = {}
    for record in manifest.objects:
        prediction_name = f"{Path(record.name).stem}_pred.exr"
        previous = generated_by_name.get(prediction_name)
        if previous is not None and previous != record.name:
            raise ValueError(
                "generated SDM prediction basename collision: "
                f"{previous!r} and {record.name!r} both map to {prediction_name!r}"
            )
        generated_by_name[prediction_name] = record.name
        expected[record.name] = root / prediction_name
    expected_names = {path.name for path in expected.values()}
    expected_png_names = {path.with_suffix(".png").name for path in expected.values()}
    allowed_names = expected_names | expected_png_names
    extras = [entry.name for entry in root.iterdir() if entry.name not in allowed_names]
    if extras:
        raise ValueError(f"extra SDM prediction artifact(s): {', '.join(sorted(extras))}")
    for png_name in expected_png_names:
        png_path = root / png_name
        if png_path.exists() and (not png_path.is_file() or png_path.is_symlink()):
            raise ValueError(f"invalid SDM preview artifact: {png_path}")
    for object_name, path in expected.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing SDM prediction for {object_name}: {path}")
    return expected


def _read_regular_file_once(path: Path, *, label: str) -> bytes:
    """Read one pinned, non-symlink regular file into immutable bytes."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"secure {label} reads require O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(str(path), flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {label}: {path}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _secure_output_directory_fd(path: Path, *, label: str) -> tuple[int, dict[str, int]]:
    """Open and pin one output directory without following its final symlink."""

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"secure {label} reads require O_DIRECTORY/O_NOFOLLOW")
    expected = directory_identity(path, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(str(path), flags)
        info = os.fstat(descriptor)
        actual = {"dev": int(info.st_dev), "ino": int(info.st_ino)}
        if actual != expected:
            os.close(descriptor)
            raise ValueError(f"{label} was replaced while opening: {path}")
        return descriptor, expected
    except OSError as exc:
        raise ValueError(f"failed to securely open {label}: {path}") from exc


def _read_regular_fd_once(
    directory_fd: int,
    basename: str,
    *,
    label: str,
) -> tuple[bytes, dict[str, int]]:
    """Read one regular output artifact through a pinned directory FD."""

    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(basename, flags, dir_fd=directory_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {basename}")
        identity = {"dev": int(info.st_dev), "ino": int(info.st_ino)}
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks), identity
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {basename}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {label}: {basename}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _read_prediction_artifact(path: Path, *, label: str) -> dict[str, Any]:
    """Attest and decode one prediction from one immutable byte snapshot."""

    raw = _read_regular_file_once(path, label=label)
    digest = sha256_bytes(raw)
    array = read_signed_normal_exr_bytes(raw, label=label)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} path disappeared after read: {path}") from exc
    return {"path": str(resolved), "sha256": digest, "array": array}


def _validated_sdm_predictions(
    output_dir: Path,
    manifest: DatasetManifest,
) -> tuple[dict[str, Path], list[dict[str, Any]], dict[str, np.ndarray]]:
    root = Path(output_dir)
    generated_by_name: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for record in manifest.objects:
        prediction_name = f"{Path(record.name).stem}_pred.exr"
        previous = generated_by_name.get(prediction_name)
        if previous is not None and previous != record.name:
            raise ValueError(
                "generated SDM prediction basename collision: "
                f"{previous!r} and {record.name!r} both map to {prediction_name!r}"
            )
        generated_by_name[prediction_name] = record.name
        paths[record.name] = root / prediction_name

    expected_names = {path.name for path in paths.values()}
    expected_png_names = {path.with_suffix(".png").name for path in paths.values()}
    allowed_names = expected_names | expected_png_names
    directory_fd, pinned_identity = _secure_output_directory_fd(root, label="SDM output directory")
    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    try:
        try:
            actual_names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise ValueError(f"failed to inspect SDM output directory: {root}") from exc
        extras = actual_names - allowed_names
        if extras:
            raise ValueError(f"extra SDM prediction artifact(s): {', '.join(sorted(extras))}")
        for png_name in expected_png_names:
            try:
                info = os.stat(png_name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(f"failed to inspect SDM preview artifact: {root / png_name}") from exc
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"invalid SDM preview artifact: {root / png_name}")
        for record in manifest.objects:
            prediction_name = paths[record.name].name
            if prediction_name not in actual_names:
                raise ValueError(f"missing SDM prediction for {record.name}: {paths[record.name]}")
            raw, prediction_identity = _read_regular_fd_once(
                directory_fd,
                prediction_name,
                label=f"SDM prediction for {record.name}",
            )
            digest = sha256_bytes(raw)
            prediction = read_signed_normal_exr_bytes(
                raw,
                label=f"SDM prediction for {record.name}",
            )
            expected_shape = (record.height, record.width, 3)
            if prediction.shape != expected_shape:
                raise ValueError(
                    f"SDM prediction geometry mismatch for {record.name}: "
                    f"{prediction.shape} != {expected_shape}"
                )
            try:
                path_info = os.stat(
                    prediction_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ValueError(
                    f"SDM prediction path changed during validation: {paths[record.name]}"
                ) from exc
            current_prediction_identity = {
                "dev": int(path_info.st_dev),
                "ino": int(path_info.st_ino),
            }
            if current_prediction_identity != prediction_identity:
                raise ValueError(
                    f"SDM prediction path changed during validation: {paths[record.name]}"
                )
            records.append(
                {
                    "object_name": record.name,
                    "output_path": str(paths[record.name].resolve(strict=True)),
                    "output_sha256": digest,
                    "output_identity": prediction_identity,
                    "source_geometry": {
                        "height": int(record.height),
                        "width": int(record.width),
                    },
                }
            )
            arrays[record.name] = prediction

        info = os.fstat(directory_fd)
        if {"dev": int(info.st_dev), "ino": int(info.st_ino)} != pinned_identity:
            raise ValueError(f"SDM output directory was replaced during validation: {root}")
        if not same_directory_identity(root, pinned_identity, label="SDM output directory"):
            raise ValueError(f"SDM output directory was replaced during validation: {root}")
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass
    return paths, records, arrays


def finalize_sdm_run(
    config: SdmExrInferenceConfig,
    request_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and seal one manually executed SDM prediction directory."""

    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    (
        request,
        request_bytes,
        manifest,
        digests,
        selection_paths,
        output_dir,
        completion_path,
        view_attestation,
    ) = _validate_sdm_request(config, request_path, config_path=config_path)
    _, prediction_records, _ = _validated_sdm_predictions(output_dir, manifest)
    if not same_directory_identity(
        output_dir,
        request["sdm_output_identity"],
        label="SDM output directory",
    ):
        raise ValueError(f"SDM output directory was replaced before completion publication: {output_dir}")
    completion: dict[str, Any] = {
        "schema_version": 1,
        "model": "SDM-UniPS",
        "completion_id": secrets.token_hex(16),
        "request_id": request["request_id"],
        "request_path": str(Path(request_path).resolve(strict=True)),
        "request_sha256": sha256_bytes(request_bytes),
        "run_fingerprint": request["run_fingerprint"],
        "output_path": str(output_dir),
        "config_sha256": request["config_sha256"],
        "mask_policy": config.mask_policy,
        "view_attestation": view_attestation,
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
        "sdm_output_base_identity": request["sdm_output_base_identity"],
        "sdm_output_identity": request["sdm_output_identity"],
        "sdm_request_dir_identity": request["sdm_request_dir_identity"],
        "sdm_completion_dir_identity": request["sdm_completion_dir_identity"],
        "input_manifest_path": str(config.input_manifest_path),
        "input_manifest_sha256": digests["input_manifest_sha256"],
        "selection_manifest_path": str(selection_paths[0]),
        "selection_manifest_sha256": digests["selection_manifest_sha256"],
        "effective_selection_manifest_path": str(selection_paths[1]),
        "effective_selection_manifest_sha256": digests[
            "effective_selection_manifest_sha256"
        ],
        "predictions": prediction_records,
        "prediction_set_sha256": sha256_json(prediction_records),
    }
    atomic_create_json(completion_path, completion)
    completion["completion_path"] = str(completion_path)
    return completion


def _validate_sdm_completion(
    config: SdmExrInferenceConfig,
    request: Mapping[str, Any],
    request_bytes: bytes,
    completion_path: Path,
    manifest: DatasetManifest,
    digests: Mapping[str, str],
    selection_paths: tuple[Path, Path],
    output_dir: Path,
    request_path: Path,
) -> tuple[dict[str, Path], dict[str, Any], bytes, dict[str, np.ndarray]]:
    completion, completion_bytes = read_json_mapping(
        completion_path,
        label="SDM completion record",
    )
    if completion.get("schema_version") != 1 or completion.get("model") != "SDM-UniPS":
        raise ValueError("unsupported SDM completion schema or model")
    if not isinstance(completion.get("completion_id"), str) or not completion[
        "completion_id"
    ]:
        raise ValueError("SDM completion record is missing completion_id")
    if completion.get("request_id") != request.get("request_id"):
        raise ValueError("SDM completion request_id does not match request")
    if not same_path(completion.get("request_path"), request_path):
        raise ValueError("SDM completion request_path does not match request")
    if completion.get("request_sha256") != sha256_bytes(request_bytes):
        raise ValueError("SDM completion does not match the exact request bytes")
    if completion.get("run_fingerprint") != request.get("run_fingerprint"):
        raise ValueError("SDM completion run fingerprint does not match request")
    if completion.get("config_sha256") != request.get("config_sha256"):
        raise ValueError("SDM completion config digest does not match request")
    for field in (
        "view_attestation",
        "view_tree_sha256",
        "view_root_identity",
        "view_object_identities",
        "sdm_output_base_identity",
        "sdm_output_identity",
        "sdm_request_dir_identity",
        "sdm_completion_dir_identity",
    ):
        if field not in completion:
            raise ValueError(f"SDM completion record is missing {field}")
        if completion.get(field) != request.get(field):
            raise ValueError(f"SDM completion {field} does not match request")
    completion_identity_specs = (
        ("sdm_output_base_identity", Path(config.sdm_output_dir), "SDM output base"),
        ("sdm_output_identity", output_dir, "SDM output directory"),
        ("sdm_request_dir_identity", request_path.parent, "SDM request directory"),
        ("sdm_completion_dir_identity", completion_path.parent, "SDM completion directory"),
    )
    for field, path, label in completion_identity_specs:
        if not same_directory_identity(path, completion[field], label=label):
            raise ValueError(f"SDM completion {field} does not match current {label}")
    _validate_provenance(
        completion,
        label="SDM completion record",
        config=config,
        digests=digests,
        paths=selection_paths,
        sdm_output_dir=output_dir,
        required_fields=(
            "input_manifest_path",
            "selection_manifest_path",
            "effective_selection_manifest_path",
            "output_path",
        ),
    )
    sdm_paths, current_predictions, sdm_arrays = _validated_sdm_predictions(output_dir, manifest)
    if completion.get("predictions") != current_predictions:
        raise ValueError("SDM prediction digest or geometry changed after finalization")
    if completion.get("prediction_set_sha256") != sha256_json(current_predictions):
        raise ValueError("SDM completion prediction-set digest is invalid")
    return sdm_paths, completion, completion_bytes, sdm_arrays


def load_source_gt(
    config: SdmExrInferenceConfig, record: ObjectRecord
) -> tuple[np.ndarray, Path]:
    root = Path(config.data_root).resolve(strict=False)
    object_dir = (root / record.relative_dir).resolve(strict=False)
    source = (object_dir / record.normal_file).resolve(strict=False)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source GT path escapes data_root for {record.name}") from exc
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"source GT normal is missing for {record.name}: {source}")
    raw = _read_regular_file_once(source, label=f"source GT normal for {record.name}")
    if sha256_bytes(raw) != record.normal_sha256:
        raise ValueError(f"source GT digest mismatch for {record.name}")
    encoded_gt = read_signed_normal_exr_bytes(
        raw,
        label=f"source GT normal for {record.name}",
    )
    gt = decode_ground_truth_normal(
        encoded_gt,
        config.normal_encoding,
        label=f"source GT normal for {record.name}",
    )
    if gt.shape != (record.height, record.width, 3):
        raise ValueError(
            f"source GT geometry mismatch for {record.name}: {gt.shape} != "
            f"{(record.height, record.width, 3)}"
        )
    return gt, source


def _validate_source_geometry(value: Any, record: ObjectRecord) -> None:
    """Require the exact Task 4 JSON source-geometry contract."""

    if not isinstance(value, Mapping) or set(value) != {"height", "width"}:
        raise ValueError(f"LINO provenance source geometry mismatch for {record.name}")
    height = value["height"]
    width = value["width"]
    if (
        type(height) is not int
        or type(width) is not int
        or height <= 0
        or width <= 0
        or height != record.height
        or width != record.width
    ):
        raise ValueError(f"LINO provenance source geometry mismatch for {record.name}")


def _aggregate(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate an empty comparison")
    counts = [int(row[f"{prefix}_valid_pixel_count"]) for row in rows]
    total = sum(counts)
    if total <= 0:
        raise ValueError("cannot aggregate comparison with no valid pixels")
    means: dict[str, float] = {}
    weighted: dict[str, float] = {}
    for metric_name in _METRIC_NAMES:
        values = [float(row[f"{prefix}_{metric_name}"]) for row in rows]
        means[metric_name] = float(sum(values) / len(values))
        weighted[metric_name] = float(
            sum(value * count for value, count in zip(values, counts)) / total
        )
    return {**means, **{f"pixel_weighted_{name}": value for name, value in weighted.items()}}


def score_lino_and_sdm(
    config: SdmExrInferenceConfig,
    request_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score one finalized SDM request against its paired LINO run."""

    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    explicit_request = Path(request_path)
    if not str(explicit_request).strip():
        raise ValueError("request_path must be an explicit non-empty path")
    (
        sdm_request,
        request_bytes,
        manifest,
        digests,
        selection_paths,
        explicit_sdm_dir,
        completion_path,
        view_attestation,
    ) = _validate_sdm_request(config, explicit_request, config_path=config_path)
    sdm_paths, sdm_completion, completion_bytes, sdm_arrays = _validate_sdm_completion(
        config,
        sdm_request,
        request_bytes,
        completion_path,
        manifest,
        digests,
        selection_paths,
        explicit_sdm_dir,
        explicit_request,
    )
    if not same_directory_identity(
        explicit_sdm_dir,
        sdm_request["sdm_output_identity"],
        label="SDM output directory",
    ):
        raise ValueError(f"SDM output directory was replaced before score acceptance: {explicit_sdm_dir}")
    lino_run = _read_json_mapping(Path(config.provenance_path), label="LINO run provenance")
    _validate_provenance(
        lino_run,
        label="LINO run provenance",
        config=config,
        digests=digests,
        paths=selection_paths,
        required_fields=("selection_manifest_path", "effective_selection_manifest_path"),
    )
    _validate_lino_runtime_provenance(
        lino_run,
        config=config,
        config_path=config_path,
    )
    lino_paths = _expected_lino_paths(config, manifest)

    run_objects = lino_run.get("objects")
    if not isinstance(run_objects, list):
        raise ValueError("LINO run provenance objects must be a list")
    run_by_name: dict[str, Mapping[str, Any]] = {}
    for item in run_objects:
        if not isinstance(item, Mapping) or not isinstance(item.get("object_name"), str):
            raise ValueError("LINO run provenance has an invalid object record")
        name = item["object_name"]
        if name in run_by_name:
            raise ValueError(f"duplicate LINO provenance object: {name}")
        run_by_name[name] = item
    expected_names = {record.name for record in manifest.objects}
    if set(run_by_name) != expected_names:
        raise ValueError("LINO run provenance object set does not match current manifest")

    rows: list[dict[str, Any]] = []
    for record in manifest.objects:
        gt, gt_path = load_source_gt(config, record)
        support = normal_validity_mask(gt)
        lino_path = lino_paths[record.name]
        run_item = run_by_name[record.name]
        required_object_fields = ("output_path", "output_sha256", "source_geometry")
        missing_object_fields = [field for field in required_object_fields if field not in run_item]
        if missing_object_fields:
            raise ValueError(
                f"LINO provenance object {record.name} is missing required field(s): "
                f"{', '.join(missing_object_fields)}"
            )
        if not _same_path(run_item["output_path"], lino_path):
            raise ValueError(f"LINO provenance output_path mismatch for {record.name}")
        lino_artifact = _read_prediction_artifact(
            lino_path,
            label=f"LINO prediction for {record.name}",
        )
        if run_item["output_sha256"] != lino_artifact["sha256"]:
            raise ValueError(f"LINO prediction digest mismatch for {record.name}")
        _validate_source_geometry(run_item["source_geometry"], record)

        lino = lino_artifact["array"]
        sdm = sdm_arrays[record.name]
        expected_shape = (record.height, record.width, 3)
        if lino.shape != expected_shape:
            raise ValueError(f"LINO prediction geometry mismatch for {record.name}")
        if sdm.shape != expected_shape:
            raise ValueError(f"SDM prediction geometry mismatch for {record.name}")
        lino_metrics = angular_metrics(gt, lino, support)
        sdm_metrics = angular_metrics(gt, sdm, support)
        row: dict[str, Any] = {
            "object_name": record.name,
            "height": int(record.height),
            "width": int(record.width),
            "lino_valid_pixel_count": int(lino_metrics["valid_pixel_count"]),
            "sdm_valid_pixel_count": int(sdm_metrics["valid_pixel_count"]),
        }
        for name in _METRIC_NAMES:
            row[f"lino_{name}"] = float(lino_metrics[name])
            row[f"sdm_{name}"] = float(sdm_metrics[name])
        row["lino_minus_sdm_mae"] = float(row["lino_mae"] - row["sdm_mae"])
        rows.append(row)

    lino_macro = {
        name: float(sum(float(row[f"lino_{name}"]) for row in rows) / len(rows))
        for name in _METRIC_NAMES
    }
    sdm_macro = {
        name: float(sum(float(row[f"sdm_{name}"]) for row in rows) / len(rows))
        for name in _METRIC_NAMES
    }
    total_valid = sum(int(row["lino_valid_pixel_count"]) for row in rows)
    if total_valid <= 0:
        raise ValueError("comparison has no valid source-GT support")
    lino_weighted = {
        name: float(
            sum(float(row[f"lino_{name}"]) * int(row["lino_valid_pixel_count"]) for row in rows)
            / total_valid
        )
        for name in _LINEAR_METRIC_NAMES
    }
    sdm_weighted = {
        name: float(
            sum(float(row[f"sdm_{name}"]) * int(row["sdm_valid_pixel_count"]) for row in rows)
            / total_valid
        )
        for name in _LINEAR_METRIC_NAMES
    }
    lino_weighted["valid_pixel_count"] = total_valid
    sdm_weighted["valid_pixel_count"] = total_valid

    comparison_dir = Path(config.policy_root) / "comparison"
    csv_fields = [
        "object_name",
        "height",
        "width",
        "lino_valid_pixel_count",
        "sdm_valid_pixel_count",
        *(f"lino_{name}" for name in _METRIC_NAMES),
        *(f"sdm_{name}" for name in _METRIC_NAMES),
        "lino_minus_sdm_mae",
    ]
    if not same_directory_identity(
        explicit_sdm_dir,
        sdm_request["sdm_output_identity"],
        label="SDM output directory",
    ):
        raise ValueError(f"SDM output directory was replaced before score acceptance: {explicit_sdm_dir}")
    csv_path = _atomic_csv_write(comparison_dir / "per_object.csv", rows, csv_fields)
    summary: dict[str, Any] = {
        "object_count": len(rows),
        "objects": len(rows),
        "valid_pixel_count": total_valid,
        "mask_policy": config.mask_policy,
        "input_manifest_sha256": digests["input_manifest_sha256"],
        "selection_manifest_sha256": digests["selection_manifest_sha256"],
        "effective_selection_manifest_sha256": digests["effective_selection_manifest_sha256"],
        "input_manifest_path": str(config.input_manifest_path),
        "selection_manifest_path": str(selection_paths[0]),
        "effective_selection_manifest_path": str(selection_paths[1]),
        "lino_output_dir": str(config.lino_output_dir),
        "sdm_output_dir": str(explicit_sdm_dir),
        "sdm_request_path": str(explicit_request.resolve(strict=True)),
        "sdm_request_sha256": sha256_bytes(request_bytes),
        "sdm_completion_path": str(completion_path),
        "sdm_completion_sha256": sha256_bytes(completion_bytes),
        "sdm_request_id": sdm_request["request_id"],
        "sdm_run_fingerprint": sdm_request["run_fingerprint"],
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
        "macro_object": {"lino": lino_macro, "sdm": sdm_macro},
        "pixel_weighted": {"lino": lino_weighted, "sdm": sdm_weighted},
        "valid_pixel_weighted": {"lino": lino_weighted, "sdm": sdm_weighted},
        # Keep the aggregate terminology used by the existing metrics helper
        # available to downstream consumers while retaining concise names.
        "macro_object_mean": {"lino": lino_macro, "sdm": sdm_macro},
        "global_pixel_weighted_mean": {"lino": lino_weighted, "sdm": sdm_weighted},
        "lino_minus_sdm_mae": float(lino_macro["mae"] - sdm_macro["mae"]),
        "pixel_weighted_lino_minus_sdm_mae": float(
            lino_weighted["mae"] - sdm_weighted["mae"]
        ),
        "lino_minus_sdm": {
            "mae": float(lino_macro["mae"] - sdm_macro["mae"]),
            "pixel_weighted_mae": float(lino_weighted["mae"] - sdm_weighted["mae"]),
        },
        "manifests": {
            "input_sha256": digests["input_manifest_sha256"],
            "selection_sha256": digests["selection_manifest_sha256"],
            "effective_selection_sha256": digests["effective_selection_manifest_sha256"],
        },
        "per_object_csv": str(csv_path),
    }
    summary_path = _atomic_json_write(comparison_dir / "summary.json", summary)
    summary["summary_json"] = str(summary_path)
    return {
        **summary,
        "summary": summary,
        "per_object": rows,
        "summary_json": str(summary_path),
    }


__all__ = [
    "normal_validity_mask",
    "angular_metrics",
    "load_source_gt",
    "finalize_sdm_run",
    "score_lino_and_sdm",
]
