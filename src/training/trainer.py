"""Explicit, dependency-injected trainer for private LINO EXR data.

The released model remains the only owner of trainable parameters.  This
module only owns startup ordering, data-loader scheduling, objective/metric
aggregation, and durable epoch publication.  Keeping these boundaries
explicit makes the CPU contract testable without constructing the released
network or allocating CUDA memory.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import io
import json
import platform
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.comparison.metrics import GT_VALIDITY_POLICY, angular_metrics
from src.comparison.reporting import format_clock_duration
from src.data.lino_native_preprocessing import restore_lino_prediction
from src.data.private_exr_train import PrivateExrTrainDataset, collate_private_exr
from src.training.checkpointing import (
    BestMetrics,
    TrainingProgress,
    apply_artifact_retention,
    build_run_contract,
    export_inference_weights,
    load_initial_weights,
    load_resume_checkpoint,
    preflight_startup_checkpoint,
    publish_tree_artifacts,
    save_resume_checkpoint,
    schema_fingerprint,
)
from src.training.config import PrivateTrainConfig, resolved_config_dict
from src.training.model_adapter import decode_private_chunks, encode_private_batch
from src.training.objective import (
    component_sse_batch,
    finite_gradients_or_raise,
    plan_target_chunks,
    sampled_angular_mae,
)
from src.training.private_manifest import (
    PrivateSplitIndex,
    build_private_split_index,
    private_index_sha256,
)
from src.training.reproducibility import (
    epoch_permutation,
    seed_everything,
    seed_worker,
    stable_seed,
)


METRIC_FIELDS = (
    "epoch",
    "train_loss",
    "train_mae",
    "validation_loss",
    "validation_mae",
    "learning_rate",
    "global_step",
    "epoch_seconds",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_mae: float
    validation_loss: float
    validation_mae: float
    learning_rate: float
    global_step: int
    epoch_seconds: float
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None


@dataclass(frozen=True)
class TrainingSummary:
    run_dir: Path
    last_checkpoint: Path
    best_checkpoint: Path
    final_export: Path
    completed_epoch: int


@dataclass(frozen=True)
class _SplitStats:
    loss: float
    mae: float
    objects: int


def resolve_device(spec: str) -> torch.device:
    """Resolve the configured runtime device without silently falling back."""

    if spec not in {"cuda", "cpu", "auto"}:
        raise ValueError("device must be cuda, cpu, or auto")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _autocast(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def create_optimizer_scheduler(
    model: torch.nn.Module,
    config: PrivateTrainConfig,
):
    """Create the fixed AdamW/StepLR pair from the private config."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("released model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adamw_betas,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )
    return optimizer, scheduler


def _batch_tensor(batch: Mapping[str, Any], name: str, device: torch.device) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{name!r}] must be a tensor")
    return value.to(device=device, non_blocking=device.type == "cuda")


def _metadata_names(batch: Mapping[str, Any], count: int) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise ValueError("batch metadata must be a sequence")
    if len(metadata) != count:
        raise ValueError("batch metadata length does not match batch size")
    names: list[str] = []
    for item in metadata:
        if not isinstance(item, Mapping) or not isinstance(item.get("object_name"), str):
            raise ValueError("batch metadata must contain object_name")
        names.append(str(item["object_name"]))
    return names


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device=device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
    return moved


def _validate_loader(loader: Any) -> int:
    try:
        length = len(loader)
    except TypeError as exc:
        raise ValueError("training loader must have a finite length") from exc
    if length <= 0:
        raise ValueError("private training dataset must contain at least one object")
    return int(length)


def _cuda_peak(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return int(torch.cuda.max_memory_allocated(device)), int(torch.cuda.max_memory_reserved(device))


def _cuda_progress(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    gib = float(1024**3)
    allocated = torch.cuda.memory_allocated(device) / gib
    reserved = torch.cuda.memory_reserved(device) / gib
    return f" | VRAM: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved"


@contextmanager
def _preserve_validation_numpy_state(seed: int):
    """Seed released NumPy pixel grouping without perturbing caller state."""

    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def train_epoch(
    model: torch.nn.Module,
    train_loader: Any,
    config: PrivateTrainConfig,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device | None = None,
    epoch: int = 0,
    global_step: int = 0,
    clock: Callable[[], float] = time.perf_counter,
    first_batch_callback: Callable[[], None] | None = None,
) -> tuple[_SplitStats, int]:
    """Run one target-mask-only training epoch and return object-weighted stats."""

    device = device or next(model.parameters()).device
    model.train()
    batches = _validate_loader(train_loader)
    total_loss = 0.0
    total_mae = 0.0
    total_objects = 0
    started = clock()
    first_batch_reported = False
    for batch_index, batch in enumerate(train_loader, start=1):
        batch_started = clock()
        if not first_batch_reported:
            first_batch_reported = True
            if first_batch_callback is not None:
                first_batch_callback()
        if not isinstance(batch, Mapping):
            raise ValueError("private training batch must be a mapping")
        batch = _move_batch(batch, device)
        observations = _batch_tensor(batch, "imgs", device)
        model_mask = _batch_tensor(batch, "model_mask", device)
        targets = _batch_tensor(batch, "target_normal", device)
        target_mask = _batch_tensor(batch, "target_mask", device)
        batch_size = int(observations.shape[0])
        names = _metadata_names(batch, batch_size)
        chunks = tuple(
            plan_target_chunks(
                target_mask[index],
                pixel_samples=config.pixel_samples,
                pixel_budget=config.train_pixel_budget,
                base_seed=config.seed,
                split="train",
                epoch=epoch,
                object_name=names[index],
            )
            for index in range(batch_size)
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, config.precision):
            encoded = encode_private_batch(
                model,
                observations,
                model_mask,
                canonical_resolution=config.canonical_resolution,
                activation_checkpointing=config.activation_checkpointing,
            )
            predictions = decode_private_chunks(
                model,
                encoded,
                chunks,
                activation_checkpointing=config.activation_checkpointing,
            )
            loss = component_sse_batch(predictions, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("training loss is non-finite")
        loss.backward()
        finite_gradients_or_raise(model)
        optimizer.step()
        with torch.no_grad():
            mae = sampled_angular_mae(predictions, targets)
        if not torch.isfinite(mae).item():
            raise FloatingPointError("training angular MAE is non-finite")
        batch_loss = float(loss.detach().cpu())
        batch_mae = float(mae.detach().cpu())
        total_loss += batch_loss * batch_size
        total_mae += batch_mae * batch_size
        total_objects += batch_size
        global_step += 1
        interval = config.train_log_every_batches
        should_report = interval > 0 and (
            batch_index == 1 or batch_index % interval == 0 or batch_index == batches
        )
        if should_report:
            elapsed = float(clock() - started)
            step_seconds = float(clock() - batch_started)
            remaining = batches - batch_index
            eta = (elapsed / batch_index) * remaining
            width = max(4, len(str(batches)))
            print(
                f"[Batch {batch_index:0{width}d}/{batches}] "
                f"Loss: {batch_loss:.4f} | MAE (avg so far): "
                f"{total_mae / total_objects:.4f} | "
                f"Elapsed: {format_clock_duration(elapsed)} | "
                f"Step: {format_clock_duration(step_seconds)} | "
                f"ETA: {format_clock_duration(eta)}"
                f"{_cuda_progress(device)}",
                flush=True,
            )
    if total_objects == 0:  # pragma: no cover - guarded by loader length
        raise ValueError("private training dataset must contain at least one object")
    del started, batches
    return _SplitStats(total_loss / total_objects, total_mae / total_objects, total_objects), global_step


def _prediction_arrays(
    prediction: Any,
    batch: Mapping[str, Any],
) -> list[np.ndarray]:
    if isinstance(prediction, torch.Tensor):
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("released validation prediction must have shape [B, 3, H, W]")
        values = [prediction[index].detach().float().cpu().permute(1, 2, 0).numpy() for index in range(prediction.shape[0])]
    elif isinstance(prediction, np.ndarray) and prediction.ndim == 4:
        values = [prediction[index] for index in range(prediction.shape[0])]
    elif isinstance(prediction, (list, tuple)):
        values = list(prediction)
    else:
        raise ValueError("released validation prediction must be a tensor or array sequence")
    metadata = batch.get("metadata")
    if not isinstance(metadata, Sequence) or len(metadata) != len(values):
        raise ValueError("validation metadata length does not match prediction batch")
    restored: list[np.ndarray] = []
    for value, item in zip(values, metadata):
        if not isinstance(item, Mapping):
            raise ValueError("validation metadata must be a mapping")
        source_geometry = item.get("source_geometry")
        roi = item.get("roi")
        if not isinstance(source_geometry, Mapping) or not isinstance(roi, (list, tuple)):
            raise ValueError("validation metadata must contain source_geometry and roi")
        height = int(source_geometry["height"])
        width = int(source_geometry["width"])
        restored.append(restore_lino_prediction(value, roi, source_height=height, source_width=width))
    return restored


def _invoke_source_predictor(
    callback: Callable[..., Any],
    model: torch.nn.Module,
    prediction: Any,
    batch: Mapping[str, Any],
) -> Any:
    """Support compact test callbacks while retaining a stable production hook."""

    try:
        parameters = inspect.signature(callback).parameters
        positional = [item for item in parameters.values() if item.kind in (item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD)]
        if len(positional) >= 3:
            return callback(model, prediction, batch)
    except (TypeError, ValueError):
        pass
    return callback(prediction, batch)


def validate_epoch(
    model: torch.nn.Module,
    validation_loader: Any,
    config: PrivateTrainConfig,
    *,
    source_predictor: Callable[..., Any] | None = None,
    device: torch.device | None = None,
) -> _SplitStats:
    """Run sequential validation with the released ``{imgs, mask}`` boundary."""

    device = device or next(model.parameters()).device
    model.eval()
    _validate_loader(validation_loader)
    total_loss = 0.0
    total_mae = 0.0
    total_objects = 0
    with torch.inference_mode():
        for raw_batch in validation_loader:
            if not isinstance(raw_batch, Mapping):
                raise ValueError("validation batch must be a mapping")
            batch = _move_batch(raw_batch, device)
            released_batch = {
                "imgs": _batch_tensor(batch, "imgs", device),
                "mask": _batch_tensor(batch, "model_mask", device),
            }
            names = _metadata_names(batch, int(released_batch["imgs"].shape[0]))
            if len(names) != 1:
                raise ValueError("validation loader must use batch size one")
            validation_seed = stable_seed(
                config.seed,
                "test",
                0,
                names[0],
                "pixel_grouping",
            )
            with _autocast(device, config.precision):
                with _preserve_validation_numpy_state(validation_seed):
                    prediction = model.model_step(released_batch)
                    if source_predictor is not None:
                        prediction = _invoke_source_predictor(
                            source_predictor,
                            model,
                            prediction,
                            batch,
                        )
                    arrays = (
                        list(prediction)
                        if isinstance(prediction, (list, tuple))
                        else _prediction_arrays(prediction, batch)
                    )
            targets = _batch_tensor(batch, "source_target_normal", device).detach().cpu().numpy()
            masks = _batch_tensor(batch, "source_target_mask", device).detach().cpu().numpy()
            if len(arrays) != int(targets.shape[0]):
                raise ValueError("validation prediction count does not match target batch")
            for index, array in enumerate(arrays):
                target = np.transpose(targets[index], (1, 2, 0))
                support = masks[index, 0] > 0
                if not np.any(support):
                    raise ValueError("validation target support is empty")
                array = np.asarray(array, dtype=np.float32)
                if array.shape != target.shape:
                    raise ValueError("restored prediction geometry does not match source target")
                diff = (array.astype(np.float64) - target.astype(np.float64)) ** 2
                object_loss = float(np.mean(np.sum(diff[support], axis=-1)))
                metrics = angular_metrics(target, array, support.astype(bool))
                object_mae = float(metrics["mae"])
                if not np.isfinite(object_loss) or not np.isfinite(object_mae):
                    raise FloatingPointError("validation metric is non-finite")
                total_loss += object_loss
                total_mae += object_mae
                total_objects += 1
    if total_objects == 0:
        raise ValueError("validation dataset must contain at least one object")
    return _SplitStats(total_loss / total_objects, total_mae / total_objects, total_objects)


def _read_final_selection(path: Path) -> str:
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read final-selection manifest: {path}") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("final-selection manifest must be a nonempty JSON mapping")
    for object_name, values in raw.items():
        if not isinstance(object_name, str) or not isinstance(values, list) or len(values) != 16:
            raise ValueError("final-selection manifest must contain exactly 16 lights per object")
        if any(not isinstance(value, str) or "/" in value or "\\" in value for value in values):
            raise ValueError("final-selection manifest contains duplicate or unsafe light names")
        if len(set(values)) != 16:
            raise ValueError("final-selection manifest contains duplicate or unsafe light names")
    return __import__("hashlib").sha256(raw_bytes).hexdigest()


def _production_schema_provider(config: PrivateTrainConfig):
    from src.models.Net_module import LiNo_UniPS

    # The released DINO positional-token constructor calls ``linspace``
    # without an explicit device.  Pin that one operation to CPU while the
    # surrounding parameters are meta tensors; this keeps schema preflight
    # allocation-free and mirrors the adapter's real-schema guard.
    original_linspace = torch.linspace

    def _cpu_linspace(*args, **kwargs):
        values = dict(kwargs)
        values["device"] = "cpu"
        return original_linspace(*args, **values)

    torch.linspace = _cpu_linspace  # type: ignore[assignment]
    try:
        with torch.device("meta"):
            model = LiNo_UniPS(
                pixel_samples=config.pixel_samples,
                model_resolution=config.max_image_resolution,
                canonical_resolution=config.canonical_resolution,
            )
    finally:
        torch.linspace = original_linspace  # type: ignore[assignment]
    return tuple((name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in model.state_dict().items())


def _production_model_factory(config: PrivateTrainConfig) -> torch.nn.Module:
    from src.models.Net_module import LiNo_UniPS

    return LiNo_UniPS(
        pixel_samples=config.pixel_samples,
        model_resolution=config.max_image_resolution,
        canonical_resolution=config.canonical_resolution,
    )


def _runtime_versions() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _save_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _metric_bytes(rows: Sequence[EpochMetrics]) -> bytes:
    expected_epochs = list(range(1, len(rows) + 1))
    if [row.epoch for row in rows] != expected_epochs:
        raise ValueError("metrics rows must form a contiguous epoch prefix")
    previous_step = -1
    for row in rows:
        if any(
            not np.isfinite(float(value))
            for value in (
                row.train_loss,
                row.train_mae,
                row.validation_loss,
                row.validation_mae,
                row.learning_rate,
                row.epoch_seconds,
            )
        ):
            raise ValueError("metrics rows must contain finite values")
        if row.global_step < previous_step:
            raise ValueError("metrics rows global_step must be monotonic")
        previous_step = row.global_step
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "epoch": row.epoch,
            "train_loss": f"{row.train_loss:.9g}",
            "train_mae": f"{row.train_mae:.9g}",
            "validation_loss": f"{row.validation_loss:.9g}",
            "validation_mae": f"{row.validation_mae:.9g}",
            "learning_rate": f"{row.learning_rate:.9g}",
            "global_step": row.global_step,
            "epoch_seconds": f"{row.epoch_seconds:.9g}",
            "peak_cuda_allocated_bytes": "" if row.peak_cuda_allocated_bytes is None else row.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": "" if row.peak_cuda_reserved_bytes is None else row.peak_cuda_reserved_bytes,
        })
    return stream.getvalue().encode("utf-8")


def _load_metric_rows(path: Path) -> list[EpochMetrics]:
    if not path.exists():
        return []
    rows: list[EpochMetrics] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError("metrics.csv header does not match the exact trainer schema")
        for expected_epoch, raw in enumerate(reader, start=1):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError("metrics.csv contains an incomplete row")
            try:
                epoch = int(raw["epoch"])
                global_step = int(raw["global_step"])
                peak_allocated = (
                    int(raw["peak_cuda_allocated_bytes"])
                    if raw["peak_cuda_allocated_bytes"]
                    else None
                )
                peak_reserved = (
                    int(raw["peak_cuda_reserved_bytes"])
                    if raw["peak_cuda_reserved_bytes"]
                    else None
                )
                numeric = {
                    key: float(raw[key])
                    for key in (
                        "train_loss",
                        "train_mae",
                        "validation_loss",
                        "validation_mae",
                        "learning_rate",
                        "epoch_seconds",
                    )
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("metrics.csv contains an invalid numeric row") from exc
            if epoch != expected_epoch or epoch <= 0:
                raise ValueError("metrics.csv epochs must be a contiguous prefix")
            if global_step < 0 or (rows and global_step < rows[-1].global_step):
                raise ValueError("metrics.csv global_step must be nonnegative and monotonic")
            if peak_allocated is not None and peak_allocated < 0:
                raise ValueError("metrics.csv allocated bytes must be nonnegative")
            if peak_reserved is not None and peak_reserved < 0:
                raise ValueError("metrics.csv reserved bytes must be nonnegative")
            if any(not np.isfinite(value) for value in numeric.values()):
                raise ValueError("metrics.csv contains non-finite metrics")
            rows.append(EpochMetrics(
                epoch=epoch,
                train_loss=numeric["train_loss"],
                train_mae=numeric["train_mae"],
                validation_loss=numeric["validation_loss"],
                validation_mae=numeric["validation_mae"],
                learning_rate=numeric["learning_rate"],
                global_step=global_step,
                epoch_seconds=numeric["epoch_seconds"],
                peak_cuda_allocated_bytes=peak_allocated,
                peak_cuda_reserved_bytes=peak_reserved,
            ))
    if not rows:
        raise ValueError("metrics.csv contains no completed epoch rows")
    return rows


def run_private_training(
    config: PrivateTrainConfig,
    *,
    config_path: str | Path | None = None,
    smoke: bool = False,
    schema_provider: Callable[[], Sequence[tuple[str, Sequence[int], str]]] | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    device_resolver: Callable[[str], torch.device] = resolve_device,
    index_builder: Callable[[PrivateTrainConfig, str], PrivateSplitIndex] = build_private_split_index,
    dataset_factory: Callable[..., Dataset] = PrivateExrTrainDataset,
    source_predictor: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingSummary:
    """Run cold-start, initialization, or exact epoch-boundary resume training."""

    if not isinstance(config, PrivateTrainConfig):
        raise TypeError("config must be a PrivateTrainConfig")
    run_started = clock()
    seed_everything(config.seed, config.deterministic)
    indexes: dict[str, PrivateSplitIndex] = {}
    for split, root in (("train", config.train_dir), ("test", config.test_dir)):
        index_started = clock()
        print(f"Indexing {root}")
        index = index_builder(config, split)
        if not isinstance(index, PrivateSplitIndex):
            raise TypeError("index_builder must return PrivateSplitIndex values")
        indexes[split] = index
        elapsed = float(clock() - index_started)
        print(
            f"Indexed {len(index.objects)} objects in "
            f"{format_clock_duration(elapsed)} (content validation: lazy)"
        )
    train_manifest = indexes["train"]
    test_manifest = indexes["test"]
    final_digest = (
        None
        if config.final_selection_manifest is None
        else _read_final_selection(config.final_selection_manifest)
    )
    print("Preparing LINO model schema...", flush=True)
    schema_provider = schema_provider or (lambda: _production_schema_provider(config))
    expected_schema = tuple(schema_provider())
    architecture_digest = schema_fingerprint(expected_schema)
    print("LINO model schema ready.", flush=True)
    source_revision = "lino-private-exr-training-v2"
    contract = build_run_contract(
        config,
        architecture_schema=expected_schema,
        train_manifest_sha256=private_index_sha256(train_manifest),
        test_manifest_sha256=private_index_sha256(test_manifest),
        final_selection_manifest_sha256=final_digest,
        source_revision=source_revision,
        gt_validity_policy=GT_VALIDITY_POLICY,
        runtime_versions=_runtime_versions(),
        run_kind="smoke" if smoke else "experiment",
        comparable=not smoke,
    )
    startup = preflight_startup_checkpoint(
        config.startup_mode,
        config.init_checkpoint if config.startup_mode == "init_checkpoint" else config.resume_checkpoint,
        expected_schema=expected_schema,
        expected_contract=contract if config.startup_mode == "resume" else None,
    )
    run_dir = config.save_dir / "smoke" if smoke else config.save_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_config_dict(config)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _save_json(run_dir / "data_contract.json", contract)
    # Keep JSON aliases useful to simple operators while the documented YAML
    # and contract filenames remain the canonical artifacts.
    _save_json(run_dir / "resolved_config.json", resolved)
    _save_json(run_dir / "run_contract.json", contract)
    existing_rows = _load_metric_rows(run_dir / "metrics.csv") if config.startup_mode == "resume" else []
    if config.startup_mode == "resume":
        startup_progress = startup.load_report.get("progress")
        if not isinstance(startup_progress, Mapping):
            raise ValueError("resume checkpoint preflight has no progress report")
        expected_completed = int(startup_progress["completed_epoch"])
        if expected_completed > 0 and not existing_rows:
            raise ValueError("resume requires metrics.csv through completed_epoch")
        if existing_rows and existing_rows[-1].epoch != expected_completed:
            raise ValueError("resume metrics.csv does not end at checkpoint completed_epoch")
    device = device_resolver(config.device)
    print(f"Using device: {device}", flush=True)
    if config.precision == "bf16" and device.type != "cuda":
        raise RuntimeError("private LINO bf16 training requires CUDA")
    factory = model_factory or (lambda: _production_model_factory(config))
    print("Initializing LINO model weights...", flush=True)
    model = factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    live_digest = schema_fingerprint(model)
    if live_digest != architecture_digest:
        raise ValueError("live model schema differs from CPU schema preflight")
    model.to(device=device)
    print("LINO model initialized and moved to the training device.", flush=True)
    optimizer, scheduler = create_optimizer_scheduler(model, config)
    progress_epoch = 0
    global_step = 0
    best = BestMetrics(float("inf"), float("inf"), 0)
    if config.startup_mode == "init_checkpoint":
        load_initial_weights(startup, model=model)
    elif config.startup_mode == "resume":
        resumed = load_resume_checkpoint(
            startup,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=contract,
        )
        progress_epoch = resumed.progress.completed_epoch
        global_step = resumed.progress.global_step
        best = resumed.best
    train_dataset = dataset_factory(config, train_manifest, split="train")
    test_dataset = dataset_factory(config, test_manifest, split="test")
    if len(train_dataset) <= 0 or len(test_dataset) <= 0:
        raise ValueError("private training and validation datasets must be nonempty")
    if config.startup_mode == "resume" and progress_epoch > 0 and not existing_rows:
        raise ValueError("resume requires metrics.csv through completed_epoch")
    if existing_rows and existing_rows[-1].epoch != progress_epoch:
        raise ValueError("resume metrics.csv does not end at checkpoint completed_epoch")
    if smoke:
        train_indices = list(range(min(2, len(train_dataset))))
        test_indices = list(range(min(2, len(test_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    train_object_count = len(train_dataset)
    test_object_count = len(test_dataset)
    train_batch_count = (
        train_object_count + config.train_batch_size - 1
    ) // config.train_batch_size
    print(
        "Model geometry: "
        f"internal {config.max_image_resolution} | "
        f"canonical {config.canonical_resolution}",
        flush=True,
    )
    print(f"Precision: {config.precision}", flush=True)
    print(
        f"Training objects: {train_object_count} | "
        f"Validation objects: {test_object_count}",
        flush=True,
    )
    print(
        f"Batch size: {config.train_batch_size} | "
        f"Lights/object: {config.max_image_num} | "
        f"Batches/epoch: {train_batch_count}",
        flush=True,
    )
    print(
        f"AdamW optimizer: lr={config.learning_rate:g}, "
        f"weight_decay={config.weight_decay:g}, betas={config.adamw_betas}",
        flush=True,
    )
    print(
        f"Scheduler: StepLR step_size={config.scheduler_step_size}, "
        f"gamma={config.scheduler_gamma:g}",
        flush=True,
    )
    print(
        "Model startup: "
        f"{config.startup_mode} | activation checkpointing: "
        f"{'enabled' if config.activation_checkpointing else 'disabled'} | "
        f"batch log interval: {config.train_log_every_batches}",
        flush=True,
    )
    rows = list(existing_rows)
    final_export = run_dir / "exports" / f"lino_epoch_{progress_epoch:03d}.pth"
    first_batch_reported = False

    def report_first_batch() -> None:
        nonlocal first_batch_reported
        if not first_batch_reported:
            first_batch_reported = True
            print(
                "First training batch ready after "
                f"{format_clock_duration(float(clock() - run_started))}"
            )

    for epoch_index in range(progress_epoch, config.epochs):
        logical_epoch = epoch_index
        if hasattr(train_dataset, "dataset"):
            base_dataset = train_dataset.dataset
        else:
            base_dataset = train_dataset
        if hasattr(base_dataset, "set_epoch"):
            base_dataset.set_epoch(logical_epoch)
        order = epoch_permutation(len(train_dataset), base_seed=config.seed, epoch=logical_epoch)
        ordered = Subset(train_dataset, order)
        train_loader = DataLoader(
            ordered,
            batch_size=config.train_batch_size,
            shuffle=False,
            num_workers=config.train_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_private_exr,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.test_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_private_exr,
            pin_memory=device.type == "cuda",
        )
        print(
            f"[Epoch {logical_epoch + 1}/{config.epochs}] START "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"elapsed_total={format_clock_duration(float(clock() - run_started))}",
            flush=True,
        )
        print(
            f"Training {len(ordered)} objects in {len(train_loader)} batches",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = clock()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_stats, global_step = train_epoch(
            model,
            train_loader,
            config,
            optimizer=optimizer,
            device=device,
            epoch=logical_epoch,
            global_step=global_step,
            clock=clock,
            first_batch_callback=report_first_batch,
        )
        validation = validate_epoch(model, test_loader, config, source_predictor=source_predictor, device=device)
        scheduler.step()
        if not np.isfinite(validation.mae) or not np.isfinite(validation.loss):
            raise FloatingPointError("validation metric is non-finite")
        is_best = validation.mae < best.mae
        if is_best:
            best = BestMetrics(validation.mae, validation.loss, logical_epoch + 1)
        peak_allocated, peak_reserved = _cuda_peak(device)
        metrics = EpochMetrics(
            epoch=logical_epoch + 1,
            train_loss=train_stats.loss,
            train_mae=train_stats.mae,
            validation_loss=validation.loss,
            validation_mae=validation.mae,
            learning_rate=learning_rate,
            global_step=global_step,
            epoch_seconds=float(clock() - started),
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
        )
        next_rows = rows + [metrics]
        with tempfile.TemporaryDirectory(dir=run_dir) as temporary:
            temp_export = Path(temporary) / "epoch.pth"
            export = export_inference_weights(
                temp_export,
                model=model,
                model_factory=factory,
                metadata={
                    "epoch": metrics.epoch,
                    "data_contract": contract,
                    "preprocessing_version": config.preprocessing_version,
                    "source_revision": source_revision,
                    "gt_validity_policy": GT_VALIDITY_POLICY,
                    "architecture_schema_sha256": architecture_digest,
                    "run_kind": "smoke" if smoke else "experiment",
                    "comparable": not smoke,
                },
            )
            export_weights = export.weights_path.read_bytes()
            export_sidecar = export.metadata_path.read_bytes()
            # Capture RNG after fresh-model export construction.  A resumed
            # run therefore starts at the same epoch-boundary state as an
            # uninterrupted run, including any export-time allocations.
            temp_checkpoint = Path(temporary) / "epoch.ckpt"
            save_resume_checkpoint(
                temp_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                progress=TrainingProgress(metrics.epoch, metrics.epoch + 1, global_step),
                best=best,
                contract=contract,
            )
            checkpoint_bytes = temp_checkpoint.read_bytes()
            artifacts = {
                "metrics.csv": _metric_bytes(next_rows),
                "checkpoints/last.ckpt": checkpoint_bytes,
            }
            publish_epoch_artifact = (
                metrics.epoch % config.save_every_epochs == 0
                or metrics.epoch == config.epochs
                or metrics.epoch in config.keep_milestone_epochs
            )
            if publish_epoch_artifact:
                artifacts[f"checkpoints/lino_epoch_{metrics.epoch:03d}.ckpt"] = checkpoint_bytes
                artifacts[f"exports/lino_epoch_{metrics.epoch:03d}.pth"] = export_weights
                artifacts[f"exports/lino_epoch_{metrics.epoch:03d}.json"] = export_sidecar
            if is_best:
                artifacts["checkpoints/best_validation.ckpt"] = checkpoint_bytes
                artifacts["exports/lino_best_validation.pth"] = export_weights
                artifacts["exports/lino_best_validation.json"] = export_sidecar
            publish_tree_artifacts(run_dir, artifacts=artifacts)
        rows = next_rows
        final_export = run_dir / "exports" / f"lino_epoch_{metrics.epoch:03d}.pth"
        apply_artifact_retention(
            run_dir / "checkpoints",
            keep_milestone_epochs=config.keep_milestone_epochs,
            latest_epoch=metrics.epoch,
        )
        apply_artifact_retention(
            run_dir / "exports",
            keep_milestone_epochs=config.keep_milestone_epochs,
            latest_epoch=metrics.epoch,
        )
        print(
            f"Epoch {metrics.epoch}: train_loss={metrics.train_loss:.6f} "
            f"train_mae={metrics.train_mae:.4f} val_loss={metrics.validation_loss:.6f} "
            f"val_mae={metrics.validation_mae:.4f} lr={metrics.learning_rate:.6g} "
            f"time={metrics.epoch_seconds:.2f}s"
        )
    if progress_epoch == config.epochs:
        final_export = run_dir / "exports" / f"lino_epoch_{progress_epoch:03d}.pth"
    print(
        "Total training time: "
        f"{format_clock_duration(float(clock() - run_started))}"
    )
    return TrainingSummary(
        run_dir=run_dir,
        last_checkpoint=run_dir / "checkpoints" / "last.ckpt",
        best_checkpoint=run_dir / "checkpoints" / "best_validation.ckpt",
        final_export=final_export,
        completed_epoch=max(progress_epoch, config.epochs),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train released LINO on private EXR objects")
    parser.add_argument("--config", required=True, help="Private LINO training YAML")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run an isolated non-comparable two-object acceptance job",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_private_training(
        __import__("src.training.config", fromlist=["load_private_train_config"]).load_private_train_config(args.config),
        config_path=args.config,
        smoke=args.smoke,
    )
    return 0


__all__ = [
    "EpochMetrics",
    "METRIC_FIELDS",
    "TrainingSummary",
    "build_parser",
    "create_optimizer_scheduler",
    "main",
    "resolve_device",
    "run_private_training",
    "train_epoch",
    "validate_epoch",
]
