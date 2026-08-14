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
import os
import platform
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.comparison.metrics import angular_metrics
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
    publish_epoch_artifacts,
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
    PrivateSplitManifest,
    build_private_split_manifest,
    private_manifest_sha256,
)
from src.training.reproducibility import epoch_permutation, seed_everything, seed_worker


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
) -> tuple[_SplitStats, int]:
    """Run one target-mask-only training epoch and return object-weighted stats."""

    device = device or next(model.parameters()).device
    model.train()
    batches = _validate_loader(train_loader)
    total_loss = 0.0
    total_mae = 0.0
    total_objects = 0
    started = clock()
    for batch in train_loader:
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
            )
            predictions = decode_private_chunks(model, encoded, chunks)
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
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mae += float(mae.detach().cpu()) * batch_size
        total_objects += batch_size
        global_step += 1
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
            prediction = model.model_step(released_batch)
            if source_predictor is not None:
                prediction = _invoke_source_predictor(source_predictor, model, prediction, batch)
                arrays = list(prediction) if isinstance(prediction, (list, tuple)) else _prediction_arrays(prediction, batch)
            else:
                arrays = _prediction_arrays(prediction, batch)
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
            model = LiNo_UniPS(pixel_samples=config.pixel_samples)
    finally:
        torch.linspace = original_linspace  # type: ignore[assignment]
    return tuple((name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in model.state_dict().items())


def _production_model_factory(config: PrivateTrainConfig) -> torch.nn.Module:
    from src.models.Net_module import LiNo_UniPS

    return LiNo_UniPS(pixel_samples=config.pixel_samples)


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
        for raw in csv.DictReader(stream):
            rows.append(EpochMetrics(
                epoch=int(raw["epoch"]),
                train_loss=float(raw["train_loss"]),
                train_mae=float(raw["train_mae"]),
                validation_loss=float(raw["validation_loss"]),
                validation_mae=float(raw["validation_mae"]),
                learning_rate=float(raw["learning_rate"]),
                global_step=int(raw["global_step"]),
                epoch_seconds=float(raw["epoch_seconds"]),
                peak_cuda_allocated_bytes=int(raw["peak_cuda_allocated_bytes"]) if raw.get("peak_cuda_allocated_bytes") else None,
                peak_cuda_reserved_bytes=int(raw["peak_cuda_reserved_bytes"]) if raw.get("peak_cuda_reserved_bytes") else None,
            ))
    return rows


def _publish_bundle(
    run_dir: Path,
    *,
    epoch: int,
    checkpoint_bytes: bytes,
    export_weights: bytes,
    export_sidecar: bytes,
    metrics: bytes,
    best: bool,
    model: torch.nn.Module,
    model_factory: Callable[[], torch.nn.Module],
) -> None:
    """Validate one staged bundle, then publish all public aliases together."""

    stage = Path(tempfile.mkdtemp(prefix=f".epoch-{epoch:03d}-", dir=run_dir))
    try:
        validated = publish_epoch_artifacts(
            stage,
            artifacts={
                "epoch.ckpt": checkpoint_bytes,
                "epoch.pth": export_weights,
                "epoch.json": export_sidecar,
                "metrics.csv": metrics,
            },
        )
        staged = {name: path.read_bytes() for name, path in validated.items()}
        destinations: dict[Path, bytes] = {
            run_dir / "metrics.csv": staged["metrics.csv"],
            run_dir / "checkpoints" / "last.ckpt": staged["epoch.ckpt"],
            run_dir / "checkpoints" / f"lino_epoch_{epoch:03d}.ckpt": staged["epoch.ckpt"],
            run_dir / "exports" / f"lino_epoch_{epoch:03d}.pth": staged["epoch.pth"],
            run_dir / "exports" / f"lino_epoch_{epoch:03d}.json": staged["epoch.json"],
        }
        if best:
            destinations[run_dir / "checkpoints" / "best_validation.ckpt"] = staged["epoch.ckpt"]
            destinations[run_dir / "exports" / "lino_best_validation.pth"] = staged["epoch.pth"]
            destinations[run_dir / "exports" / "lino_best_validation.json"] = staged["epoch.json"]
        old: dict[Path, bytes | None] = {}
        temporary: dict[Path, Path] = {}
        try:
            for destination, payload in destinations.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                old[destination] = destination.read_bytes() if destination.exists() else None
                fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary[destination] = Path(name)
            for destination, source in temporary.items():
                os.replace(source, destination)
            for parent in {destination.parent for destination in destinations}:
                try:
                    fd = os.open(parent, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except OSError:
                    pass
        except Exception:
            for destination, source in temporary.items():
                if source.exists():
                    source.unlink()
                if old.get(destination) is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.write_bytes(old[destination])
            raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def run_private_training(
    config: PrivateTrainConfig,
    *,
    config_path: str | Path | None = None,
    smoke: bool = False,
    schema_provider: Callable[[], Sequence[tuple[str, Sequence[int], str]]] | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    device_resolver: Callable[[str], torch.device] = resolve_device,
    manifest_builder: Callable[[PrivateTrainConfig, str], PrivateSplitManifest] = build_private_split_manifest,
    dataset_factory: Callable[..., Dataset] = PrivateExrTrainDataset,
    source_predictor: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingSummary:
    """Run cold-start, initialization, or exact epoch-boundary resume training."""

    if not isinstance(config, PrivateTrainConfig):
        raise TypeError("config must be a PrivateTrainConfig")
    seed_everything(config.seed, config.deterministic)
    train_manifest = manifest_builder(config, "train")
    test_manifest = manifest_builder(config, "test")
    if not isinstance(train_manifest, PrivateSplitManifest) or not isinstance(test_manifest, PrivateSplitManifest):
        raise TypeError("manifest_builder must return PrivateSplitManifest values")
    final_digest = _read_final_selection(config.final_selection_manifest)
    schema_provider = schema_provider or (lambda: _production_schema_provider(config))
    expected_schema = tuple(schema_provider())
    architecture_digest = schema_fingerprint(expected_schema)
    source_revision = "lino-private-exr-training-v1"
    contract = build_run_contract(
        config,
        architecture_schema=expected_schema,
        train_manifest_sha256=private_manifest_sha256(train_manifest),
        test_manifest_sha256=private_manifest_sha256(test_manifest),
        final_selection_manifest_sha256=final_digest,
        source_revision=source_revision,
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
    device = device_resolver(config.device)
    if config.precision == "bf16" and device.type != "cuda":
        raise RuntimeError("private LINO bf16 training requires CUDA")
    factory = model_factory or (lambda: _production_model_factory(config))
    model = factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    live_digest = schema_fingerprint(model)
    if live_digest != architecture_digest:
        raise ValueError("live model schema differs from CPU schema preflight")
    model.to(device=device)
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
    existing_rows = _load_metric_rows(run_dir / "metrics.csv") if config.startup_mode == "resume" else []
    if config.startup_mode == "resume" and progress_epoch > 0 and not existing_rows:
        raise ValueError("resume requires metrics.csv through completed_epoch")
    if existing_rows and existing_rows[-1].epoch != progress_epoch:
        raise ValueError("resume metrics.csv does not end at checkpoint completed_epoch")
    if smoke:
        train_indices = list(range(min(2, len(train_dataset))))
        test_indices = list(range(min(2, len(test_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    rows = list(existing_rows)
    final_export = run_dir / "exports" / f"lino_epoch_{progress_epoch:03d}.pth"
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
            _publish_bundle(
                run_dir,
                epoch=metrics.epoch,
                checkpoint_bytes=checkpoint_bytes,
                export_weights=export_weights,
                export_sidecar=export_sidecar,
                metrics=_metric_bytes(next_rows),
                best=is_best,
                model=model,
                model_factory=factory,
            )
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
