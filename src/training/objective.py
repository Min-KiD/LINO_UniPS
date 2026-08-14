"""Pixel planning, normal loss, and reporting metrics for private training."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import torch

from src.training.model_adapter import DecodedNormalChunk
from src.training.reproducibility import stable_seed


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_epoch(epoch: Any) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, Integral):
        raise TypeError("epoch must be a non-negative integer")
    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    return epoch


def _effective_epoch(split: Any, epoch: Any) -> tuple[str, int]:
    if not isinstance(split, str) or not split.strip():
        raise TypeError("split must be a non-empty string")
    split_name = split.lower()
    validated_epoch = _validate_epoch(epoch)
    return split_name, validated_epoch if split_name == "train" else 0


def _single_object_target_mask(mask: torch.Tensor, object_name: str) -> torch.Tensor:
    """Validate and flatten one object's binary target support mask."""

    mask = _require_tensor(mask, "target mask")
    if mask.ndim == 4:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError("target mask must describe one object as [1, 1, H, W]")
        mask = mask[0, 0]
    elif mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError("target mask must describe one object as [1, H, W]")
        mask = mask[0]
    elif mask.ndim != 2:
        raise ValueError("target mask must have shape [H, W], [1, H, W], or [1, 1, H, W]")
    if mask.numel() == 0:
        raise ValueError(f"target support is empty for {object_name}")
    if mask.dtype == torch.bool:
        binary = mask
    else:
        if mask.is_complex():
            raise TypeError("target mask must use a real floating-point or boolean dtype")
        if not mask.is_floating_point():
            mask = mask.to(dtype=torch.float32)
        if not torch.isfinite(mask).all().item():
            raise ValueError(f"target mask contains non-finite values for {object_name}")
        binary = (mask == 0) | (mask == 1)
        if not binary.all().item():
            raise ValueError(f"target mask must be binary for {object_name}")
        binary = mask > 0
    valid = torch.nonzero(binary.reshape(-1), as_tuple=False).flatten().to(device="cpu")
    if valid.numel() == 0:
        raise ValueError(f"target support is empty for {object_name}")
    return valid


def plan_target_chunks(
    mask: torch.Tensor,
    *,
    pixel_samples: int,
    pixel_budget: int,
    base_seed: int,
    split: str,
    epoch: int,
    object_name: str,
) -> tuple[torch.Tensor, ...]:
    """Create a replayable, disjoint chunk plan from one object's target mask.

    The returned indices are flattened ``H*W`` indices.  Only target-valid
    pixels are eligible; model-support halo pixels never enter the plan.  A
    non-training split uses epoch zero so validation plans remain fixed across
    epochs.
    """

    pixel_samples = _require_positive_integer(pixel_samples, "pixel_samples")
    pixel_budget = _require_positive_integer(pixel_budget, "pixel_budget")
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("base_seed must be an integer")
    if not isinstance(object_name, str) or not object_name:
        raise TypeError("object_name must be a non-empty string")
    split_name, effective_epoch = _effective_epoch(split, epoch)
    valid = _single_object_target_mask(mask, object_name)
    count = min(pixel_budget, int(valid.numel()))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        stable_seed(
            int(base_seed),
            split_name,
            effective_epoch,
            object_name,
            "pixel_grouping",
        )
    )
    selected = valid[torch.randperm(valid.numel(), generator=generator)[:count]]
    return tuple(torch.split(selected, pixel_samples))


def _validate_targets(targets: torch.Tensor) -> tuple[int, int, int]:
    targets = _require_tensor(targets, "targets")
    if targets.ndim != 4 or targets.shape[1] != 3:
        raise ValueError("targets must have shape [B, 3, H, W]")
    batch, _channels, height, width = (int(value) for value in targets.shape)
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("targets must contain at least one non-empty object")
    if not targets.is_floating_point() or targets.is_complex():
        raise TypeError("targets must use a floating-point dtype")
    if not torch.isfinite(targets).all().item():
        raise ValueError("targets contain non-finite values")
    return batch, height, width


def _validate_prediction_chunks(
    predictions: Sequence[Sequence[DecodedNormalChunk]],
    targets: torch.Tensor,
) -> tuple[tuple[tuple[DecodedNormalChunk, ...], ...], tuple[int, int, int]]:
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        raise TypeError("predictions must be a sequence of per-object chunk sequences")
    batch, height, width = _validate_targets(targets)
    if len(predictions) != batch:
        raise ValueError("predictions must contain one chunk sequence per target object")
    pixel_count = height * width
    validated: list[tuple[DecodedNormalChunk, ...]] = []
    for object_number, object_chunks in enumerate(predictions):
        if isinstance(object_chunks, (str, bytes)) or not isinstance(object_chunks, Sequence):
            raise TypeError(f"predictions[{object_number}] must be a sequence")
        if not object_chunks:
            raise ValueError(f"predictions[{object_number}] must contain at least one chunk")
        seen: set[int] = set()
        chunks: list[DecodedNormalChunk] = []
        for chunk_number, chunk in enumerate(object_chunks):
            if not isinstance(chunk, DecodedNormalChunk):
                raise TypeError(
                    f"predictions[{object_number}][{chunk_number}] must be DecodedNormalChunk"
                )
            indices = _require_tensor(
                chunk.indices,
                f"predictions[{object_number}][{chunk_number}].indices",
            )
            prediction = _require_tensor(
                chunk.prediction,
                f"predictions[{object_number}][{chunk_number}].prediction",
            )
            if indices.ndim != 1 or not indices.numel():
                raise ValueError("prediction indices must be one-dimensional and nonempty")
            if indices.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("prediction indices must use an integer dtype")
            if prediction.ndim != 2 or prediction.shape != (indices.numel(), 3):
                raise ValueError("prediction must have shape [len(indices), 3]")
            if not prediction.is_floating_point() or not torch.isfinite(prediction).all().item():
                raise ValueError("prediction must contain finite floating-point values")
            minimum = int(indices.min().item())
            maximum = int(indices.max().item())
            if minimum < 0 or maximum >= pixel_count:
                raise ValueError("prediction indices must be in target range")
            values = [int(value) for value in indices.detach().cpu().tolist()]
            if len(set(values)) != len(values):
                raise ValueError("prediction indices must be unique within each chunk")
            if seen.intersection(values):
                raise ValueError("prediction indices must not overlap across chunks")
            if prediction.device != targets.device:
                raise ValueError("predictions and targets must be on the same device")
            seen.update(values)
            chunks.append(chunk)
        validated.append(tuple(chunks))
    return tuple(validated), (batch, height, width)


def _target_vectors(targets: torch.Tensor, object_number: int) -> torch.Tensor:
    return targets[object_number].permute(1, 2, 0).reshape(-1, 3)


def _normalized_for_reporting(vectors: torch.Tensor, name: str) -> torch.Tensor:
    vectors = vectors.to(dtype=torch.float64)
    if not torch.isfinite(vectors).all().item():
        raise ValueError(f"{name} contains non-finite values")
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if not torch.isfinite(lengths).all().item() or (lengths <= 0).any().item():
        raise ValueError(f"{name} contains zero or non-finite normals")
    return vectors / lengths


def component_sse_batch(
    predictions: Sequence[Sequence[DecodedNormalChunk]],
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return macro-object component squared error for trusted pixel chunks."""

    validated, (_batch, _height, _width) = _validate_prediction_chunks(predictions, targets)
    per_object: list[torch.Tensor] = []
    for object_number, object_chunks in enumerate(validated):
        target = _target_vectors(targets, object_number)
        errors: list[torch.Tensor] = []
        for chunk in object_chunks:
            indices = chunk.indices.to(device=targets.device, dtype=torch.long)
            # The released adapter already returns normalized normals.  Keep
            # those exact values here: this objective is the component-SSE
            # contract, not another normalization/clamping boundary.
            prediction = chunk.prediction
            target_values = target[indices]
            error = (prediction - target_values).square().sum(dim=-1)
            if not torch.isfinite(error).all().item():
                raise ValueError("component squared error is non-finite")
            errors.append(error)
        object_error = torch.cat(errors).mean()
        if not torch.isfinite(object_error).item():
            raise ValueError(f"component squared error is non-finite for object {object_number}")
        per_object.append(object_error)
    result = torch.stack(per_object).mean()
    if not torch.isfinite(result).item():
        raise ValueError("component squared error is non-finite")
    return result.to(dtype=torch.float32)


def sampled_angular_mae(
    predictions: Sequence[Sequence[DecodedNormalChunk]],
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return macro-object angular mean absolute error in degrees."""

    validated, (_batch, _height, _width) = _validate_prediction_chunks(predictions, targets)
    per_object: list[torch.Tensor] = []
    for object_number, object_chunks in enumerate(validated):
        target = _target_vectors(targets, object_number)
        angles: list[torch.Tensor] = []
        for chunk in object_chunks:
            indices = chunk.indices.to(device=targets.device, dtype=torch.long)
            prediction = _normalized_for_reporting(chunk.prediction, "prediction")
            target_values = _normalized_for_reporting(target[indices], "target")
            cosine = (prediction * target_values).sum(dim=-1).clamp(min=-1.0, max=1.0)
            angle = torch.rad2deg(torch.acos(cosine))
            if not torch.isfinite(angle).all().item():
                raise ValueError("angular error is non-finite")
            angles.append(angle)
        object_angle = torch.cat(angles).mean()
        if not torch.isfinite(object_angle).item():
            raise ValueError(f"angular error is non-finite for object {object_number}")
        per_object.append(object_angle)
    result = torch.stack(per_object).mean()
    if not torch.isfinite(result).item():
        raise ValueError("angular MAE is non-finite")
    return result.to(dtype=torch.float32)


def finite_gradients_or_raise(model: torch.nn.Module) -> None:
    """Reject non-finite gradients while allowing unused parameters."""

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
            raise FloatingPointError(f"non-finite gradient: {name}")


__all__ = [
    "component_sse_batch",
    "finite_gradients_or_raise",
    "plan_target_chunks",
    "sampled_angular_mae",
]
