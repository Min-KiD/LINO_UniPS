"""Differentiable access to the released LiNo_UniPS normal-prediction path.

The adapter deliberately contains no ``nn.Module`` of its own.  It keeps the
released model as the only owner of trainable state and only arranges tensors
for the encoder/decoder calls used by ``LiNo_UniPS.model_step``.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.models.utils.gauss_filter import gauss_filter


_EXPECTED_CHANNELS = 3
_EXPECTED_LIGHTS = 6
_SUPPORTED_GEOMETRIES = frozenset({(512, 256), (256, 128)})
_SMOOTHING_SIGMA = 1
_SMOOTHING_SCALE = 10
_DECODE_CHUNK = 16


def _run_checkpointed(function, *args, enabled: bool):
    if not isinstance(enabled, bool):
        raise TypeError("activation_checkpointing must be a boolean")
    if enabled and torch.is_grad_enabled():
        return torch_checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def _model_device(model: torch.nn.Module) -> torch.device | None:
    """Find the released model execution device without moving any state."""

    declared_device = getattr(model, "device", None)
    if declared_device is not None:
        try:
            return torch.device(declared_device)
        except (TypeError, RuntimeError):
            pass
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return None


def _is_released_lino(model: torch.nn.Module) -> bool:
    model_class = type(model)
    return model_class.__name__ == "LiNo_UniPS" or model_class.__module__.endswith("models.Net_module")


def _has_bfloat16_boundary(model: torch.nn.Module) -> bool:
    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is None:
        return False
    return any(buffer.dtype == torch.bfloat16 for buffer in image_encoder.buffers())


@contextmanager
def _released_dtype_bridge(
    model: torch.nn.Module,
    input_device: torch.device,
):
    """Bridge float32 dataset tensors into the released CUDA BF16 boundary."""

    model_device = _model_device(model) or input_device
    if model_device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield torch.bfloat16
        return
    if _is_released_lino(model) or _has_bfloat16_boundary(model):
        raise RuntimeError(
            "released LiNo_UniPS execution requires CUDA BF16 autocast; "
            "CPU/fp32 execution is unsupported"
        )
    # Small CPU fake components intentionally remain in their native dtype.
    yield None


@dataclass(frozen=True)
class EncodedPrivateBatch:
    """Encoder context retained for differentiable pixel decoding."""

    observations: torch.Tensor
    glc: torch.Tensor
    light_counts: tuple[int, ...]
    height: int
    width: int


@dataclass(frozen=True)
class DecodedNormalChunk:
    """Normal predictions corresponding to one trusted pixel-index chunk."""

    indices: torch.Tensor
    prediction: torch.Tensor


def released_state_schema(model: torch.nn.Module) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Return the released module's registered state without adding adapter state."""

    return tuple(
        (name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    )


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")


def _validate_encoder_inputs(
    observations: torch.Tensor,
    model_mask: torch.Tensor,
    canonical_resolution: int,
) -> tuple[int, int, int]:
    observations = _require_tensor(observations, "observations")
    model_mask = _require_tensor(model_mask, "model_mask")
    if observations.dtype != torch.float32:
        raise TypeError("observations must have dtype torch.float32")
    if observations.ndim != 5:
        raise ValueError("observations must have shape [B, 3, S, S, 6]")
    batch, channels, height, width, lights = observations.shape
    if channels != _EXPECTED_CHANNELS or height != width:
        raise ValueError("observations must have shape [B, 3, S, S, 6]")
    if lights != _EXPECTED_LIGHTS:
        raise ValueError("private adapter requires exactly six lights per object")
    if batch <= 0:
        raise ValueError("observations must contain at least one object")
    if model_mask.ndim != 4 or tuple(model_mask.shape) != (batch, 1, height, width):
        raise ValueError("model_mask must have shape [B, 1, S, S]")
    if model_mask.device != observations.device:
        raise ValueError("observations and model_mask must be on the same device")
    if model_mask.dtype not in (torch.float32, torch.bool):
        raise TypeError("model_mask must have dtype torch.float32 or torch.bool")
    _require_finite(observations, "observations")
    _require_finite(model_mask, "model_mask")
    binary = (model_mask == 0) | (model_mask == 1)
    if not bool(binary.all().item()):
        raise ValueError("model_mask must be binary")
    nonempty = model_mask.reshape(batch, -1).any(dim=1)
    if not bool(nonempty.all().item()):
        raise ValueError("every model_mask must contain at least one foreground pixel")
    if not isinstance(canonical_resolution, Integral) or isinstance(canonical_resolution, bool):
        raise TypeError("canonical_resolution must be an integer")
    canonical_resolution = int(canonical_resolution)
    if (height, canonical_resolution) not in _SUPPORTED_GEOMETRIES:
        raise ValueError(
            "private adapter geometry must be internal/canonical 512/256 or 256/128"
        )
    return batch, height, width


def _smooth_glc(
    glc: torch.Tensor,
    height: int,
    canonical_resolution: int,
    *,
    activation_checkpointing: bool = False,
) -> torch.Tensor:
    f_scale = height // canonical_resolution
    kernel_size = _SMOOTHING_SCALE * f_scale + 1
    smoothing = gauss_filter(glc.shape[1], kernel_size, _SMOOTHING_SIGMA).to(
        device=glc.device,
        dtype=glc.dtype,
    )
    def _smooth_chunk(glc_chunk: torch.Tensor) -> torch.Tensor:
        return smoothing(glc_chunk)

    smoothed = [
        _run_checkpointed(
            _smooth_chunk,
            glc_chunk,
            enabled=activation_checkpointing,
        )
        for glc_chunk in torch.split(glc, _DECODE_CHUNK, dim=0)
    ]
    result = torch.cat(smoothed, dim=0)
    _require_finite(result, "smoothed GLC")
    return result


def encode_private_batch(
    model: torch.nn.Module,
    observations: torch.Tensor,
    model_mask: torch.Tensor,
    canonical_resolution: int = 256,
    *,
    activation_checkpointing: bool = False,
) -> EncodedPrivateBatch:
    """Encode six masked observations per object with the released image encoder."""

    batch, height, width = _validate_encoder_inputs(observations, model_mask, canonical_resolution)
    light_counts = (_EXPECTED_LIGHTS,) * batch
    # LiNo_UniPS receives object-major light rows: [object 0 lights, object 1
    # lights, ...].  Keep this operation differentiable with respect to RGB.
    object_major = observations.permute(0, 4, 1, 2, 3)
    expanded_mask = model_mask.unsqueeze(1)
    encoder_input = (object_major * expanded_mask).reshape(
        batch * _EXPECTED_LIGHTS,
        _EXPECTED_CHANNELS,
        height,
        width,
    )
    with _released_dtype_bridge(model, encoder_input.device) as bridge_dtype:
        released_input = encoder_input if bridge_dtype is None else encoder_input.to(bridge_dtype)

        def _encode_glc(images: torch.Tensor) -> torch.Tensor:
            encoder_output = model.image_encoder(
                images,
                light_counts,
                int(canonical_resolution),
            )
            if not isinstance(encoder_output, (tuple, list)) or not encoder_output:
                raise TypeError(
                    "released image_encoder must return GLC features and auxiliary tokens"
                )
            return _require_tensor(encoder_output[0], "image_encoder GLC output")

        glc = _run_checkpointed(
            _encode_glc,
            released_input,
            enabled=activation_checkpointing,
        )
    if glc.ndim != 4:
        raise ValueError("image_encoder GLC output must have shape [B*6, C, H, W]")
    if glc.shape[0] != batch * _EXPECTED_LIGHTS or tuple(glc.shape[-2:]) != (height, width):
        raise ValueError("image_encoder GLC output must preserve object/light and input geometry")
    if glc.shape[1] <= 0:
        raise ValueError("image_encoder GLC output must have channels")
    _require_finite(glc, "image_encoder GLC output")
    glc = _smooth_glc(
        glc,
        height,
        int(canonical_resolution),
        activation_checkpointing=activation_checkpointing,
    )
    return EncodedPrivateBatch(
        observations=observations,
        glc=glc,
        light_counts=light_counts,
        height=height,
        width=width,
    )


def _validate_encoded_batch(encoded: EncodedPrivateBatch) -> tuple[int, int, int]:
    if not isinstance(encoded, EncodedPrivateBatch):
        raise TypeError("encoded must be an EncodedPrivateBatch")
    observations = _require_tensor(encoded.observations, "encoded.observations")
    glc = _require_tensor(encoded.glc, "encoded.glc")
    if observations.dtype != torch.float32 or observations.ndim != 5:
        raise ValueError("encoded.observations must retain [B, 3, H, W, 6] float32 observations")
    batch, channels, height, width, lights = observations.shape
    if channels != _EXPECTED_CHANNELS or lights != _EXPECTED_LIGHTS:
        raise ValueError("encoded.observations has an invalid channel/light layout")
    if (height, width) != (encoded.height, encoded.width):
        raise ValueError("encoded geometry does not match encoded.observations")
    if glc.ndim != 4 or glc.shape[0] != batch * _EXPECTED_LIGHTS:
        raise ValueError("encoded.glc must have shape [B*6, C, H, W]")
    if tuple(glc.shape[-2:]) != (height, width):
        raise ValueError("encoded.glc geometry does not match encoded.observations")
    if len(encoded.light_counts) != batch or any(
        count != _EXPECTED_LIGHTS for count in encoded.light_counts
    ):
        raise ValueError("encoded.light_counts must contain six lights for every object")
    if observations.device != glc.device:
        raise ValueError("encoded observations and GLC must be on the same device")
    _require_finite(observations, "encoded.observations")
    _require_finite(glc, "encoded.glc")
    return batch, height, width


def _validate_index_chunks(
    index_chunks: Sequence[Sequence[torch.Tensor]],
    batch: int,
    pixel_count: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if isinstance(index_chunks, (str, bytes)) or not isinstance(index_chunks, Sequence):
        raise TypeError("index_chunks must be a sequence of per-object chunk sequences")
    if len(index_chunks) != batch:
        raise ValueError("index_chunks must contain one chunk sequence per object")
    validated: list[tuple[torch.Tensor, ...]] = []
    for object_number, object_chunks in enumerate(index_chunks):
        if isinstance(object_chunks, (str, bytes)) or not isinstance(object_chunks, Sequence):
            raise TypeError(f"index_chunks[{object_number}] must be a sequence")
        seen: set[int] = set()
        chunks: list[torch.Tensor] = []
        for chunk_number, raw_indices in enumerate(object_chunks):
            indices = _require_tensor(raw_indices, f"index_chunks[{object_number}][{chunk_number}]")
            if indices.ndim != 1:
                raise ValueError("trusted indices must be one-dimensional")
            if indices.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("trusted indices must use an integer dtype")
            if not indices.numel():
                raise ValueError("trusted index chunks must be nonempty")
            minimum = int(indices.min().item())
            maximum = int(indices.max().item())
            if minimum < 0 or maximum >= pixel_count:
                raise ValueError("trusted indices must be in range")
            values = [int(value) for value in indices.detach().cpu().tolist()]
            if len(set(values)) != len(values):
                raise ValueError("trusted indices must be unique within each object")
            if seen.intersection(values):
                raise ValueError("trusted indices must not overlap across chunks")
            seen.update(values)
            chunks.append(indices)
        validated.append(tuple(chunks))
    return tuple(validated)


def _decode_one_chunk(
    model: torch.nn.Module,
    object_observations: torch.Tensor,
    object_glc: torch.Tensor,
    indices: torch.Tensor,
    *,
    activation_checkpointing: bool = False,
) -> DecodedNormalChunk:
    device_indices = indices.to(device=object_observations.device, dtype=torch.long)
    if not indices.numel():
        raise ValueError("trusted index chunks must be nonempty")

    def _decode_chunk_tensor(
        observations: torch.Tensor,
        glc: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        observation_pixels = observations[selected_indices]
        glc_pixels = glc[selected_indices]
        _require_finite(observation_pixels, "observation pixels")
        _require_finite(glc_pixels, "GLC pixels")
        with _released_dtype_bridge(model, observations.device) as bridge_dtype:
            if bridge_dtype is None:
                released_observations = observation_pixels
                released_glc = glc_pixels
            else:
                released_observations = observation_pixels.to(bridge_dtype)
                released_glc = glc_pixels.to(bridge_dtype)
            embedded = model.img_embedding(released_observations)
            if not isinstance(embedded, torch.Tensor):
                raise TypeError("img_embedding must return a tensor")
            _require_finite(embedded, "embedded observation pixels")
            features = embedded + released_glc
            _require_finite(features, "embedded plus GLC features")
            features = model.glc_upsample(features)
            if not isinstance(features, torch.Tensor):
                raise TypeError("glc_upsample must return a tensor")
            _require_finite(features, "upsampled GLC features")
            features = embedded + features
            _require_finite(features, "residual GLC features")
            features = model.glc_aggregation(features)
            if not isinstance(features, torch.Tensor):
                raise TypeError("glc_aggregation must return a tensor")
            _require_finite(features, "aggregated GLC features")
            regressor_output = model.regressor(features, len(selected_indices))
            if not isinstance(regressor_output, (tuple, list)) or not regressor_output:
                raise TypeError("released regressor must return normal and auxiliary outputs")
            raw_normal = _require_tensor(regressor_output[0], "regressor normal output")
            _require_finite(raw_normal, "regressor normal output")
            for auxiliary_number, auxiliary in enumerate(regressor_output[1:], start=1):
                if isinstance(auxiliary, torch.Tensor):
                    _require_finite(auxiliary, f"regressor auxiliary output {auxiliary_number}")
            prediction = F.normalize(raw_normal.reshape(-1, 3), p=2, dim=-1, eps=1.0e-6)
            if prediction.shape != (selected_indices.numel(), 3):
                raise ValueError("regressor normal output does not match trusted index count")
            _require_finite(prediction, "normal prediction")
            return prediction

    prediction = _run_checkpointed(
        _decode_chunk_tensor,
        object_observations,
        object_glc,
        device_indices,
        enabled=activation_checkpointing,
    )
    return DecodedNormalChunk(indices=device_indices, prediction=prediction)


def decode_private_chunks(
    model: torch.nn.Module,
    encoded: EncodedPrivateBatch,
    index_chunks: Sequence[Sequence[torch.Tensor]],
    *,
    activation_checkpointing: bool = False,
) -> tuple[tuple[DecodedNormalChunk, ...], ...]:
    """Decode trusted pixel chunks through the released normal-prediction chain."""

    batch, height, width = _validate_encoded_batch(encoded)
    validated = _validate_index_chunks(index_chunks, batch, height * width)
    observations = encoded.observations
    glc = encoded.glc
    decoded: list[tuple[DecodedNormalChunk, ...]] = []
    for object_number, object_chunks in enumerate(validated):
        object_observations = observations[object_number].permute(1, 2, 3, 0).reshape(height * width, 6, 3)
        light_start = object_number * _EXPECTED_LIGHTS
        object_glc = (
            glc[light_start : light_start + _EXPECTED_LIGHTS]
            .permute(2, 3, 0, 1)
            .reshape(height * width, 6, glc.shape[1])
        )
        decoded.append(
            tuple(
                _decode_one_chunk(
                    model,
                    object_observations,
                    object_glc,
                    indices,
                    activation_checkpointing=activation_checkpointing,
                )
                for indices in object_chunks
            )
        )
    return tuple(decoded)
