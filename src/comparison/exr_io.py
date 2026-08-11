"""Strict, dependency-light OpenCV readers and writers for EXR normals."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

# OpenCV only enables its OpenEXR codec when this switch is set before the
# first import.  Assignment (rather than setdefault) keeps a caller's stale
# disabled value from silently changing the comparison contract.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_raw(path: str | Path, *, label: str) -> np.ndarray:
    source = _as_path(path)
    raw = cv2.imread(str(source), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise ValueError(f"failed to read {label} EXR: {source}")
    if not isinstance(raw, np.ndarray) or raw.size == 0:
        raise ValueError(f"{label} EXR is empty: {source}")
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{label} EXR must decode to floating-point data: {source}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{label} EXR contains non-finite values: {source}")
    return np.asarray(raw, dtype=np.float32)


def _read_raw_bytes(payload: bytes, *, label: str) -> np.ndarray:
    """Decode one immutable EXR byte snapshot without reopening a pathname."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"{label} EXR bytes must be a bytes-like value")
    encoded = np.frombuffer(payload, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"{label} EXR is empty")
    raw = cv2.imdecode(encoded, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if raw is None:
        raise ValueError(f"failed to decode {label} EXR bytes")
    if not isinstance(raw, np.ndarray) or raw.size == 0:
        raise ValueError(f"{label} EXR is empty")
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{label} EXR must decode to floating-point data")
    if not np.isfinite(raw).all():
        raise ValueError(f"{label} EXR contains non-finite values")
    return np.asarray(raw, dtype=np.float32)


def _require_rgb(raw: np.ndarray, path: str | Path, *, label: str) -> np.ndarray:
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"{label} EXR must be a nonempty three-channel array: {_as_path(path)}")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(f"{label} EXR must be a nonempty three-channel array: {_as_path(path)}")
    # OpenCV decodes color images as BGR; make an explicit contiguous RGB
    # float32 array so callers cannot accidentally depend on channel strides.
    return np.ascontiguousarray(raw[..., ::-1], dtype=np.float32)


def read_rgb_exr(path: str | Path) -> np.ndarray:
    """Read a finite, nonempty three-channel EXR and return RGB float32 data."""

    source = _as_path(path)
    return _require_rgb(_read_raw(source, label="RGB"), source, label="RGB")


def read_signed_normal_exr(path: str | Path) -> np.ndarray:
    """Read a signed RGB normal EXR as finite RGB float32 data."""

    source = _as_path(path)
    return _require_rgb(_read_raw(source, label="signed normal"), source, label="signed normal")


def read_rgb_exr_bytes(payload: bytes, *, label: str = "RGB") -> np.ndarray:
    """Read a finite RGB EXR from one immutable byte snapshot."""

    return _require_rgb(_read_raw_bytes(payload, label=label), label, label=label)


def read_file_bytes(path: str | Path, *, label: str = "file") -> bytes:
    """Read one immutable regular-file snapshot without following symlinks."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"secure {label} reads require O_NOFOLLOW")
    source = _as_path(path)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(str(source), flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {source}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {source}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {label}: {source}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def read_signed_normal_exr_bytes(
    payload: bytes,
    *,
    label: str = "signed normal",
) -> np.ndarray:
    """Read signed RGB normals from one immutable EXR byte snapshot."""

    return _require_rgb(_read_raw_bytes(payload, label=label), label, label=label)


def read_mask_exr(path: str | Path) -> np.ndarray:
    """Read a one- or three-channel EXR mask and threshold it at zero."""

    source = _as_path(path)
    raw = _read_raw(source, label="mask")
    if raw.ndim == 2:
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError(f"mask EXR must be a nonempty one- or three-channel array: {source}")
        channel = raw
    elif raw.ndim == 3 and raw.shape[2] in (1, 3):
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError(f"mask EXR must be a nonempty one- or three-channel array: {source}")
        # OpenCV decodes color EXRs as BGR.  SDM's mask reader consumes the
        # EXR red channel, so select decoded channel 2 explicitly rather than
        # accidentally using blue channel 0.
        channel = raw[..., 0] if raw.shape[2] == 1 else raw[..., 2]
    else:
        raise ValueError(f"mask EXR must be a nonempty one- or three-channel array: {source}")
    return (channel > 0).astype(np.float32, copy=False)


def read_mask_exr_bytes(payload: bytes, *, label: str = "mask") -> np.ndarray:
    """Read and threshold one- or three-channel masks from immutable bytes."""

    raw = _read_raw_bytes(payload, label=label)
    if raw.ndim == 2:
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError(f"{label} EXR must be a nonempty one- or three-channel array")
        channel = raw
    elif raw.ndim == 3 and raw.shape[2] in (1, 3):
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError(f"{label} EXR must be a nonempty one- or three-channel array")
        channel = raw[..., 0] if raw.shape[2] == 1 else raw[..., 2]
    else:
        raise ValueError(f"{label} EXR must be a nonempty one- or three-channel array")
    return (channel > 0).astype(np.float32, copy=False)


def _normal_array(normal: Any, *, path: str | Path | None = None) -> np.ndarray:
    try:
        array = np.asarray(normal)
    except (TypeError, ValueError) as exc:
        where = f": {_as_path(path)}" if path is not None else ""
        raise ValueError(f"normal must be a numeric three-channel array{where}") from exc
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        where = f": {_as_path(path)}" if path is not None else ""
        raise ValueError(f"normal must be a numeric three-channel array{where}")
    if array.ndim != 3 or array.shape[2] != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        where = f": {_as_path(path)}" if path is not None else ""
        raise ValueError(f"normal must be a nonempty three-channel array{where}")
    converted = np.asarray(array, dtype=np.float32)
    if not np.isfinite(converted).all():
        where = f": {_as_path(path)}" if path is not None else ""
        raise ValueError(f"normal contains non-finite values{where}")
    return converted


def _write(path: str | Path, image: np.ndarray, *, label: str) -> None:
    destination = _as_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(destination), image)
    if not ok:
        raise RuntimeError(f"failed to write {label}: {destination}")


def write_normal_exr(path: str | Path, normal: np.ndarray) -> None:
    """Write signed RGB normals as float32 BGR EXR data."""

    destination = _as_path(path)
    array = _normal_array(normal, path=destination)
    _write(destination, np.ascontiguousarray(array[..., ::-1], dtype=np.float32), label="normal EXR")


def encode_normal_exr(normal: np.ndarray) -> bytes:
    """Encode signed RGB normals as float32 BGR EXR bytes."""

    array = _normal_array(normal)
    ok, encoded = cv2.imencode(
        ".exr",
        np.ascontiguousarray(array[..., ::-1], dtype=np.float32),
    )
    if not ok or encoded is None or encoded.size == 0:
        raise RuntimeError("failed to encode normal EXR")
    return encoded.tobytes()


def write_normal_png(path: str | Path, normal: np.ndarray) -> None:
    """Write a display-only uint8 preview of signed RGB normals."""

    destination = _as_path(path)
    array = _normal_array(normal, path=destination)
    preview = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    quantized = np.asarray(preview * 255.0, dtype=np.uint8)
    _write(destination, np.ascontiguousarray(quantized[..., ::-1]), label="normal PNG")


def encode_normal_png(normal: np.ndarray) -> bytes:
    """Encode display-only signed RGB normals as uint8 PNG bytes."""

    array = _normal_array(normal)
    preview = np.clip((array + 1.0) / 2.0, 0.0, 1.0)
    quantized = np.asarray(preview * 255.0, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(quantized[..., ::-1]))
    if not ok or encoded is None or encoded.size == 0:
        raise RuntimeError("failed to encode normal PNG")
    return encoded.tobytes()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    source = _as_path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "read_rgb_exr",
    "read_rgb_exr_bytes",
    "read_file_bytes",
    "read_mask_exr",
    "read_mask_exr_bytes",
    "read_signed_normal_exr",
    "read_signed_normal_exr_bytes",
    "write_normal_exr",
    "encode_normal_exr",
    "write_normal_png",
    "encode_normal_png",
    "sha256_file",
]
