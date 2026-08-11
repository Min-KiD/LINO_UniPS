"""Small real-OpenCV EXR fixtures shared by comparison tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def write_rgb_exr(path: Path, rgb: np.ndarray) -> None:
    """Write an RGB float fixture through OpenCV's BGR EXR interface."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.asarray(rgb, np.float32)[..., ::-1])
    if not ok:
        raise RuntimeError(f"failed to write EXR fixture: {path}")


def write_mask_exr(path: Path, mask: np.ndarray) -> None:
    """Write a one-channel float mask fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.asarray(mask, np.float32))
    if not ok:
        raise RuntimeError(f"failed to write EXR fixture: {path}")


def make_object(root: Path, name: str, image_count: int = 4) -> Path:
    """Create a tiny object directory with observations, normal and mask."""

    object_dir = root / name
    object_dir.mkdir(parents=True, exist_ok=True)
    height, width = 2, 3
    for index in range(image_count):
        image = np.zeros((height, width, 3), np.float32)
        image[..., 0] = index + 0.125
        image[..., 1] = 10.0 + index
        image[..., 2] = 100.0 + index
        write_rgb_exr(object_dir / f"image_{index:03d}.exr", image)

    normal = np.zeros((height, width, 3), np.float32)
    normal[..., 2] = 1.0
    write_rgb_exr(object_dir / "local_normal.exr", normal)

    mask = np.ones((height, width), np.float32)
    write_mask_exr(object_dir / "binary_mask.exr", mask)
    return object_dir
