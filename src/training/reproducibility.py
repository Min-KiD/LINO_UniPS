"""Utilities for deterministic private-training random-number generation."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Sequence

import numpy as np
import torch


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable unsigned 32-bit seed from JSON-compatible parts."""

    payload = json.dumps(
        [base_seed, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def select_observation_names(
    available: Sequence[str],
    *,
    count: int,
    base_seed: int,
    split: str,
    epoch: int,
    object_name: str,
) -> tuple[str, ...]:
    """Select a deterministic subset of observations for one object and epoch."""

    if count <= 0 or len(available) < count:
        raise ValueError(f"{object_name} requires {count} observations; found {len(available)}")
    split_name = split.lower()
    effective_epoch = int(epoch) if split_name == "train" else 0
    generator = np.random.default_rng(
        stable_seed(base_seed, split_name, effective_epoch, object_name, "light_selection")
    )
    order = generator.permutation(len(available))[:count]
    return tuple(available[int(index)] for index in order)


def epoch_permutation(length: int, *, base_seed: int, epoch: int) -> list[int]:
    """Return a deterministic complete permutation of dataset indices."""

    if length <= 0:
        raise ValueError("dataset length must be positive")
    generator = np.random.default_rng(np.random.SeedSequence([int(base_seed), int(epoch)]))
    return [int(index) for index in generator.permutation(length)]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch, optionally requesting deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a PyTorch DataLoader worker process."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


__all__ = [
    "epoch_permutation",
    "seed_everything",
    "seed_worker",
    "select_observation_names",
    "stable_seed",
]
