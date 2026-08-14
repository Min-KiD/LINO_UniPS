"""Tests for deterministic private-training scheduling and seeding."""

from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from torch.utils.data import DataLoader, Dataset

from src.training.reproducibility import (
    epoch_permutation,
    seed_everything,
    seed_worker,
    select_observation_names,
    stable_seed,
)


class _ScheduleDataset(Dataset):
    """Picklable dataset that evaluates one explicit schedule per item."""

    def __init__(
        self,
        object_names: tuple[str, ...],
        available: tuple[str, ...],
        *,
        count: int,
        base_seed: int,
        split: str,
        epoch: int,
    ) -> None:
        self.object_names = object_names
        self.available = available
        self.count = count
        self.base_seed = base_seed
        self.split = split
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.object_names)

    def __getitem__(self, index: int) -> tuple[int, str, tuple[str, ...]]:
        object_name = self.object_names[index]
        schedule = select_observation_names(
            self.available,
            count=self.count,
            base_seed=self.base_seed,
            split=self.split,
            epoch=self.epoch,
            object_name=object_name,
        )
        return index, object_name, schedule


def _take_single(batch: list[tuple[int, str, tuple[str, ...]]]):
    if len(batch) != 1:
        raise ValueError(f"expected one sample, found {len(batch)}")
    return batch[0]


def reference_seed(base_seed: int, *parts: object) -> int:
    payload = json.dumps(
        [base_seed, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


class PrivateTrainingReproducibilityTests(unittest.TestCase):
    def test_seed_matches_corrected_sdm_algorithm(self):
        actual = stable_seed(20260710, "train", 7, "alpha.data", "light_selection")
        self.assertEqual(
            actual,
            reference_seed(20260710, "train", 7, "alpha.data", "light_selection"),
        )

    def test_train_lights_change_by_epoch_but_validation_does_not(self):
        names = tuple(f"image{i:03d}.exr" for i in range(20))
        train0 = select_observation_names(
            names,
            count=6,
            base_seed=20260710,
            split="train",
            epoch=0,
            object_name="alpha.data",
        )
        train1 = select_observation_names(
            names,
            count=6,
            base_seed=20260710,
            split="train",
            epoch=1,
            object_name="alpha.data",
        )
        val0 = select_observation_names(
            names,
            count=6,
            base_seed=20260710,
            split="test",
            epoch=0,
            object_name="alpha.data",
        )
        val9 = select_observation_names(
            names,
            count=6,
            base_seed=20260710,
            split="test",
            epoch=9,
            object_name="alpha.data",
        )
        self.assertNotEqual(train0, train1)
        self.assertEqual(val0, val9)

    def test_epoch_permutation_matches_seed_sequence_vector_and_changes_by_epoch(self):
        first = epoch_permutation(11, base_seed=20260710, epoch=4)
        expected = [10, 3, 8, 4, 0, 7, 9, 5, 2, 6, 1]
        independent = np.random.default_rng(
            np.random.SeedSequence([20260710, 4])
        ).permutation(11).tolist()

        self.assertEqual(expected, independent)
        self.assertEqual(first, expected)
        self.assertEqual(first, epoch_permutation(11, base_seed=20260710, epoch=4))
        self.assertEqual(sorted(first), list(range(11)))
        self.assertNotEqual(first, epoch_permutation(11, base_seed=20260710, epoch=5))

    def test_non_train_split_names_are_epoch_invariant(self):
        names = tuple(f"image{i:03d}.exr" for i in range(20))
        for split in ("test", "TEST", "val", "Validation", "infer"):
            with self.subTest(split=split):
                first = select_observation_names(
                    names,
                    count=6,
                    base_seed=20260710,
                    split=split,
                    epoch=0,
                    object_name="alpha.data",
                )
                later = select_observation_names(
                    names,
                    count=6,
                    base_seed=20260710,
                    split=split,
                    epoch=9,
                    object_name="alpha.data",
                )
                self.assertEqual(first, later)

    def test_schedule_is_independent_of_dataloader_worker_count(self):
        object_names = (
            "alpha.data",
            "beta.data",
            "gamma.data",
            "delta.data",
            "epsilon.data",
        )
        available = tuple(f"image{i:03d}.exr" for i in range(20))

        def collect(num_workers: int) -> dict[int, tuple[str, tuple[str, ...]]]:
            dataset = _ScheduleDataset(
                object_names,
                available,
                count=6,
                base_seed=20260710,
                split="train",
                epoch=4,
            )
            results = {}
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=_take_single,
            )
            for index, object_name, schedule in loader:
                results[int(index)] = (object_name, tuple(schedule))
            return results

        single_process = collect(num_workers=0)
        multi_process = collect(num_workers=2)
        self.assertEqual(single_process, multi_process)

    def test_selection_rejects_nonpositive_or_insufficient_count(self):
        names = ("image000.exr", "image001.exr")
        for count in (0, 3):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, r"alpha\.data requires"
            ):
                select_observation_names(
                    names,
                    count=count,
                    base_seed=20260710,
                    split="train",
                    epoch=0,
                    object_name="alpha.data",
                )

    def test_epoch_permutation_rejects_empty_dataset(self):
        with self.assertRaisesRegex(ValueError, "dataset length must be positive"):
            epoch_permutation(0, base_seed=20260710, epoch=0)


class SeedEverythingTests(unittest.TestCase):
    @staticmethod
    def _fake_dependencies():
        python_random = SimpleNamespace(seed=Mock())
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        torch_module = SimpleNamespace(
            manual_seed=Mock(),
            cuda=SimpleNamespace(manual_seed_all=Mock()),
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(
                    benchmark=True,
                    deterministic=False,
                    allow_tf32=True,
                ),
                cuda=SimpleNamespace(
                    matmul=SimpleNamespace(allow_tf32=True),
                ),
            ),
            use_deterministic_algorithms=Mock(),
        )
        return python_random, numpy, torch_module

    def test_seed_everything_seeds_all_rngs(self):
        python_random, numpy, torch_module = self._fake_dependencies()
        with (
            patch("src.training.reproducibility.random", python_random),
            patch("src.training.reproducibility.np", numpy),
            patch("src.training.reproducibility.torch", torch_module),
        ):
            seed_everything(20260710)

        python_random.seed.assert_called_once_with(20260710)
        numpy.random.seed.assert_called_once_with(20260710)
        torch_module.manual_seed.assert_called_once_with(20260710)
        torch_module.cuda.manual_seed_all.assert_called_once_with(20260710)
        torch_module.use_deterministic_algorithms.assert_not_called()

    def test_seed_everything_configures_deterministic_backends_when_requested(self):
        python_random, numpy, torch_module = self._fake_dependencies()
        with (
            patch("src.training.reproducibility.random", python_random),
            patch("src.training.reproducibility.np", numpy),
            patch("src.training.reproducibility.torch", torch_module),
        ):
            seed_everything(7, deterministic=True)

        self.assertFalse(torch_module.backends.cudnn.benchmark)
        self.assertTrue(torch_module.backends.cudnn.deterministic)
        self.assertFalse(torch_module.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch_module.backends.cudnn.allow_tf32)
        torch_module.use_deterministic_algorithms.assert_called_once_with(
            True,
            warn_only=True,
        )


class SeedWorkerTests(unittest.TestCase):
    def test_seed_worker_uses_torch_initial_seed_modulo_numpy_range(self):
        python_random = SimpleNamespace(seed=Mock())
        numpy = SimpleNamespace(random=SimpleNamespace(seed=Mock()))
        torch_module = SimpleNamespace(initial_seed=Mock(return_value=2**32 + 123))

        with (
            patch("src.training.reproducibility.random", python_random),
            patch("src.training.reproducibility.np", numpy),
            patch("src.training.reproducibility.torch", torch_module),
        ):
            seed_worker(worker_id=4)

        torch_module.initial_seed.assert_called_once_with()
        python_random.seed.assert_called_once_with(123)
        numpy.random.seed.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
