"""Contract tests for the epoch-aware private EXR training adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.training.config import PrivateTrainConfig
from src.training.private_manifest import (
    PrivateSplitManifest,
    build_private_split_manifest,
)
from tests.comparison_helpers import write_mask_exr, write_rgb_exr

from src.data.private_exr_train import EXPECTED_FIELDS, PrivateExrTrainDataset, collate_private_exr


class PrivateExrTrainDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.train_root = self.root / "train"
        self.test_root = self.root / "test"
        self.train_root.mkdir()
        self.test_root.mkdir()

    def config(self, **overrides: object) -> PrivateTrainConfig:
        values: dict[str, object] = {
            "train_dir": self.train_root,
            "test_dir": self.test_root,
            "save_dir": self.root / "runs",
            "startup_mode": "cold_start",
            "init_checkpoint": None,
            "resume_checkpoint": None,
            "final_selection_manifest": self.root / "selection.json",
            "object_suffix": ".data",
            "image_prefix": "image",
            "image_extension": ".exr",
            "normal_filenames": ("local_normal.exr",),
            "external_mask_filename": "binary_mask.exr",
            "normal_encoding": "unsigned",
            "expected_source_geometry": (256, 256),
            "mask_policy": "external",
            "mask_margin": 8,
            "max_image_num": 6,
            "light_selection": "seeded",
            "seed": 20260710,
            "preprocessing_version": "private_external_lino_native_v1",
            "max_image_resolution": 512,
            "canonical_resolution": 256,
            "pixel_samples": 16,
            "train_pixel_budget": 1024,
            "precision": "fp32",
            "device": "cpu",
            "deterministic": True,
            "epochs": 4,
            "train_batch_size": 1,
            "train_workers": 0,
            "test_workers": 0,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "adamw_betas": (0.9, 0.98),
            "scheduler_step_size": 1,
            "scheduler_gamma": 0.5,
            "save_every_epochs": 1,
            "keep_milestone_epochs": (1,),
        }
        values.update(overrides)
        return PrivateTrainConfig(**values)  # type: ignore[arg-type]

    @staticmethod
    def _write_object(root: Path, name: str, *, image_count: int = 8) -> Path:
        object_dir = root / name
        object_dir.mkdir(parents=True)
        rows, cols = np.indices((256, 256), dtype=np.float32)
        for index in range(image_count):
            image = np.empty((256, 256, 3), dtype=np.float32)
            image[..., 0] = 0.5 + index + rows / 256.0
            image[..., 1] = 1.0 + index + cols / 256.0
            image[..., 2] = 2.0 + index
            write_rgb_exr(object_dir / f"image_{index:03d}.exr", image)

        encoded_normal = np.full((256, 256, 3), 0.5, dtype=np.float32)
        encoded_normal[..., 2] = 1.0
        encoded_normal[:32, :, :] = 0.5
        encoded_normal[224:, :, :] = 0.5
        encoded_normal[:, :32, :] = 0.5
        encoded_normal[:, 224:, :] = 0.5
        write_rgb_exr(object_dir / "local_normal.exr", encoded_normal)

        external_mask = np.zeros((256, 256), dtype=np.float32)
        external_mask[24:232, 24:232] = 1.0
        write_mask_exr(object_dir / "binary_mask.exr", external_mask)
        return object_dir

    def manifest(self, split: str = "train", *, name: str = "alpha.data") -> tuple[PrivateTrainConfig, PrivateSplitManifest]:
        config = self.config()
        root = self.train_root if split == "train" else self.test_root
        self._write_object(root, name)
        return config, build_private_split_manifest(config, split)

    def test_train_sample_has_six_epoch_selected_lights_and_separate_masks(self):
        config, manifest = self.manifest()
        dataset = PrivateExrTrainDataset(config, manifest, split="train")
        dataset.set_epoch(3)
        sample = dataset[0]
        self.assertEqual(set(sample), EXPECTED_FIELDS)
        self.assertEqual(tuple(sample["imgs"].shape), (3, 512, 512, 6))
        self.assertEqual(tuple(sample["model_mask"].shape), (1, 512, 512))
        self.assertEqual(tuple(sample["target_normal"].shape), (3, 512, 512))
        self.assertEqual(tuple(sample["target_mask"].shape), (1, 512, 512))
        self.assertGreater(sample["model_mask"].sum().item(), sample["target_mask"].sum().item())
        self.assertEqual(sample["metadata"]["epoch"], 3)
        self.assertEqual(sample["metadata"]["split"], "train")
        self.assertEqual(len(sample["metadata"]["selected_images"]), 6)
        self.assertEqual(len(sample["metadata"]["selected_image_sha256"]), 6)
        self.assertEqual(sample["imgs"].dtype, torch.float32)
        self.assertTrue(sample["imgs"].is_contiguous())

    def test_train_selection_and_normalization_vary_by_epoch(self):
        config, manifest = self.manifest()
        dataset = PrivateExrTrainDataset(config, manifest, split="train")
        dataset.set_epoch(0)
        first = dataset[0]
        dataset.set_epoch(1)
        second = dataset[0]
        self.assertNotEqual(first["metadata"]["selected_images"], second["metadata"]["selected_images"])
        self.assertFalse(torch.equal(first["imgs"], second["imgs"]))

    def test_validation_selection_is_epoch_invariant(self):
        config, manifest = self.manifest(split="test")
        dataset = PrivateExrTrainDataset(config, manifest, split="test")
        dataset.set_epoch(0)
        first = dataset[0]["metadata"]["selected_images"]
        dataset.set_epoch(90)
        second = dataset[0]["metadata"]["selected_images"]
        self.assertEqual(first, second)
        self.assertEqual(dataset[0]["metadata"]["epoch"], 0)

    def test_metadata_is_complete_and_json_safe(self):
        config, manifest = self.manifest()
        sample = PrivateExrTrainDataset(config, manifest, split="train")[0]
        metadata = sample["metadata"]
        json.dumps(metadata, allow_nan=False)
        for key in (
            "object_name",
            "split",
            "epoch",
            "selected_images",
            "selected_image_sha256",
            "roi",
            "source_geometry",
            "resized_geometry",
            "mask_counts",
            "normalization",
            "normalization_alpha",
            "normalization_scales",
            "normalization_seed",
            "preprocessing_version",
        ):
            self.assertIn(key, metadata)

    def test_collator_preserves_metadata_order_and_float32_targets(self):
        def sample(name: str) -> dict[str, object]:
            return {
                "imgs": torch.zeros((3, 512, 512, 6), dtype=torch.float32),
                "model_mask": torch.ones((1, 512, 512), dtype=torch.float32),
                "target_normal": torch.zeros((3, 512, 512), dtype=torch.float32),
                "target_mask": torch.ones((1, 512, 512), dtype=torch.float32),
                "source_target_normal": torch.zeros((3, 256, 256), dtype=torch.float32),
                "source_target_mask": torch.ones((1, 256, 256), dtype=torch.float32),
                "source_model_mask": torch.ones((1, 256, 256), dtype=torch.float32),
                "roi": torch.tensor([256, 256, 0, 256, 0, 256], dtype=torch.int64),
                "metadata": {"object_name": name},
            }

        batch = collate_private_exr([sample("a.data"), sample("b.data")])
        self.assertEqual(tuple(batch["imgs"].shape), (2, 3, 512, 512, 6))
        self.assertEqual([item["object_name"] for item in batch["metadata"]], ["a.data", "b.data"])
        self.assertEqual(batch["target_normal"].dtype, torch.float32)
        self.assertEqual(batch["roi"].dtype, torch.int64)

    def test_collator_rejects_empty_missing_extra_and_incompatible_batches(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            collate_private_exr([])
        good = {
            "imgs": torch.zeros((3, 1, 1, 1)),
            "model_mask": torch.ones((1, 1, 1)),
            "target_normal": torch.zeros((3, 1, 1)),
            "target_mask": torch.ones((1, 1, 1)),
            "source_target_normal": torch.zeros((3, 1, 1)),
            "source_target_mask": torch.ones((1, 1, 1)),
            "source_model_mask": torch.ones((1, 1, 1)),
            "roi": torch.zeros((6,), dtype=torch.int64),
            "metadata": {},
        }
        missing = dict(good)
        del missing["target_mask"]
        with self.assertRaisesRegex(ValueError, "fields"):
            collate_private_exr([missing])
        extra = dict(good, unexpected=torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "fields"):
            collate_private_exr([extra])
        incompatible = dict(good, imgs=torch.zeros((3, 2, 1, 1)))
        with self.assertRaises((RuntimeError, ValueError)):
            collate_private_exr([good, incompatible])

    def test_set_epoch_rejects_bool_negative_and_non_integer_values(self):
        config, manifest = self.manifest()
        dataset = PrivateExrTrainDataset(config, manifest, split="train")
        for value in (True, -1, 1.0, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "epoch"):
                dataset.set_epoch(value)  # type: ignore[arg-type]

    def test_mutated_selected_observation_digest_is_rejected_before_tensor_creation(self):
        config, manifest = self.manifest()
        record = manifest.objects[0]
        selected = record.observation_files[0]
        changed = np.full((256, 256, 3), 99.0, dtype=np.float32)
        write_rgb_exr(self.train_root / record.name / selected, changed)
        with self.assertRaisesRegex(ValueError, "digest"):
            PrivateExrTrainDataset(config, manifest, split="train")[0]

    def test_mutated_ground_truth_digest_is_rejected_before_tensor_creation(self):
        config, manifest = self.manifest()
        record = manifest.objects[0]
        changed = np.full((256, 256, 3), 0.5, dtype=np.float32)
        write_rgb_exr(self.train_root / record.name / record.normal_file, changed)
        with self.assertRaisesRegex(ValueError, "digest"):
            PrivateExrTrainDataset(config, manifest, split="train")[0]

    def test_mutated_external_mask_digest_is_rejected_before_tensor_creation(self):
        config, manifest = self.manifest()
        record = manifest.objects[0]
        changed = np.zeros((256, 256), dtype=np.float32)
        changed[48:208, 48:208] = 1.0
        write_mask_exr(self.train_root / record.name / record.mask_file, changed)
        with self.assertRaisesRegex(ValueError, "digest"):
            PrivateExrTrainDataset(config, manifest, split="train")[0]

    def test_source_and_target_masks_remain_separate(self):
        config, manifest = self.manifest()
        sample = PrivateExrTrainDataset(config, manifest, split="train")[0]
        self.assertGreater(
            sample["source_model_mask"].sum().item(),
            sample["source_target_mask"].sum().item(),
        )
        self.assertGreater(
            sample["model_mask"].sum().item(),
            sample["target_mask"].sum().item(),
        )


if __name__ == "__main__":
    unittest.main()
