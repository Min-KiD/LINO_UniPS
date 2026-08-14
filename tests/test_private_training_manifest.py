"""Tests for explicit private-training split manifests and source preflight."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.training.config import PrivateTrainConfig
from src.training import private_manifest
from src.training.private_manifest import (
    build_private_split_manifest,
    private_manifest_bytes,
    private_manifest_sha256,
)
from tests.comparison_helpers import write_mask_exr, write_rgb_exr


class PrivateTrainingManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_root = self.root / "train"
        self.test_root = self.root / "test"
        self.train_root.mkdir()
        self.test_root.mkdir()

    def config(self, **overrides: object) -> PrivateTrainConfig:
        values: dict[str, object] = dict(
            train_dir=self.train_root,
            test_dir=self.test_root,
            save_dir=self.root / "runs",
            startup_mode="cold_start",
            init_checkpoint=None,
            resume_checkpoint=None,
            final_selection_manifest=self.root / "runs" / "selection.json",
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            normal_filenames=("local_normal.exr",),
            external_mask_filename="binary_mask.exr",
            normal_encoding="unsigned",
            expected_source_geometry=(256, 256),
            mask_policy="external",
            mask_margin=8,
            max_image_num=6,
            light_selection="seeded",
            seed=20260710,
            preprocessing_version="private_external_lino_native_v1",
            max_image_resolution=512,
            canonical_resolution=256,
            pixel_samples=1,
            train_pixel_budget=1024,
            precision="fp32",
            device="cpu",
            deterministic=True,
            epochs=1,
            train_batch_size=1,
            train_workers=0,
            test_workers=0,
            learning_rate=1.0e-4,
            weight_decay=0.0,
            adamw_betas=(0.9, 0.98),
            scheduler_step_size=1,
            scheduler_gamma=0.5,
            save_every_epochs=1,
            keep_milestone_epochs=(1,),
        )
        values.update(overrides)
        return PrivateTrainConfig(**values)  # type: ignore[arg-type]

    def _write_object(
        self,
        name: str,
        *,
        root: Path | None = None,
        observation_names: list[str] | tuple[str, ...] | None = None,
        gt_box: tuple[int, int, int, int] | None = (2, 2, 5, 5),
        mask_box: tuple[int, int, int, int] | None = (1, 1, 6, 6),
    ) -> Path:
        object_dir = (self.train_root if root is None else root) / name
        object_dir.mkdir(parents=True)
        names = observation_names or [f"image_{index:02d}.exr" for index in range(8)]
        for index, filename in enumerate(names):
            image = np.full((256, 256, 3), 1.0 + index, dtype=np.float32)
            image[..., 1] = 2.0 + index
            image[..., 2] = 3.0 + index
            write_rgb_exr(object_dir / filename, image)

        encoded_gt = np.full((256, 256, 3), 0.5, dtype=np.float32)
        if gt_box is not None:
            row_start, col_start, row_end, col_end = gt_box
            encoded_gt[row_start:row_end, col_start:col_end, 2] = 1.0
        write_rgb_exr(object_dir / "local_normal.exr", encoded_gt)

        mask = np.zeros((256, 256), dtype=np.float32)
        if mask_box is not None:
            row_start, col_start, row_end, col_end = mask_box
            mask[row_start:row_end, col_start:col_end] = 1.0
        write_mask_exr(object_dir / "binary_mask.exr", mask)
        return object_dir

    def test_manifest_indexes_every_observation_in_sdm_lexicographic_order(self):
        self._write_object(
            "zeta.data",
            observation_names=["image10.exr", "image2.exr", "image1.exr", "image3.exr", "image4.exr", "image5.exr", "image6.exr", "image7.exr"],
        )
        manifest = build_private_split_manifest(self.config(), split="train")
        record = manifest.objects[0]
        self.assertEqual(
            record.observation_files,
            ("image1.exr", "image10.exr", "image2.exr", "image3.exr", "image4.exr", "image5.exr", "image6.exr", "image7.exr"),
        )
        self.assertEqual(len(record.observation_files), len(record.observation_sha256))
        self.assertEqual(record.relative_dir, "zeta.data")

    def test_external_halo_is_allowed(self):
        self._write_object("alpha.data", gt_box=(2, 2, 5, 5), mask_box=(1, 1, 6, 6))
        manifest = build_private_split_manifest(self.config(), split="train")
        record = manifest.objects[0]
        self.assertEqual(record.gt_valid_pixels, 9)
        self.assertEqual(record.mask_valid_pixels, 25)
        self.assertEqual(record.mask_only_pixels, 16)

    def test_gt_outside_external_mask_fails_before_model_creation(self):
        self._write_object("alpha.data", gt_box=(1, 1, 6, 6), mask_box=(2, 2, 5, 5))
        with self.assertRaisesRegex(ValueError, "GT-valid pixel.*outside.*alpha.data"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_symlinked_object(self):
        outside = self.root / "outside"
        outside.mkdir()
        self._write_object("alpha.data", root=outside)
        os.symlink(outside / "alpha.data", self.train_root / "alpha.data", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink|regular|escapes|root"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_symlinked_observation(self):
        object_dir = self._write_object("alpha.data")
        target = self.root / "outside-image.exr"
        target.write_bytes((object_dir / "image_00.exr").read_bytes())
        (object_dir / "image_00.exr").unlink()
        os.symlink(target, object_dir / "image_00.exr")
        with self.assertRaisesRegex(ValueError, "symlink|regular|escapes|root"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_symlinked_ground_truth_and_mask(self):
        object_dir = self._write_object("alpha.data")
        for filename in ("local_normal.exr", "binary_mask.exr"):
            target = self.root / f"outside-{filename}"
            target.write_bytes((object_dir / filename).read_bytes())
            (object_dir / filename).unlink()
            os.symlink(target, object_dir / filename)
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, "symlink|regular|escapes|root"
            ):
                build_private_split_manifest(self.config(), split="train")
            (object_dir / filename).unlink()
            (object_dir / filename).write_bytes(target.read_bytes())

    def test_manifest_rejects_wrong_geometry_in_any_observation(self):
        object_dir = self._write_object("alpha.data")
        write_rgb_exr(object_dir / "image_07.exr", np.ones((255, 256, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "image_07.exr.*geometry"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_nonfinite_observation(self):
        object_dir = self._write_object("alpha.data")
        nonfinite = np.ones((256, 256, 3), dtype=np.float32)
        nonfinite[0, 0, 0] = np.nan
        write_rgb_exr(object_dir / "image_07.exr", nonfinite)
        with self.assertRaisesRegex(ValueError, "non-finite|finite"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_requires_six_observations(self):
        self._write_object("alpha.data", observation_names=[f"image_{i:02d}.exr" for i in range(5)])
        with self.assertRaisesRegex(ValueError, "only 5.*6|requires 6"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_empty_ground_truth_support(self):
        self._write_object("alpha.data", gt_box=None)
        with self.assertRaisesRegex(ValueError, "empty.*GT|GT.*empty"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_rejects_empty_external_mask(self):
        self._write_object("alpha.data", mask_box=None)
        with self.assertRaisesRegex(ValueError, "empty.*mask|mask.*empty"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_uses_selected_split_root(self):
        self._write_object("alpha.data", root=self.test_root)
        manifest = build_private_split_manifest(self.config(), split="test")
        self.assertEqual(manifest.split, "test")
        self.assertEqual(manifest.data_root, str(self.test_root))
        self.assertEqual([record.name for record in manifest.objects], ["alpha.data"])

    def test_manifest_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "split.*train.*test|train or test"):
            build_private_split_manifest(self.config(), split="validation")

    def test_manifest_rejects_missing_split_root_and_empty_split(self):
        missing_config = self.config(train_dir=self.root / "missing")
        with self.assertRaisesRegex(ValueError, "directory|does not exist"):
            build_private_split_manifest(missing_config, split="train")
        with self.assertRaisesRegex(ValueError, "no object"):
            build_private_split_manifest(self.config(), split="train")

    def test_manifest_hash_and_decode_use_one_immutable_snapshot(self):
        object_dir = self._write_object("alpha.data")
        config = self.config()
        original = (object_dir / "image_00.exr").read_bytes()
        replacement = np.full((256, 256, 3), 99.0, dtype=np.float32)
        write_rgb_exr(object_dir / "image_00.exr", replacement)
        changed = (object_dir / "image_00.exr").read_bytes()
        object_dir.joinpath("image_00.exr").write_bytes(original)

        original_reader = private_manifest.read_file_bytes

        def swap_after_read(path: Path, *, label: str) -> bytes:
            if Path(path) == object_dir / "image_00.exr":
                (object_dir / "image_00.exr").write_bytes(changed)
                return original
            return original_reader(path, label=label)

        with mock.patch.object(private_manifest, "read_file_bytes", side_effect=swap_after_read):
            manifest = build_private_split_manifest(config, split="train")
        self.assertEqual(
            manifest.objects[0].observation_sha256[0],
            hashlib.sha256(original).hexdigest(),
        )

    def test_manifest_digest_is_canonical_and_stable(self):
        self._write_object("zeta.data")
        self._write_object("alpha.data")
        first = build_private_split_manifest(self.config(), split="train")
        second = build_private_split_manifest(self.config(), split="train")
        first_bytes = private_manifest_bytes(first)
        self.assertEqual(first_bytes, private_manifest_bytes(second))
        self.assertEqual(private_manifest_sha256(first), hashlib.sha256(first_bytes).hexdigest())
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertEqual(first_bytes[:-1], json.dumps(
            {
                "data_root": str(self.train_root),
                "objects": [
                    {
                        "height": 256,
                        "gt_valid_pixels": 9,
                        "mask_file": "binary_mask.exr",
                        "mask_only_pixels": 16,
                        "mask_sha256": first.objects[0].mask_sha256,
                        "mask_valid_pixels": 25,
                        "name": "alpha.data",
                        "normal_file": "local_normal.exr",
                        "normal_sha256": first.objects[0].normal_sha256,
                        "observation_files": list(first.objects[0].observation_files),
                        "observation_sha256": list(first.objects[0].observation_sha256),
                        "relative_dir": "alpha.data",
                        "width": 256,
                    },
                    {
                        "height": 256,
                        "gt_valid_pixels": 9,
                        "mask_file": "binary_mask.exr",
                        "mask_only_pixels": 16,
                        "mask_sha256": first.objects[1].mask_sha256,
                        "mask_valid_pixels": 25,
                        "name": "zeta.data",
                        "normal_file": "local_normal.exr",
                        "normal_sha256": first.objects[1].normal_sha256,
                        "observation_files": list(first.objects[1].observation_files),
                        "observation_sha256": list(first.objects[1].observation_sha256),
                        "relative_dir": "zeta.data",
                        "width": 256,
                    },
                ],
                "split": "train",
                "version": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))

    def test_records_and_manifest_are_immutable(self):
        self._write_object("alpha.data")
        manifest = build_private_split_manifest(self.config(), split="train")
        self.assertIsInstance(manifest.objects, tuple)
        self.assertIsInstance(manifest.objects[0].observation_files, tuple)
        with self.assertRaises(AttributeError):
            manifest.split = "test"
        with self.assertRaises(AttributeError):
            manifest.objects[0].name = "other.data"

    def test_serializers_reject_wrong_types(self):
        with self.assertRaises(TypeError):
            private_manifest_bytes(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            private_manifest_sha256(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
