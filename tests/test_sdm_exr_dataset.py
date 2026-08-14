"""Focused tests for the LINO SDM-EXR dataset adapter."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import cv2

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import read_mask_exr, read_rgb_exr, sha256_file
from src.comparison.manifest import (
    DatasetManifest,
    ObjectRecord,
    build_dataset_manifest,
    stable_seed as legacy_manifest_seed,
)
from src.data.data_module import get_roi
from src.data.sdm_exr_data import SdmExrDataset, collate_single_sdm_exr
from tests.comparison_helpers import make_object, write_mask_exr, write_rgb_exr


class SdmExrDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.output_root = self.root / "output"

    def config(self, **overrides) -> SdmExrInferenceConfig:
        values = dict(
            checkpoint=self.root / "weights" / "lino.pth",
            data_root=self.data_root,
            output_root=self.output_root,
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            max_image_num=2,
            light_selection="seeded",
            selection_manifest=None,
            seed=20260710,
            mask_policy="external",
            external_mask_filename="binary_mask.exr",
            normal_filenames=("local_normal.exr",),
            normal_encoding="signed",
            mask_margin=0,
            max_image_resolution=1024,
            pixel_samples=1,
            precision="fp32",
            device="cpu",
            num_workers=0,
            save_exr=True,
            save_png=True,
        )
        values.update(overrides)
        return SdmExrInferenceConfig(**values)

    def manifest(self, **config_overrides) -> tuple[SdmExrInferenceConfig, DatasetManifest]:
        config = self.config(**config_overrides)
        return config, build_dataset_manifest(config)

    def record(self, **overrides) -> ObjectRecord:
        values = dict(
            name="alpha.data",
            relative_dir="alpha.data",
            height=2,
            width=3,
            selected_images=("image_000.exr", "image_001.exr"),
            image_sha256=("a", "b"),
            normal_file="local_normal.exr",
            normal_sha256="n",
            mask_file="binary_mask.exr",
            mask_sha256="m",
        )
        values.update(overrides)
        return ObjectRecord(**values)

    def standalone_manifest(
        self, record: ObjectRecord, *, data_root: Path | None = None
    ) -> DatasetManifest:
        return DatasetManifest(
            version=1,
            data_root=str(self.data_root if data_root is None else data_root),
            seed=1,
            max_image_num=2,
            objects=(record,),
        )

    def test_external_policy_uses_binary_mask_for_input_and_original_mask(self):
        object_dir = make_object(self.data_root, "alpha.data")
        source_mask = np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.float32)
        write_mask_exr(object_dir / "binary_mask.exr", source_mask)
        config, manifest = self.manifest()

        sample = SdmExrDataset(config, manifest)[0]

        self.assertEqual(set(sample), {"imgs", "mask", "mask_original", "roi", "metadata"})
        np.testing.assert_array_equal(sample["mask_original"].numpy(), source_mask[None])
        self.assertEqual(sample["mask"].dtype, torch.float32)
        self.assertEqual(sample["mask_original"].dtype, torch.float32)
        self.assertTrue(set(np.unique(sample["mask"].numpy())).issubset({0.0, 1.0}))
        self.assertTrue(np.any(sample["mask"].numpy() > 0))
        self.assertEqual(sample["metadata"]["mask_policy"], "external")

    def test_tensor_contract_and_metadata_are_strictly_serializable(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        sample = SdmExrDataset(config, manifest)[0]

        self.assertEqual(sample["imgs"].dtype, torch.float32)
        self.assertEqual(sample["imgs"].shape[0], 3)
        self.assertEqual(sample["imgs"].shape[-1], len(manifest.objects[0].selected_images))
        self.assertEqual(sample["mask"].dtype, torch.float32)
        self.assertEqual(sample["mask_original"].dtype, torch.float32)
        self.assertEqual(sample["roi"].dtype, torch.int64)
        self.assertEqual(sample["roi"].shape, (6,))
        json.dumps(sample["metadata"], allow_nan=False)

    def test_dataset_rejects_manifest_root_mismatch_and_accepts_resolved_equivalence(self):
        record = self.record()
        other_root = self.root / "other-data"
        mismatch = self.standalone_manifest(record, data_root=other_root)
        with self.assertRaisesRegex(ValueError, "data_root"):
            SdmExrDataset(self.config(), mismatch)

        equivalent_config = self.config(data_root=self.data_root / "nested" / "..")
        equivalent = self.standalone_manifest(record, data_root=self.data_root)
        # Path.resolve() equivalence is accepted even when the textual paths
        # differ; this is the form produced by callers composing run roots.
        self.assertEqual(
            SdmExrDataset(equivalent_config, equivalent).manifest,
            equivalent,
        )

    def test_dataset_rejects_unsafe_record_and_selection_metadata_at_init(self):
        config = self.config()
        cases = (
            ("relative_dir_mismatch", self.record(relative_dir="other.data")),
            ("relative_dir_backslash", self.record(relative_dir=r"alpha\\data")),
            ("relative_dir_drive", self.record(relative_dir="C:alpha.data")),
            ("selected_traversal", self.record(selected_images=("../image.exr", "image_001.exr"))),
            ("selected_backslash", self.record(selected_images=(r"nested\\image.exr", "image_001.exr"))),
            ("selected_drive", self.record(selected_images=("C:image.exr", "image_001.exr"))),
            ("selected_nonstring", self.record(selected_images=(["nested"], "image_001.exr"))),
            ("selected_duplicate", self.record(selected_images=("image_000.exr", "image_000.exr"))),
            ("digest_cardinality", self.record(image_sha256=("a",))),
        )
        for case_name, record in cases:
            with self.subTest(case_name=case_name), self.assertRaisesRegex(ValueError, "manifest"):
                SdmExrDataset(config, self.standalone_manifest(record))

    def test_object_and_selected_symlink_targets_must_stay_inside_root(self):
        outside_root = self.root / "outside"
        outside_root.mkdir()

        outside_object = make_object(outside_root, "alpha.data")
        linked_object = self.data_root / "alpha.data"
        self.data_root.mkdir(parents=True, exist_ok=True)
        os.symlink(outside_object, linked_object, target_is_directory=True)
        config, manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "root"):
            SdmExrDataset(config, manifest)[0]

    def test_selected_image_symlink_target_must_stay_inside_object(self):
        object_dir = make_object(self.data_root, "alpha.data")
        outside_image = self.root / "outside-image.exr"
        outside_image.write_bytes((object_dir / "image_000.exr").read_bytes())
        image_path = object_dir / "image_000.exr"
        image_path.unlink()
        os.symlink(outside_image, image_path)
        config, manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            SdmExrDataset(config, manifest)[0]

    def test_external_mask_symlink_target_must_stay_inside_object(self):
        object_dir = make_object(self.data_root, "alpha.data")
        outside_mask = self.root / "outside-mask.exr"
        outside_mask.write_bytes((object_dir / "binary_mask.exr").read_bytes())
        mask_path = object_dir / "binary_mask.exr"
        mask_path.unlink()
        os.symlink(outside_mask, mask_path)
        config, manifest = self.manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            SdmExrDataset(config, manifest)[0]

    def test_mutated_selected_observation_digest_is_rejected(self):
        object_dir = make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        selected_name = manifest.objects[0].selected_images[0]
        changed = np.full((2, 3, 3), 17.0, dtype=np.float32)
        write_rgb_exr(object_dir / selected_name, changed)
        with self.assertRaisesRegex(ValueError, "digest"):
            SdmExrDataset(config, manifest)[0]

    def test_selected_observation_hash_and_decode_share_one_immutable_snapshot(self):
        from src.data import sdm_exr_data

        object_dir = make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        image_path = object_dir / manifest.objects[0].selected_images[0]
        original_bytes = image_path.read_bytes()
        changed = np.full((2, 3, 3), 17.0, dtype=np.float32)
        write_rgb_exr(image_path, changed)
        changed_bytes = image_path.read_bytes()
        image_path.write_bytes(original_bytes)

        def swap_after_snapshot(path, *, label):
            source = Path(path)
            if source == image_path:
                image_path.write_bytes(changed_bytes)
                return original_bytes
            return source.read_bytes()

        with mock.patch.object(sdm_exr_data, "read_file_bytes", side_effect=swap_after_snapshot):
            sample = SdmExrDataset(config, manifest)[0]
        self.assertEqual(sample["metadata"]["selected_image_sha256"][0], manifest.objects[0].image_sha256[0])

    def test_mutated_external_mask_digest_is_rejected(self):
        object_dir = make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        write_mask_exr(object_dir / "binary_mask.exr", np.full((2, 3), 2.0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "digest"):
            SdmExrDataset(config, manifest)[0]

    def test_metadata_reports_verified_manifest_digests(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        sample = SdmExrDataset(config, manifest)[0]
        record = manifest.objects[0]

        self.assertEqual(sample["metadata"]["selected_image_sha256"], list(record.image_sha256))
        self.assertEqual(sample["metadata"]["mask_digest"], record.mask_sha256)

    def test_dataset_access_never_reads_ground_truth_normals(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        with mock.patch(
            "src.comparison.exr_io.read_signed_normal_exr",
            side_effect=AssertionError("ground-truth normal read"),
        ):
            sample = SdmExrDataset(config, manifest)[0]
        self.assertEqual(sample["imgs"].dtype, torch.float32)

    def test_full_policy_uses_ones_for_input_and_original_mask(self):
        object_dir = make_object(self.data_root, "alpha.data")
        write_mask_exr(object_dir / "binary_mask.exr", np.zeros((2, 3), dtype=np.float32))
        config, manifest = self.manifest(mask_policy="full")

        sample = SdmExrDataset(config, manifest)[0]

        np.testing.assert_array_equal(sample["mask_original"].numpy(), np.ones((1, 2, 3), np.float32))
        np.testing.assert_array_equal(sample["mask"].numpy(), np.ones_like(sample["mask"].numpy()))
        self.assertEqual(sample["metadata"]["mask_policy"], "full")
        self.assertEqual(sample["metadata"]["mask_source"], "full")

    def test_batch_has_no_ground_truth_or_evaluation_mask(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        sample = SdmExrDataset(config, manifest)[0]

        self.assertNotIn("nml", sample)
        self.assertNotIn("normal", sample)
        self.assertNotIn("ground_truth", sample)
        self.assertNotIn("evaluation_mask", sample)

    def test_manifest_order_is_the_tensor_light_order(self):
        make_object(self.data_root, "alpha.data", image_count=3)
        selection_path = self.root / "selected.json"
        selection_path.write_text(
            json.dumps({"alpha.data": ["image_002.exr", "image_000.exr"]}), encoding="utf-8"
        )
        config, manifest = self.manifest(light_selection="manifest", selection_manifest=selection_path)
        dataset = SdmExrDataset(config, manifest)

        pre = dataset._load_pre_normalization(manifest.objects[0])

        self.assertEqual(pre["selected_images"], ("image_002.exr", "image_000.exr"))
        # The fixture encodes the light index in the red channel.  This helper
        # intentionally exposes values before normalization for an analytical
        # order check, while __getitem__ returns the model tensor contract.
        self.assertAlmostEqual(float(pre["images"][0, 0, 0, 0]), 2.125, places=5)
        self.assertAlmostEqual(float(pre["images"][0, 0, 0, 1]), 0.125, places=5)

    def test_output_height_and_width_are_multiples_of_512(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest(max_image_resolution=2048)
        sample = SdmExrDataset(config, manifest)[0]

        height, width = sample["imgs"].shape[1:3]
        self.assertEqual(height % 512, 0)
        self.assertEqual(width % 512, 0)
        self.assertEqual(sample["metadata"]["resized_geometry"], {"height": height, "width": width})

    def test_normalization_is_repeatable_for_object_and_seed(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        first = SdmExrDataset(config, manifest)[0]
        second = SdmExrDataset(config, manifest)[0]

        torch.testing.assert_close(first["imgs"], second["imgs"])
        self.assertEqual(first["metadata"], second["metadata"])

    def test_legacy_dataset_without_version_matches_released_transfer_formula(self):
        object_dir = make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest(max_image_resolution=1024, mask_margin=0)
        dataset = SdmExrDataset(config, manifest)
        sample = dataset[0]
        record = manifest.objects[0]

        source_images = [read_rgb_exr(object_dir / name) for name in record.selected_images]
        source_mask = read_mask_exr(object_dir / "binary_mask.exr")
        roi = np.asarray(get_roi(source_mask, margin=0), dtype=np.int64)
        _, _, row_start, row_end, col_start, col_end = map(int, roi)
        cropped_height = row_end - row_start
        cropped_width = col_end - col_start
        target = max(512, min(1024, (max(cropped_height, cropped_width) // 512) * 512))
        expected_images = np.stack(
            [
                cv2.resize(
                    image[row_start:row_end, col_start:col_end, :],
                    (target, target),
                    interpolation=cv2.INTER_CUBIC,
                )
                for image in source_images
            ],
            axis=-1,
        ).astype(np.float32)
        expected_mask = np.asarray(
            cv2.resize(
                source_mask[row_start:row_end, col_start:col_end],
                (target, target),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0.5,
            dtype=np.float32,
        )
        foreground = expected_images[expected_mask > 0]
        intensity = np.mean(foreground, axis=1, dtype=np.float64)
        spatial_mean = np.mean(intensity, axis=0, dtype=np.float64)
        spatial_max = np.max(intensity, axis=0)
        alpha = np.random.default_rng(
            legacy_manifest_seed(config.seed, record.name, "lino_normalization")
        ).random(expected_images.shape[-1])
        scales = (1.0 - alpha) * spatial_mean + alpha * spatial_max
        expected_images = expected_images / (
            scales.reshape(1, 1, 1, expected_images.shape[-1]) + 1.0e-6
        )

        np.testing.assert_allclose(sample["imgs"].numpy(), expected_images.transpose(2, 0, 1, 3))
        np.testing.assert_array_equal(sample["mask"].numpy()[0], expected_mask)
        np.testing.assert_array_equal(sample["mask_original"].numpy()[0], source_mask)
        np.testing.assert_array_equal(sample["roi"].numpy(), roi)
        np.testing.assert_allclose(sample["metadata"]["normalization_alpha"], alpha)
        np.testing.assert_allclose(sample["metadata"]["normalization_scales"], scales)

    def test_external_and_full_use_their_own_native_normalization_support(self):
        object_dir = make_object(self.data_root, "alpha.data")
        # Deliberately make foreground and background intensities different;
        # external and full support must therefore produce different scales.
        for index in range(4):
            image = np.full((2, 3, 3), 100.0, dtype=np.float32)
            image[:, 0, :] = 1.0
            write_rgb_exr(object_dir / f"image_{index:03d}.exr", image)
        write_mask_exr(object_dir / "binary_mask.exr", np.asarray([[1, 0, 0], [1, 0, 0]], np.float32))

        external_config, external_manifest = self.manifest(mask_policy="external")
        full_config, full_manifest = self.manifest(mask_policy="full")
        external = SdmExrDataset(external_config, external_manifest)[0]
        full = SdmExrDataset(full_config, full_manifest)[0]

        self.assertNotEqual(
            external["metadata"]["normalization"]["scales"],
            full["metadata"]["normalization"]["scales"],
        )

    def test_collate_rejects_batch_size_other_than_one(self):
        make_object(self.data_root, "alpha.data")
        config, manifest = self.manifest()
        dataset = SdmExrDataset(config, manifest)
        sample = dataset[0]

        with self.assertRaisesRegex(ValueError, "exactly one"):
            collate_single_sdm_exr([])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            collate_single_sdm_exr([sample, sample])

        batch = collate_single_sdm_exr([sample])
        self.assertEqual(batch["imgs"].shape[0], 1)
        self.assertEqual(batch["mask"].shape[0], 1)
        self.assertEqual(batch["mask_original"].shape[0], 1)
        self.assertEqual(batch["roi"].shape, (1, 6))
        self.assertIs(batch["metadata"], sample["metadata"])

    def test_source_geometry_mismatch_is_rejected(self):
        object_dir = make_object(self.data_root, "alpha.data")
        record = ObjectRecord(
            name="alpha.data",
            relative_dir="alpha.data",
            height=99,
            width=99,
            selected_images=("image_000.exr", "image_001.exr"),
            image_sha256=(
                sha256_file(object_dir / "image_000.exr"),
                sha256_file(object_dir / "image_001.exr"),
            ),
            normal_file="local_normal.exr",
            normal_sha256="n",
            mask_file="binary_mask.exr",
            mask_sha256=sha256_file(object_dir / "binary_mask.exr"),
        )
        manifest = DatasetManifest(
            version=1,
            data_root=str(self.data_root),
            seed=1,
            max_image_num=2,
            objects=(record,),
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            SdmExrDataset(self.config(), manifest)[0]


if __name__ == "__main__":
    unittest.main()
