"""CPU-only preflight and diagnostic tests for the private LINO transfer gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.manifest import build_dataset_manifest
from src.comparison.transfer_gate import preflight_transfer_sources
from tests.comparison_helpers import (
    make_unsigned_object,
    write_mask_exr,
    write_rgb_exr,
)


class PrivateTransferGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"

    def config(self, **overrides) -> SdmExrInferenceConfig:
        values = dict(
            checkpoint=self.root / "lino.pth",
            data_root=self.data_root,
            output_root=self.root / "output",
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
            normal_encoding="unsigned",
            mask_margin=0,
            max_image_resolution=512,
            pixel_samples=1,
            precision="fp32",
            device="cpu",
            num_workers=0,
            save_exr=True,
            save_png=False,
            expected_source_geometry=(2, 3),
        )
        values.update(overrides)
        return SdmExrInferenceConfig(**values)

    def test_preflight_accepts_mask_halo_and_records_unclipped_hdr_stats(self):
        make_unsigned_object(self.data_root, "alpha.data")
        config = self.config()
        manifest = build_dataset_manifest(config)
        report = preflight_transfer_sources(config, manifest)["alpha.data"]

        self.assertEqual(report["source_geometry"], {"height": 2, "width": 3})
        self.assertEqual(report["decoded_gt_valid_pixel_count"], 4)
        self.assertEqual(report["external_mask_pixel_count"], 5)
        self.assertEqual(report["intersection_pixel_count"], 4)
        self.assertEqual(report["gt_outside_mask_pixel_count"], 0)
        self.assertEqual(report["mask_only_pixel_count"], 1)
        self.assertEqual(
            [item["filename"] for item in report["selected_observations"]],
            list(manifest.objects[0].selected_images),
        )
        self.assertTrue(any(item["maximum"] > 1000.0 for item in report["selected_observations"]))

    def test_preflight_rejects_gt_support_outside_external_mask(self):
        object_dir = make_unsigned_object(self.data_root, "alpha.data")
        mask = np.zeros((2, 3), dtype=np.float32)
        mask[0, 0] = 1.0
        write_mask_exr(object_dir / "binary_mask.exr", mask)
        config = self.config()
        manifest = build_dataset_manifest(config)
        with self.assertRaisesRegex(ValueError, "GT-valid.*outside.*mask"):
            preflight_transfer_sources(config, manifest)

    def test_preflight_enforces_declared_source_geometry(self):
        make_unsigned_object(self.data_root, "alpha.data")
        config = self.config(expected_source_geometry=(256, 256))
        manifest = build_dataset_manifest(config)
        with self.assertRaisesRegex(ValueError, "expected source geometry"):
            preflight_transfer_sources(config, manifest)

    def test_preflight_rejects_source_changed_after_manifest_snapshot(self):
        object_dir = make_unsigned_object(self.data_root, "alpha.data")
        config = self.config()
        manifest = build_dataset_manifest(config)
        selected = object_dir / manifest.objects[0].selected_images[0]
        write_rgb_exr(selected, np.full((2, 3, 3), 77.0, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "selected observation.*digest"):
            preflight_transfer_sources(config, manifest)

    def test_preflight_rejects_empty_decoded_gt_support(self):
        object_dir = make_unsigned_object(self.data_root, "alpha.data")
        write_rgb_exr(
            object_dir / "local_normal.exr",
            np.full((2, 3, 3), 0.5, dtype=np.float32),
        )
        config = self.config()
        manifest = build_dataset_manifest(config)
        with self.assertRaisesRegex(ValueError, "empty decoded GT support"):
            preflight_transfer_sources(config, manifest)

    def test_preflight_rejects_unsigned_gt_outside_endpoint_tolerance(self):
        object_dir = make_unsigned_object(self.data_root, "alpha.data")
        encoded = np.full((2, 3, 3), 0.5, dtype=np.float32)
        encoded[:, :2, 2] = 1.01
        write_rgb_exr(object_dir / "local_normal.exr", encoded)
        config = self.config()
        manifest = build_dataset_manifest(config)
        with self.assertRaisesRegex(ValueError, "unsigned values"):
            preflight_transfer_sources(config, manifest)
