"""Focused EXR and canonical ordered-light manifest tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import read_mask_exr, read_rgb_exr
from src.comparison.manifest import (
    build_dataset_manifest,
    load_dataset_manifest,
    save_dataset_manifest,
    save_sdm_selection_manifest,
)
from tests.comparison_helpers import make_object, write_mask_exr, write_rgb_exr


class SdmExrManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.output_root = self.root / "outputs"

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
            max_image_resolution=512,
            pixel_samples=1,
            precision="fp32",
            device="cpu",
            num_workers=0,
            save_exr=True,
            save_png=True,
        )
        values.update(overrides)
        return SdmExrInferenceConfig(**values)

    def test_rgb_exr_round_trip_preserves_channel_order(self):
        expected = np.asarray(
            [[[0.1, 1.2, 2.3], [3.4, 4.5, 5.6]]], dtype=np.float32
        )
        path = self.root / "rgb.exr"
        write_rgb_exr(path, expected)
        np.testing.assert_allclose(read_rgb_exr(path), expected, rtol=0, atol=1e-6)

    def test_mask_exr_is_thresholded_at_greater_than_zero(self):
        path = self.root / "mask.exr"
        write_mask_exr(path, np.asarray([[-1.0, 0.0, 0.5]], dtype=np.float32))
        np.testing.assert_array_equal(read_mask_exr(path), [[0.0, 0.0, 1.0]])

    def test_rgb_mask_uses_red_channel_not_opencv_blue_channel(self):
        path = self.root / "rgb-mask.exr"
        rgb_mask = np.zeros((1, 2, 3), dtype=np.float32)
        rgb_mask[..., 0] = 1.0
        write_rgb_exr(path, rgb_mask)
        np.testing.assert_array_equal(read_mask_exr(path), [[1.0, 1.0]])

    def test_non_finite_rgb_is_rejected_with_object_path(self):
        object_dir = make_object(self.data_root, "broken.data")
        image = np.ones((2, 3, 3), np.float32)
        image[0, 0, 0] = np.nan
        write_rgb_exr(object_dir / "image_000.exr", image)
        with self.assertRaisesRegex(ValueError, "broken\\.data/image_000\\.exr"):
            build_dataset_manifest(self.config(max_image_num=4))

    def test_objects_are_lexically_sorted(self):
        make_object(self.data_root, "zeta.data")
        make_object(self.data_root, "alpha.data")
        manifest = build_dataset_manifest(self.config())
        self.assertEqual(tuple(record.name for record in manifest.objects), ("alpha.data", "zeta.data"))

    def test_seeded_selection_is_stable_per_object(self):
        make_object(self.data_root, "alpha.data", image_count=4)
        make_object(self.data_root, "zeta.data", image_count=4)
        first = build_dataset_manifest(self.config())
        second = build_dataset_manifest(self.config())
        self.assertEqual(first.objects, second.objects)

    def test_manifest_selection_preserves_exact_list_order(self):
        make_object(self.data_root, "alpha.data", image_count=4)
        selection_path = self.root / "selection.json"
        selection_path.write_text(
            json.dumps({"alpha.data": ["image_003.exr", "image_001.exr"]}),
            encoding="utf-8",
        )
        manifest = build_dataset_manifest(
            self.config(light_selection="manifest", selection_manifest=selection_path)
        )
        self.assertEqual(manifest.objects[0].selected_images, ("image_003.exr", "image_001.exr"))

    def test_manifest_rejects_literal_backslash_path_components(self):
        make_object(self.data_root, "alpha.data", image_count=4)
        selection_path = self.root / "selection.json"
        selection_path.write_text(
            json.dumps({"alpha.data": [r"nested\\image_000.exr"]}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "out-of-object path"):
            build_dataset_manifest(
                self.config(light_selection="manifest", selection_manifest=selection_path)
            )

    def test_manifest_rejects_windows_drive_relative_selection(self):
        make_object(self.data_root, "alpha.data", image_count=4)
        selection_path = self.root / "selection.json"
        selection_path.write_text(
            json.dumps({"alpha.data": ["C:mask.exr"]}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "out-of-object path"):
            build_dataset_manifest(
                self.config(light_selection="manifest", selection_manifest=selection_path)
            )

    def test_external_mask_filename_must_be_object_relative_basename(self):
        object_dir = make_object(self.data_root, "alpha.data", image_count=4)
        # Ensure the currently permissive implementation would find each bad
        # path and proceed, so the test specifically catches missing validation.
        outside_relative = self.data_root / "outside_mask.exr"
        write_mask_exr(outside_relative, np.ones((2, 3), dtype=np.float32))
        outside_absolute = self.root / "absolute_mask.exr"
        write_mask_exr(outside_absolute, np.ones((2, 3), dtype=np.float32))
        windows_literal = object_dir / r"..\outside_mask.exr"
        write_mask_exr(windows_literal, np.ones((2, 3), dtype=np.float32))
        for bad_filename in (
            "../outside_mask.exr",
            str(outside_absolute),
            r"..\outside_mask.exr",
        ):
            with self.subTest(bad_filename=bad_filename), self.assertRaisesRegex(
                ValueError, "external_mask_filename"
            ):
                build_dataset_manifest(self.config(external_mask_filename=bad_filename))

    def test_dataset_manifest_round_trip_restores_immutable_records(self):
        make_object(self.data_root, "alpha.data")
        expected = build_dataset_manifest(self.config())
        path = self.root / "input_manifest.json"
        save_dataset_manifest(expected, path)
        actual = load_dataset_manifest(path)
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual.objects, tuple)
        self.assertIsInstance(actual.objects[0].selected_images, tuple)

    def test_dataset_manifest_loader_rejects_malformed_records(self):
        make_object(self.data_root, "alpha.data")
        manifest = build_dataset_manifest(self.config())
        path = self.root / "input_manifest.json"
        save_dataset_manifest(manifest, path)
        baseline = json.loads(path.read_text(encoding="utf-8"))

        cases = (
            ("version_bool", lambda value: value.update(version=True)),
            ("data_root_empty", lambda value: value.update(data_root="")),
            ("height_zero", lambda value: value["objects"][0].update(height=0)),
            ("width_bool", lambda value: value["objects"][0].update(width=False)),
            ("name_empty", lambda value: value["objects"][0].update(name="")),
            ("relative_dir_empty", lambda value: value["objects"][0].update(relative_dir="")),
            ("normal_file_empty", lambda value: value["objects"][0].update(normal_file="")),
            ("selected_images_empty", lambda value: value["objects"][0].update(selected_images=[])),
            (
                "image_hash_cardinality",
                lambda value: value["objects"][0].update(image_sha256=[]),
            ),
            (
                "mask_hash_without_name",
                lambda value: value["objects"][0].update(mask_file=None),
            ),
            (
                "mask_name_without_hash",
                lambda value: value["objects"][0].update(mask_sha256=None),
            ),
            (
                "record_not_object",
                lambda value: value.update(objects=["not-a-record"]),
            ),
        )
        for case_name, mutate in cases:
            with self.subTest(case_name=case_name):
                candidate = json.loads(json.dumps(baseline))
                mutate(candidate)
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dataset_manifest(path)

    def test_dataset_manifest_loader_rejects_unsafe_or_mismatched_relative_dir(self):
        make_object(self.data_root, "alpha.data")
        manifest = build_dataset_manifest(self.config())
        path = self.root / "input_manifest.json"
        save_dataset_manifest(manifest, path)
        baseline = json.loads(path.read_text(encoding="utf-8"))

        bad_relative_dirs = (
            "../outside.data",
            "/tmp/outside.data",
            r"C:\\tmp\\outside.data",
            r"nested\\alpha.data",
            "C:alpha.data",
            "other.data",
        )
        for bad_relative_dir in bad_relative_dirs:
            with self.subTest(relative_dir=bad_relative_dir):
                candidate = json.loads(json.dumps(baseline))
                candidate["objects"][0]["relative_dir"] = bad_relative_dir
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "relative_dir"):
                    load_dataset_manifest(path)

    def test_manifest_uses_object_relative_basenames_for_sdm(self):
        make_object(self.data_root, "alpha.data")
        manifest = build_dataset_manifest(self.config())
        output = self.root / "selected_lights.json"
        save_sdm_selection_manifest(manifest, output)
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            {"alpha.data": list(manifest.objects[0].selected_images)},
        )
        self.assertTrue(all("/" not in name for name in manifest.objects[0].selected_images))

    def test_too_few_images_fails_instead_of_using_fewer(self):
        make_object(self.data_root, "alpha.data", image_count=1)
        with self.assertRaisesRegex(ValueError, "alpha\\.data"):
            build_dataset_manifest(self.config(max_image_num=2))

    def test_external_policy_requires_aligned_nonempty_mask(self):
        object_dir = make_object(self.data_root, "alpha.data")
        write_mask_exr(object_dir / "binary_mask.exr", np.zeros((1, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "alpha\\.data/binary_mask\\.exr"):
            build_dataset_manifest(self.config())

    def test_source_hashes_change_when_an_observation_changes(self):
        object_dir = make_object(self.data_root, "alpha.data")
        before = build_dataset_manifest(self.config())
        changed = np.zeros((2, 3, 3), np.float32)
        changed[..., 0] = 99.0
        changed[..., 1] = 88.0
        changed[..., 2] = 77.0
        write_rgb_exr(object_dir / before.objects[0].selected_images[0], changed)
        after = build_dataset_manifest(self.config())
        self.assertNotEqual(before.objects[0].image_sha256, after.objects[0].image_sha256)


if __name__ == "__main__":
    unittest.main()
