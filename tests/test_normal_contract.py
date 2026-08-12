"""Unit tests for explicit source-normal encoding semantics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import sha256_file
from src.comparison.manifest import ObjectRecord
from src.comparison.metrics import load_source_gt, normal_validity_mask
from src.comparison.normal_contract import decode_ground_truth_normal
from tests.comparison_helpers import write_rgb_exr


class NormalContractTests(unittest.TestCase):
    def test_unsigned_known_vectors_decode_to_signed_coordinates(self):
        encoded = np.asarray(
            [[[0.5, 0.5, 0.5], [1.0, 0.5, 0.0]]],
            dtype=np.float32,
        )
        decoded = decode_ground_truth_normal(encoded, "unsigned")
        np.testing.assert_allclose(
            decoded,
            [[[0.0, 0.0, 0.0], [1.0, 0.0, -1.0]]],
            rtol=0,
            atol=1.0e-7,
        )
        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(decoded.flags.c_contiguous)

    def test_signed_values_are_preserved_without_remapping(self):
        signed = np.asarray([[[-0.25, 0.5, 1.0]]], dtype=np.float32)
        decoded = decode_ground_truth_normal(signed, "signed")
        np.testing.assert_array_equal(decoded, signed)
        self.assertIsNot(decoded, signed)

    def test_unsigned_endpoint_tolerance_is_clipped_before_decoding(self):
        encoded = np.asarray(
            [[[-5.0e-7, 0.5, 1.0 + 5.0e-7]]],
            dtype=np.float32,
        )
        decoded = decode_ground_truth_normal(encoded, "unsigned")
        np.testing.assert_allclose(decoded, [[[-1.0, 0.0, 1.0]]], atol=1.0e-7)

    def test_unsigned_out_of_range_or_nonfinite_values_fail_closed(self):
        for value in (-2.0e-6, 1.0 + 2.0e-6, np.nan, np.inf):
            encoded = np.full((1, 1, 3), 0.5, dtype=np.float32)
            encoded[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_ground_truth_normal(encoded, "unsigned")

    def test_unknown_encoding_and_malformed_arrays_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "normal encoding"):
            decode_ground_truth_normal(np.zeros((1, 1, 3)), "octahedral")
        with self.assertRaisesRegex(ValueError, "H,W,3"):
            decode_ground_truth_normal(np.zeros((2, 3)), "signed")

    def test_manifest_bound_source_loader_decodes_unsigned_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            object_dir = data_root / "alpha.data"
            normal_path = object_dir / "local_normal.exr"
            encoded = np.asarray(
                [[[0.5, 0.5, 0.5], [0.5, 0.5, 1.0]]],
                dtype=np.float32,
            )
            write_rgb_exr(normal_path, encoded)
            config = SdmExrInferenceConfig(
                checkpoint=root / "lino.pth",
                data_root=data_root,
                output_root=root / "output",
                object_suffix=".data",
                image_prefix="image",
                image_extension=".exr",
                max_image_num=1,
                light_selection="seeded",
                selection_manifest=None,
                seed=1,
                mask_policy="full",
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
                expected_source_geometry=(1, 2),
            )
            record = ObjectRecord(
                name="alpha.data",
                relative_dir="alpha.data",
                height=1,
                width=2,
                selected_images=("image_000.exr",),
                image_sha256=("unused-by-source-loader",),
                normal_file="local_normal.exr",
                normal_sha256=sha256_file(normal_path),
                mask_file=None,
                mask_sha256=None,
            )

            decoded, loaded_path = load_source_gt(config, record)

            np.testing.assert_allclose(decoded[0, 1], [0.0, 0.0, 1.0], atol=1.0e-7)
            np.testing.assert_array_equal(normal_validity_mask(decoded), [[False, True]])
            self.assertEqual(loaded_path, normal_path.resolve())


if __name__ == "__main__":
    unittest.main()
