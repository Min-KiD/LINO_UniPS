"""Contract tests for versioned LINO-native geometry and normalization."""

from __future__ import annotations

import hashlib
import json
import unittest

import cv2
import numpy as np

from src.data.lino_native_preprocessing import (
    normalize_lino_observations,
    prepare_lino_native_geometry,
    restore_lino_prediction,
)


class LinoNativePreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.images = np.ones((6, 8, 3, 2), dtype=np.float32)
        self.images[1, 1, :, 0] = 2.0
        self.images[2, 2, :, 1] = 3.0
        self.mask = np.zeros((6, 8), dtype=np.float32)
        self.mask[1:5, 1:7] = 1

    def test_external_halo_is_model_support_not_target_support(self):
        target_mask = np.zeros((6, 8), dtype=np.float32)
        target_mask[2:4, 2:6] = 1
        normal = np.zeros((6, 8, 3), dtype=np.float32)
        normal[..., 2] = target_mask

        result = prepare_lino_native_geometry(
            self.images,
            self.mask,
            target_normal=normal,
            target_mask=target_mask,
            margin=0,
            target_resolution=512,
            object_name="alpha.data",
        )

        self.assertGreater(int(result.model_mask.sum()), int(result.target_mask.sum()))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(result.target_normal[result.target_mask > 0], axis=1),
                1.0,
            )
        )

    def test_target_support_outside_model_mask_is_rejected_before_resize(self):
        target_mask = np.zeros_like(self.mask)
        target_mask[0, 0] = 1
        normal = np.zeros((*target_mask.shape, 3), dtype=np.float32)
        normal[..., 2] = target_mask

        with self.assertRaisesRegex(
            ValueError,
            r"alpha\.data.*target_mask.*1.*outside model support",
        ):
            prepare_lino_native_geometry(
                self.images,
                self.mask,
                target_normal=normal,
                target_mask=target_mask,
                margin=0,
                target_resolution=512,
                object_name="alpha.data",
            )

    def test_paired_transforms_use_common_half_open_roi_and_field_interpolation(self):
        height, width = 7, 11
        images = np.zeros((height, width, 3, 1), dtype=np.float32)
        rows, cols = np.indices((height, width))
        images[..., 0, 0] = cols**2 + 2 * rows
        images[..., 1, 0] = rows**2 + 3 * cols
        images[..., 2, 0] = rows * cols + 5
        model_mask = np.zeros((height, width), dtype=np.float32)
        model_mask[2:6, 3:9] = 1
        target_mask = np.zeros((height, width), dtype=np.float32)
        target_mask[3:5, 4:8] = 1
        normal = np.zeros((height, width, 3), dtype=np.float32)
        normal[..., 0] = target_mask * (rows + 1)
        normal[..., 1] = target_mask * (cols + 2)
        normal[..., 2] = target_mask * (rows + cols + 3)

        result = prepare_lino_native_geometry(
            images,
            model_mask,
            target_normal=normal,
            target_mask=target_mask,
            margin=0,
            target_resolution=512,
            object_name="asymmetric.data",
        )

        expected_roi = np.asarray([height, width, 1, 6, 3, 8], dtype=np.int64)
        self.assertEqual(result.roi.tolist(), expected_roi.tolist())
        expected_images = cv2.resize(
            images[1:6, 3:8, :, 0],
            (512, 512),
            interpolation=cv2.INTER_CUBIC,
        )[..., None]
        expected_model_mask = np.asarray(
            cv2.resize(model_mask[1:6, 3:8], (512, 512), interpolation=cv2.INTER_NEAREST)
            > 0.5,
            dtype=np.float32,
        )
        expected_target_mask = np.asarray(
            cv2.resize(target_mask[1:6, 3:8], (512, 512), interpolation=cv2.INTER_NEAREST)
            > 0.5,
            dtype=np.float32,
        )
        expected_normal = cv2.resize(
            normal[1:6, 3:8, :],
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        lengths = np.linalg.norm(expected_normal.astype(np.float64), axis=2, keepdims=True)
        expected_normal = np.divide(
            expected_normal,
            lengths,
            out=np.zeros_like(expected_normal),
            where=(expected_target_mask[..., None] > 0) & (lengths > 0),
        )

        np.testing.assert_allclose(result.images, expected_images)
        np.testing.assert_array_equal(result.model_mask, expected_model_mask)
        np.testing.assert_array_equal(result.target_mask, expected_target_mask)
        np.testing.assert_allclose(result.target_normal, expected_normal)
        np.testing.assert_array_equal(result.source_model_mask, model_mask)
        np.testing.assert_array_equal(result.source_target_mask, target_mask)

    def test_restore_prediction_places_signed_unit_vectors_and_zero_background(self):
        prediction = np.zeros((2, 3, 3), dtype=np.float32)
        prediction[..., 0] = 3
        prediction[..., 1] = -4
        roi = np.asarray([8, 10, 2, 6, 3, 8], dtype=np.int64)

        restored = restore_lino_prediction(
            prediction,
            roi,
            source_height=8,
            source_width=10,
        )

        expected = np.zeros((8, 10, 3), dtype=np.float32)
        expected[2:6, 3:8, 0] = 0.6
        expected[2:6, 3:8, 1] = -0.8
        np.testing.assert_allclose(restored, expected)
        np.testing.assert_allclose(
            np.linalg.norm(restored[2:6, 3:8], axis=2),
            1.0,
        )
        outside = ~self._region_mask()
        self.assertTrue(np.all(restored[outside] == 0))

    @staticmethod
    def _region_mask() -> np.ndarray:
        support = np.zeros((8, 10), dtype=bool)
        support[2:6, 3:8] = True
        return support

    def test_private_training_normalization_changes_only_for_train_epoch(self):
        train0 = normalize_lino_observations(
            self.images,
            self.mask,
            base_seed=20260710,
            split="train",
            epoch=0,
            object_name="alpha.data",
            version="private_external_lino_native_v1",
        )
        train1 = normalize_lino_observations(
            self.images,
            self.mask,
            base_seed=20260710,
            split="train",
            epoch=1,
            object_name="alpha.data",
            version="private_external_lino_native_v1",
        )
        test0 = normalize_lino_observations(
            self.images,
            self.mask,
            base_seed=20260710,
            split="test",
            epoch=0,
            object_name="alpha.data",
            version="private_external_lino_native_v1",
        )
        test8 = normalize_lino_observations(
            self.images,
            self.mask,
            base_seed=20260710,
            split="test",
            epoch=8,
            object_name="alpha.data",
            version="private_external_lino_native_v1",
        )

        self.assertFalse(np.array_equal(train0.images, train1.images))
        self.assertTrue(np.array_equal(test0.images, test8.images))

    def test_private_normalization_seeds_and_alpha_match_independent_canonical_hash(self):
        def expected_seed(split: str, epoch: int) -> int:
            effective_epoch = epoch if split.lower() == "train" else 0
            payload = json.dumps(
                [
                    20260710,
                    split.lower(),
                    effective_epoch,
                    "alpha.data",
                    "lino_normalization",
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

        for split, epoch in (("train", 7), ("validation", 4), ("TEST", 99)):
            with self.subTest(split=split, epoch=epoch):
                result = normalize_lino_observations(
                    self.images,
                    self.mask,
                    base_seed=20260710,
                    split=split,
                    epoch=epoch,
                    object_name="alpha.data",
                    version="private_external_lino_native_v1",
                )
                seed = expected_seed(split, epoch)
                expected_alpha = np.random.default_rng(seed).random(self.images.shape[-1])
                self.assertEqual(result.seed, seed)
                np.testing.assert_array_equal(result.alpha, expected_alpha)

    def test_nonfinite_and_empty_support_inputs_are_rejected(self):
        images = self.images.copy()
        images[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            prepare_lino_native_geometry(
                images,
                self.mask,
                margin=0,
                target_resolution=512,
            )
        with self.assertRaisesRegex(ValueError, "support is empty"):
            prepare_lino_native_geometry(
                self.images,
                np.zeros_like(self.mask),
                margin=0,
                target_resolution=512,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_lino_observations(
                images,
                self.mask,
                base_seed=1,
                split="train",
                epoch=0,
                object_name="alpha.data",
                version="private_external_lino_native_v1",
            )
        with self.assertRaisesRegex(ValueError, "support is empty"):
            normalize_lino_observations(
                self.images,
                np.zeros_like(self.mask),
                base_seed=1,
                split="train",
                epoch=0,
                object_name="alpha.data",
                version="private_external_lino_native_v1",
            )

    def test_zero_normal_and_zero_prediction_remain_zero_without_nan(self):
        target_mask = self.mask.copy()
        zero_normal = np.zeros((*target_mask.shape, 3), dtype=np.float32)
        result = prepare_lino_native_geometry(
            self.images,
            self.mask,
            target_normal=zero_normal,
            target_mask=target_mask,
            margin=0,
            target_resolution=512,
        )
        self.assertTrue(np.array_equal(result.target_normal, np.zeros_like(result.target_normal)))
        self.assertTrue(np.isfinite(result.target_normal).all())

        restored = restore_lino_prediction(
            np.zeros((4, 4, 3), dtype=np.float32),
            np.asarray([8, 10, 2, 6, 3, 8]),
            source_height=8,
            source_width=10,
        )
        self.assertTrue(np.array_equal(restored, np.zeros_like(restored)))
        self.assertTrue(np.isfinite(restored).all())


if __name__ == "__main__":
    unittest.main()
