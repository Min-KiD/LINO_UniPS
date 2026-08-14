"""Contract tests for versioned LINO-native geometry and normalization."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.lino_native_preprocessing import (
    normalize_lino_observations,
    prepare_lino_native_geometry,
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


if __name__ == "__main__":
    unittest.main()
