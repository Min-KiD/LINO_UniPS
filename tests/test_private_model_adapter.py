import inspect
import math
import unittest

import torch
from torch import nn

from src.training.model_adapter import (
    decode_private_chunks,
    encode_private_batch,
    released_state_schema,
)

try:
    from src.models.Net_module import LiNo_UniPS
except ModuleNotFoundError:  # Optional released-model runtime dependencies are absent on CPU CI.
    LiNo_UniPS = None


class _FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, images, light_counts, canonical_resolution):
        self.last_images = images
        self.last_light_counts = tuple(int(value) for value in light_counts)
        self.last_canonical_resolution = int(canonical_resolution)
        feature = self.projection(images.mean(dim=(-2, -1)))
        return feature[:, :, None, None].expand(-1, -1, images.shape[-2], images.shape[-1]), object()


class _FakeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5, 3)

    def forward(self, features, num_sample_set):
        del num_sample_set
        prediction = self.projection(features.mean(dim=1)).unsqueeze(0)
        return prediction, None, None, torch.zeros(
            1, features.shape[0], 1, device=features.device, dtype=features.dtype
        )


class _FakeReleasedLino(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = _FakeImageEncoder()
        self.img_embedding = nn.Linear(3, 4)
        self.glc_upsample = nn.Linear(4, 4)
        self.glc_aggregation = nn.Linear(4, 5)
        self.regressor = _FakeRegressor()


class PrivateModelAdapterTests(unittest.TestCase):
    @staticmethod
    def images():
        return torch.linspace(0.0, 1.0, 2 * 3 * 512 * 512 * 6).reshape(2, 3, 512, 512, 6)

    @staticmethod
    def model_mask():
        mask = torch.ones(2, 1, 512, 512)
        mask[:, :, :2, :2] = 0
        return mask

    def test_adapter_does_not_register_or_rename_model_parameters(self):
        model = _FakeReleasedLino()
        before = released_state_schema(model)
        encoded = encode_private_batch(model, self.images(), self.model_mask(), canonical_resolution=256)
        chunks = ((torch.tensor([0, 3, 9]),), (torch.tensor([1, 4, 8]),))
        decoded = decode_private_chunks(model, encoded, chunks)
        self.assertEqual(released_state_schema(model), before)
        self.assertEqual(tuple(decoded[0][0].prediction.shape), (3, 3))
        self.assertEqual(tuple(decoded[0][0].indices.tolist()), (0, 3, 9))

    def test_loss_backpropagates_to_all_used_released_components(self):
        model = _FakeReleasedLino()
        encoded = encode_private_batch(model, self.images(), self.model_mask(), canonical_resolution=256)
        decoded = decode_private_chunks(model, encoded, ((torch.tensor([0, 1]),), (torch.tensor([2, 3]),)))
        sum(chunk.prediction.sum() for object_chunks in decoded for chunk in object_chunks).backward()
        for name in ("image_encoder", "img_embedding", "glc_upsample", "glc_aggregation", "regressor"):
            self.assertTrue(any(parameter.grad is not None for parameter in getattr(model, name).parameters()))

    def test_adapter_signature_has_no_ground_truth_argument(self):
        self.assertNotIn("target_normal", inspect.signature(encode_private_batch).parameters)
        self.assertNotIn("target_normal", inspect.signature(decode_private_chunks).parameters)

    def test_real_released_schema_has_no_synthetic_auxiliary_parameters(self):
        if LiNo_UniPS is None:
            self.skipTest("released LiNo_UniPS runtime dependencies are unavailable")
        with torch.device("meta"):
            model = LiNo_UniPS(pixel_samples=2048)
        schema = released_state_schema(model)
        self.assertEqual(len(schema), 605)
        self.assertEqual(sum(math.prod(shape) for _name, shape, _dtype in schema), 82_056_900)
        self.assertFalse(
            any(
                name.startswith(("hdri_encoder.", "env_feature_proj.", "env_light_head.", "point_light_align.", "area_light_align."))
                for name, _shape, _dtype in schema
            )
        )


if __name__ == "__main__":
    unittest.main()
