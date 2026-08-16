import contextlib
import importlib.util
import inspect
import math
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from src.training.model_adapter import (
    EncodedPrivateBatch,
    decode_private_chunks,
    encode_private_batch,
    released_state_schema,
)


@contextlib.contextmanager
def _released_model_with_dependency_stubs():
    """Load the released class while stubbing only absent optional packages."""

    previous_modules = {}
    stubs = {}
    if "torchmetrics" not in sys.modules and importlib.util.find_spec("torchmetrics") is None:
        metrics = types.ModuleType("torchmetrics")

        class _MeanMetric(nn.Module):
            def forward(self, *args, **kwargs):
                del args, kwargs
                return torch.tensor(0.0)

        metrics.MeanMetric = _MeanMetric
        stubs["torchmetrics"] = metrics
    if "pytorch_lightning" not in sys.modules and importlib.util.find_spec("pytorch_lightning") is None:
        lightning = types.ModuleType("pytorch_lightning")
        lightning.LightningModule = nn.Module
        stubs["pytorch_lightning"] = lightning
    for name, module in stubs.items():
        previous_modules[name] = sys.modules.get(name)
        sys.modules[name] = module
    original_linspace = torch.linspace

    def _cpu_linspace(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)

    torch.linspace = _cpu_linspace
    try:
        from src.models.Net_module import LiNo_UniPS

        yield LiNo_UniPS
    finally:
        torch.linspace = original_linspace
        for name in stubs:
            if previous_modules[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_modules[name]


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


class _RecordingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.inputs = []

    def forward(self, features):
        self.inputs.append(features.detach().clone())
        return super().forward(features)


class _FakeReleasedLino(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = _FakeImageEncoder()
        self.img_embedding = nn.Linear(3, 4)
        self.glc_upsample = _RecordingLinear(4, 4)
        self.glc_aggregation = nn.Linear(4, 5)
        self.regressor = _FakeRegressor()


class _DtypeSensitiveImageEncoder(_FakeImageEncoder):
    def forward(self, images, light_counts, canonical_resolution):
        if images.dtype != torch.bfloat16:
            raise RuntimeError("released encoder requires bfloat16")
        return super().forward(images.float(), light_counts, canonical_resolution)


class _DtypeSensitiveEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, pixels):
        if pixels.dtype != torch.bfloat16:
            raise RuntimeError("released decoder requires bfloat16")
        return F.pad((pixels.float() * self.scale).to(torch.bfloat16), (0, 1))


class _DtypeSensitiveFeatureBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, features):
        if features.dtype != torch.bfloat16:
            raise RuntimeError("released decoder requires bfloat16")
        return (features.float() * self.scale).to(torch.bfloat16)


class _DtypeSensitiveRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4, 3))

    def forward(self, features, num_sample_set):
        del num_sample_set
        if features.dtype != torch.bfloat16:
            raise RuntimeError("released decoder requires bfloat16")
        prediction = (features.float()[..., :3] * self.scale[0]).mean(dim=1).unsqueeze(0)
        prediction = prediction.to(torch.bfloat16)
        return prediction, None, None, torch.zeros_like(prediction[..., :1])


class _DtypeSensitiveReleasedLino(_FakeReleasedLino):
    def __init__(self):
        super().__init__()
        self.image_encoder = _DtypeSensitiveImageEncoder()
        self.img_embedding = _DtypeSensitiveEmbedding()
        self.glc_upsample = _DtypeSensitiveFeatureBlock(4)
        self.glc_aggregation = _DtypeSensitiveFeatureBlock(4)
        self.regressor = _DtypeSensitiveRegressor()


class _CpuReleasedLino(_FakeReleasedLino):
    pass


_CpuReleasedLino.__name__ = "LiNo_UniPS"


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
        self.assertEqual(tuple(model.image_encoder.last_images.shape), (12, 3, 512, 512))
        self.assertEqual(model.image_encoder.last_light_counts, (6, 6))
        expected = self.images().permute(0, 4, 1, 2, 3) * self.model_mask().unsqueeze(1)
        self.assertTrue(torch.equal(model.image_encoder.last_images, expected.reshape(12, 3, 512, 512)))
        chunks = (
            (torch.tensor([0, 3]), torch.tensor([9])),
            (torch.tensor([1]), torch.tensor([4, 8])),
        )
        decoded = decode_private_chunks(model, encoded, chunks)
        self.assertEqual(released_state_schema(model), before)
        self.assertEqual(tuple(decoded[0][0].prediction.shape), (2, 3))
        self.assertEqual(tuple(decoded[0][1].indices.tolist()), (9,))
        self.assertEqual(tuple(decoded[1][1].indices.tolist()), (4, 8))
        self.assertEqual(tuple(decoded[0][0].indices.tolist()), (0, 3))
        expected_calls = (
            (0, torch.tensor([0, 3])),
            (0, torch.tensor([9])),
            (1, torch.tensor([1])),
            (1, torch.tensor([4, 8])),
        )
        for call, (object_number, indices) in zip(model.glc_upsample.inputs, expected_calls):
            object_observations = encoded.observations[object_number].permute(1, 2, 3, 0).reshape(-1, 6, 3)
            object_glc = encoded.glc[object_number * 6 : (object_number + 1) * 6]
            object_glc = object_glc.permute(2, 3, 0, 1).reshape(-1, 6, encoded.glc.shape[1])
            expected_call = model.img_embedding(object_observations[indices]) + object_glc[indices]
            self.assertTrue(torch.allclose(call, expected_call))
        for object_chunks in decoded:
            for chunk in object_chunks:
                self.assertTrue(torch.isfinite(chunk.prediction).all())
                self.assertTrue(torch.allclose(chunk.prediction.norm(dim=-1), torch.ones(chunk.prediction.shape[0]), atol=1e-5))

    def test_adapter_accepts_lino_256_with_canonical_128(self):
        model = _FakeReleasedLino()
        images = torch.linspace(0.0, 1.0, 3 * 256 * 256 * 6).reshape(
            1, 3, 256, 256, 6
        )
        mask = torch.ones(1, 1, 256, 256)

        encoded = encode_private_batch(
            model,
            images,
            mask,
            canonical_resolution=128,
        )

        self.assertEqual(tuple(model.image_encoder.last_images.shape), (6, 3, 256, 256))
        self.assertEqual(model.image_encoder.last_canonical_resolution, 128)
        self.assertEqual(tuple(encoded.glc.shape[-2:]), (256, 256))

    def test_loss_backpropagates_to_all_used_released_components(self):
        model = _FakeReleasedLino()
        encoded = encode_private_batch(model, self.images(), self.model_mask(), canonical_resolution=256)
        decoded = decode_private_chunks(model, encoded, ((torch.tensor([0, 1]),), (torch.tensor([2, 3]),)))
        sum(chunk.prediction.sum() for object_chunks in decoded for chunk in object_chunks).backward()
        for name in ("image_encoder", "img_embedding", "glc_upsample", "glc_aggregation", "regressor"):
            self.assertTrue(any(parameter.grad is not None for parameter in getattr(model, name).parameters()))

    def test_activation_checkpointing_routes_encoder_smoothing_and_decoder_chunks(self):
        model = _FakeReleasedLino()
        images = self.images()[:1]
        mask = self.model_mask()[:1]
        calls = []

        def recording_checkpoint(function, *args, **kwargs):
            calls.append((function.__name__, kwargs))
            return function(*args)

        with (
            mock.patch("src.training.model_adapter.gauss_filter", return_value=nn.Identity()),
            mock.patch(
                "src.training.model_adapter.torch_checkpoint",
                side_effect=recording_checkpoint,
            ),
        ):
            encoded = encode_private_batch(
                model,
                images,
                mask,
                canonical_resolution=256,
                activation_checkpointing=True,
            )
            decode_private_chunks(
                model,
                encoded,
                ((torch.tensor([0, 1]), torch.tensor([2, 3])),),
                activation_checkpointing=True,
            )

        self.assertEqual(
            [name for name, _kwargs in calls],
            ["_encode_glc", "_smooth_chunk", "_decode_chunk_tensor", "_decode_chunk_tensor"],
        )
        self.assertTrue(all(kwargs == {"use_reentrant": False} for _name, kwargs in calls))

    def test_activation_checkpointing_preserves_predictions_gradients_and_state_schema(self):
        reference = _FakeReleasedLino()
        initial_state = {
            name: value.detach().clone() for name, value in reference.state_dict().items()
        }

        def run(enabled):
            model = _FakeReleasedLino()
            model.load_state_dict(initial_state, strict=True)
            before = released_state_schema(model)
            with mock.patch(
                "src.training.model_adapter.gauss_filter", return_value=nn.Identity()
            ):
                encoded = encode_private_batch(
                    model,
                    self.images()[:1],
                    self.model_mask()[:1],
                    canonical_resolution=256,
                    activation_checkpointing=enabled,
                )
                decoded = decode_private_chunks(
                    model,
                    encoded,
                    ((torch.tensor([0, 1, 2, 3]),),),
                    activation_checkpointing=enabled,
                )
                prediction = decoded[0][0].prediction
                weights = torch.tensor([0.5, -0.25, 1.25], dtype=prediction.dtype)
                (prediction * weights).sum().backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            return prediction.detach(), gradients, before, released_state_schema(model)

        plain = run(False)
        checkpointed = run(True)

        self.assertTrue(torch.allclose(plain[0], checkpointed[0], atol=1.0e-6, rtol=1.0e-5))
        self.assertEqual(plain[1].keys(), checkpointed[1].keys())
        for name in plain[1]:
            self.assertTrue(
                torch.allclose(plain[1][name], checkpointed[1][name], atol=1.0e-6, rtol=1.0e-5),
                name,
            )
        self.assertEqual(plain[2], plain[3])
        self.assertEqual(checkpointed[2], checkpointed[3])
        self.assertEqual(plain[2], checkpointed[2])

    def test_activation_checkpointing_is_bypassed_when_gradients_are_disabled(self):
        model = _FakeReleasedLino()
        with (
            torch.no_grad(),
            mock.patch("src.training.model_adapter.gauss_filter", return_value=nn.Identity()),
            mock.patch("src.training.model_adapter.torch_checkpoint") as checkpoint,
        ):
            encoded = encode_private_batch(
                model,
                self.images()[:1],
                self.model_mask()[:1],
                canonical_resolution=256,
                activation_checkpointing=True,
            )
            decode_private_chunks(
                model,
                encoded,
                ((torch.tensor([0, 1]),),),
                activation_checkpointing=True,
            )

        checkpoint.assert_not_called()

    def test_adapter_signature_has_no_ground_truth_argument(self):
        self.assertNotIn("target_normal", inspect.signature(encode_private_batch).parameters)
        self.assertNotIn("target_normal", inspect.signature(decode_private_chunks).parameters)

    def test_real_released_schema_has_no_synthetic_auxiliary_parameters(self):
        with _released_model_with_dependency_stubs() as LiNo_UniPS:
            with mock.patch.object(torch, "linspace", wraps=torch.linspace):
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

    def test_lino_256_runtime_geometry_does_not_change_released_state_schema(self):
        with _released_model_with_dependency_stubs() as LiNo_UniPS:
            with torch.device("meta"):
                native = LiNo_UniPS(pixel_samples=2048)
                lino_256 = LiNo_UniPS(
                    pixel_samples=2048,
                    model_resolution=256,
                    canonical_resolution=128,
                )
        self.assertEqual(released_state_schema(native), released_state_schema(lino_256))
        self.assertEqual(lino_256.model_resolution, 256)
        self.assertEqual(lino_256.canonical_resolution, 128)

    def test_cuda_dtype_bridge_is_internal_to_encoder_and_decoder(self):
        model = _DtypeSensitiveReleasedLino()
        encoded = None
        fake_cuda = torch.device("cuda")
        with mock.patch("src.training.model_adapter._model_device", return_value=fake_cuda):
            with mock.patch("src.training.model_adapter.torch.autocast", return_value=contextlib.nullcontext()):
                encoded = encode_private_batch(model, self.images(), self.model_mask(), canonical_resolution=256)
                decoded = decode_private_chunks(model, encoded, ((torch.tensor([0, 1]),), (torch.tensor([2, 3]),)))
        self.assertEqual(tuple(decoded[0][0].prediction.shape), (2, 3))

    def test_cpu_released_model_fails_with_clear_dtype_contract_error(self):
        model = _CpuReleasedLino()
        with self.assertRaisesRegex(RuntimeError, "CUDA.*bfloat16|unsupported"):
            encode_private_batch(model, self.images(), self.model_mask(), canonical_resolution=256)

    def test_invalid_layout_dtype_masks_and_indices_are_rejected(self):
        model = _FakeReleasedLino()
        with self.assertRaises(ValueError):
            encode_private_batch(model, self.images()[:, :, :-1], self.model_mask(), canonical_resolution=256)
        with self.assertRaises(TypeError):
            encode_private_batch(model, self.images().to(torch.float64), self.model_mask(), canonical_resolution=256)
        nonbinary = self.model_mask()
        nonbinary[:, :, 0, 0] = 0.5
        with self.assertRaises(ValueError):
            encode_private_batch(model, self.images(), nonbinary, canonical_resolution=256)
        empty = torch.zeros_like(self.model_mask())
        with self.assertRaises(ValueError):
            encode_private_batch(model, self.images(), empty, canonical_resolution=256)
        encoded = EncodedPrivateBatch(
            observations=torch.zeros(1, 3, 512, 512, 6),
            glc=torch.zeros(6, 4, 512, 512),
            light_counts=(6,),
            height=512,
            width=512,
        )
        for bad in (torch.tensor([1, 1]), torch.tensor([-1]), torch.tensor([512 * 512])):
            with self.assertRaises(ValueError):
                decode_private_chunks(model, encoded, ((bad,),))

    def test_empty_trusted_chunks_and_nonfinite_intermediates_are_rejected(self):
        model = _FakeReleasedLino()
        encoded = EncodedPrivateBatch(
            observations=torch.zeros(1, 3, 512, 512, 6),
            glc=torch.zeros(6, 4, 512, 512),
            light_counts=(6,),
            height=512,
            width=512,
        )
        with self.assertRaises(ValueError):
            decode_private_chunks(model, encoded, ((torch.empty(0, dtype=torch.long),),))
        model.glc_upsample.weight.data.fill_(float("nan"))
        with self.assertRaises(ValueError):
            decode_private_chunks(model, encoded, ((torch.tensor([0]),),))


if __name__ == "__main__":
    unittest.main()
