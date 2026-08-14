import unittest

import torch
from torch import nn

from src.training.model_adapter import DecodedNormalChunk
from src.training.objective import (
    component_sse_batch,
    finite_gradients_or_raise,
    plan_target_chunks,
    sampled_angular_mae,
)


def _chunk(indices, prediction):
    return DecodedNormalChunk(
        indices=torch.tensor(indices, dtype=torch.long),
        prediction=torch.tensor(prediction, dtype=torch.float32),
    )


class PrivateTrainingObjectiveTests(unittest.TestCase):
    def test_target_chunks_never_include_external_only_halo(self):
        target = torch.zeros(1, 8, 8)
        target[:, 2:6, 2:6] = 1
        chunks = plan_target_chunks(
            target,
            pixel_samples=5,
            pixel_budget=11,
            base_seed=20260710,
            split="train",
            epoch=4,
            object_name="alpha.data",
        )
        chosen = torch.cat(chunks)
        valid = torch.nonzero(target.reshape(-1) > 0, as_tuple=False).flatten()
        self.assertTrue(set(chosen.tolist()).issubset(set(valid.tolist())))
        self.assertEqual(chosen.numel(), 11)
        self.assertTrue(all(0 < chunk.numel() <= 5 for chunk in chunks))
        self.assertEqual(chosen.unique().numel(), chosen.numel())

    def test_chunk_plan_replays_and_validation_is_epoch_invariant(self):
        mask = torch.ones(1, 6, 6)
        first = plan_target_chunks(
            mask,
            pixel_samples=4,
            pixel_budget=9,
            base_seed=7,
            split="train",
            epoch=2,
            object_name="a.data",
        )
        second = plan_target_chunks(
            mask,
            pixel_samples=4,
            pixel_budget=9,
            base_seed=7,
            split="train",
            epoch=2,
            object_name="a.data",
        )
        validation0 = plan_target_chunks(
            mask,
            pixel_samples=4,
            pixel_budget=9,
            base_seed=7,
            split="test",
            epoch=0,
            object_name="a.data",
        )
        validation9 = plan_target_chunks(
            mask,
            pixel_samples=4,
            pixel_budget=9,
            base_seed=7,
            split="test",
            epoch=9,
            object_name="a.data",
        )
        self.assertEqual([chunk.tolist() for chunk in first], [chunk.tolist() for chunk in second])
        self.assertEqual(
            [chunk.tolist() for chunk in validation0],
            [chunk.tolist() for chunk in validation9],
        )

    def test_target_planning_rejects_empty_invalid_and_nonfinite_support(self):
        common = {
            "base_seed": 7,
            "split": "train",
            "epoch": 0,
            "object_name": "empty.data",
        }
        with self.assertRaisesRegex(ValueError, "empty"):
            plan_target_chunks(torch.zeros(1, 2, 2), pixel_samples=1, pixel_budget=1, **common)
        with self.assertRaises(ValueError):
            plan_target_chunks(torch.ones(1, 2, 2), pixel_samples=0, pixel_budget=1, **common)
        with self.assertRaises(ValueError):
            plan_target_chunks(torch.ones(1, 2, 2), pixel_samples=1, pixel_budget=0, **common)
        with self.assertRaises(ValueError):
            plan_target_chunks(
                torch.tensor([[[1.0, float("nan")], [0.0, 1.0]]]),
                pixel_samples=1,
                pixel_budget=1,
                **common,
            )

    def test_validation_epoch_is_still_type_checked(self):
        mask = torch.ones(1, 2, 2)
        common = {
            "pixel_samples": 1,
            "pixel_budget": 1,
            "base_seed": 7,
            "split": "test",
            "object_name": "alpha.data",
        }
        with self.assertRaises(TypeError):
            plan_target_chunks(mask, epoch=True, **common)
        with self.assertRaises(ValueError):
            plan_target_chunks(mask, epoch=-1, **common)

    def test_component_sse_is_object_balanced_and_uses_exact_indices(self):
        predictions = (
            (_chunk([0], [[1.0, 0.0, 0.0]]),),
            (_chunk([0, 1, 2], [[0.0, 0.0, 0.0]] * 3),),
        )
        targets = torch.zeros(2, 3, 1, 3)
        targets[0, 1, 0, 0] = 1.0
        loss = component_sse_batch(predictions, targets)
        expected = torch.tensor(1.0)
        self.assertTrue(torch.allclose(loss, expected))

        # Pixel 0 and pixel 2 have different targets; only the exact supplied
        # indices may contribute, not the unsampled pixel 1.
        exact_predictions = ((_chunk([2], [[1.0, 0.0, 0.0]]),),)
        exact_targets = torch.zeros(1, 3, 1, 3)
        exact_targets[0, 0, 0, 0] = 50.0
        exact_targets[0, 0, 0, 2] = 2.0
        self.assertTrue(torch.allclose(component_sse_batch(exact_predictions, exact_targets), torch.tensor(1.0)))

    def test_component_sse_rejects_bad_batch_chunk_and_nonfinite_values(self):
        targets = torch.zeros(1, 3, 1, 2)
        with self.assertRaises(ValueError):
            component_sse_batch(
                (
                    (_chunk([0], [[0.0, 0.0, 0.0]]),),
                    (_chunk([0], [[0.0, 0.0, 0.0]]),),
                ),
                targets,
            )
        with self.assertRaises(ValueError):
            component_sse_batch(((),), targets)
        with self.assertRaises(ValueError):
            component_sse_batch(((_chunk([2], [[0.0, 0.0, 0.0]]),),), targets)
        nonfinite = _chunk([0], [[float("nan"), 0.0, 0.0]])
        with self.assertRaises(ValueError):
            component_sse_batch(((nonfinite,),), targets)
        bad_target = targets.clone()
        bad_target[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            component_sse_batch(((_chunk([0], [[0.0, 0.0, 0.0]]),),), bad_target)
        duplicate = (
            (_chunk([0], [[0.0, 0.0, 0.0]]), _chunk([0], [[0.0, 0.0, 0.0]])),
        )
        with self.assertRaises(ValueError):
            component_sse_batch(duplicate, targets)
        wrong_dtype = DecodedNormalChunk(
            indices=torch.tensor([0], dtype=torch.float32),
            prediction=torch.zeros(1, 3),
        )
        with self.assertRaises(TypeError):
            component_sse_batch(((wrong_dtype,),), targets)

    def test_component_sse_uses_exact_prediction_magnitude_and_backpropagates(self):
        model = nn.Module()
        model.vector = nn.Parameter(torch.tensor([[2.0, 0.0, 0.0]]))
        predictions = ((DecodedNormalChunk(torch.tensor([0]), model.vector),),)
        targets = torch.zeros(1, 3, 1, 1)
        loss = component_sse_batch(predictions, targets)
        self.assertTrue(torch.allclose(loss, torch.tensor(4.0)))
        loss.backward()
        self.assertTrue(torch.allclose(model.vector.grad, torch.tensor([[4.0, 0.0, 0.0]])))
        finite_gradients_or_raise(model)

    def test_sampled_angular_mae_is_normalized_finite_and_in_degrees(self):
        targets = torch.zeros(2, 3, 1, 1)
        targets[:, 2, 0, 0] = 1.0
        identity = ((
            _chunk([0], [[0.0, 0.0, 4.0]]),
        ), (
            _chunk([0], [[0.0, 0.0, 1.0]]),
        ))
        self.assertTrue(torch.isfinite(sampled_angular_mae(identity, targets)))
        self.assertTrue(torch.allclose(sampled_angular_mae(identity, targets), torch.tensor(0.0), atol=1e-5))

        orthogonal = ((
            _chunk([0], [[1.0, 0.0, 0.0]]),
        ), (
            _chunk([0], [[0.0, 1.0, 0.0]]),
        ))
        self.assertTrue(torch.allclose(sampled_angular_mae(orthogonal, targets), torch.tensor(90.0), atol=1e-5))

    def test_sampled_angular_mae_rejects_zero_and_nonfinite_normals(self):
        targets = torch.zeros(1, 3, 1, 1)
        targets[:, 2, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            sampled_angular_mae(((_chunk([0], [[0.0, 0.0, 0.0]]),),), targets)
        with self.assertRaises(ValueError):
            sampled_angular_mae(((_chunk([0], [[float("inf"), 0.0, 0.0]]),),), targets)

    def test_finite_gradients_accepts_finite_and_rejects_nonfinite(self):
        model = nn.Linear(2, 1)
        model.weight.grad = torch.ones_like(model.weight)
        model.bias.grad = None
        finite_gradients_or_raise(model)
        model.weight.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "weight"):
            finite_gradients_or_raise(model)


if __name__ == "__main__":
    unittest.main()
