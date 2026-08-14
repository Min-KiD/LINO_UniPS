"""Contract tests for private LINO checkpoints, exports, and publication."""

from __future__ import annotations

import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from src.training.checkpointing import (
    ALLOWED_AUTHOR_EXTRAS,
    BestMetrics,
    TrainingProgress,
    apply_artifact_retention,
    build_run_contract,
    export_inference_weights,
    load_initial_weights,
    load_resume_checkpoint,
    preflight_startup_checkpoint,
    publish_epoch_artifacts,
    save_resume_checkpoint,
)


class _TinyReleasedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _state_bytes(state: object) -> bytes:
    stream = io.BytesIO()
    torch.save(state, stream)
    return stream.getvalue()


class PrivateTrainingCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.model = _TinyReleasedModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5)
        self.contract = {
            "schema_version": 1,
            "artifact_kind": "lino_private_training_contract",
            "run_kind": "experiment",
            "comparable": True,
            "total_epochs": 4,
            "architecture_schema_sha256": "schema",
            "train_manifest_sha256": "train",
            "test_manifest_sha256": "test",
            "final_selection_manifest_sha256": "final",
            "source_revision": "source",
            "runtime_versions": {"python": "test"},
            "config_snapshot": {"seed": 7},
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_raw_pth_cannot_be_used_as_resume(self) -> None:
        path = self.root / "model.pth"
        torch.save(self.model.state_dict(), path)
        with self.assertRaisesRegex(ValueError, r"full private-training \.ckpt"):
            load_resume_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                expected_contract=self.contract,
            )

    def test_resume_restores_completed_epoch_optimizer_scheduler_and_rng(self) -> None:
        loss = self.model(torch.ones(2, 2)).sum()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        progress = TrainingProgress(completed_epoch=4, next_epoch=5, global_step=12)
        best = BestMetrics(mae=3.25, loss=0.01, epoch=3)
        checkpoint = self.root / "last.ckpt"

        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        expected_model = copy.deepcopy(self.model.state_dict())
        expected_optimizer = copy.deepcopy(self.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(self.scheduler.state_dict())
        save_resume_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=progress,
            best=best,
            contract=self.contract,
        )
        expected_random = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)

        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(10)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        random.random()
        np.random.rand()
        torch.rand(1)

        result = load_resume_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_contract=self.contract,
        )
        for name, tensor in expected_model.items():
            self.assertTrue(torch.equal(self.model.state_dict()[name], tensor))
        actual_optimizer = self.optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        self.assertEqual(actual_optimizer["state"].keys(), expected_optimizer["state"].keys())
        for key, state in expected_optimizer["state"].items():
            for state_name, expected_value in state.items():
                actual_value = actual_optimizer["state"][key][state_name]
                if isinstance(expected_value, torch.Tensor):
                    self.assertTrue(torch.equal(actual_value, expected_value))
                else:
                    self.assertEqual(actual_value, expected_value)
        self.assertEqual(self.scheduler.state_dict(), expected_scheduler)
        self.assertEqual(result.progress.next_epoch, 5)
        self.assertEqual(result.best.mae, 3.25)
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))

    def test_export_is_raw_and_strictly_round_trips(self) -> None:
        export = export_inference_weights(
            self.root / "lino_epoch_005.pth",
            model=self.model,
            model_factory=_TinyReleasedModel,
            metadata={"epoch": 5, "run_kind": "experiment", "comparable": True},
        )
        raw = torch.load(export.weights_path, map_location="cpu", weights_only=False)
        self.assertEqual(set(raw), set(self.model.state_dict()))
        self.assertNotIn("state_dict", raw)
        _TinyReleasedModel().load_state_dict(raw, strict=True)
        sidecar = json.loads(export.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(sidecar["artifact_kind"], "lino_private_inference_weights")
        self.assertEqual(sidecar["checkpoint_sha256"], export.checkpoint_sha256)

    def test_export_pair_rolls_back_when_sidecar_publication_fails(self) -> None:
        weights = self.root / "lino_epoch_006.pth"
        sidecar = weights.with_suffix(".json")
        weights.write_bytes(b"old-weights")
        sidecar.write_bytes(b"old-sidecar")
        original_publish = __import__("src.training.checkpointing", fromlist=["publish_epoch_artifacts"]).publish_epoch_artifacts
        with mock.patch(
            "src.training.checkpointing.publish_epoch_artifacts",
            side_effect=OSError("sidecar publication failure"),
        ):
            with self.assertRaisesRegex(OSError, "sidecar publication"):
                export_inference_weights(
                    weights,
                    model=self.model,
                    model_factory=_TinyReleasedModel,
                    metadata={"epoch": 6},
                )
        self.assertEqual(weights.read_bytes(), b"old-weights")
        self.assertEqual(sidecar.read_bytes(), b"old-sidecar")
        self.assertIsNotNone(original_publish)

    def test_initialization_allows_only_the_four_author_extras(self) -> None:
        state = {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}
        state[next(iter(ALLOWED_AUTHOR_EXTRAS))] = torch.ones(1)
        path = self.root / "author.pth"
        torch.save({"state_dict": state}, path)
        preflight = preflight_startup_checkpoint(
            "init_checkpoint",
            path,
            expected_schema=(
                (name, tuple(tensor.shape), str(tensor.dtype))
                for name, tensor in self.model.state_dict().items()
            ),
            expected_contract=self.contract,
        )
        load_initial_weights(preflight, model=self.model)
        self.assertEqual(preflight.load_report["author_extras"], [next(iter(ALLOWED_AUTHOR_EXTRAS))])

        state["not-an-author-extra"] = torch.ones(1)
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "unexpected|allowed"):
            preflight_startup_checkpoint(
                "init_checkpoint",
                path,
                expected_schema=self.model,
                expected_contract=self.contract,
            )

    def test_preflight_rejects_malformed_artifact_without_live_model(self) -> None:
        path = self.root / "bad.ckpt"
        path.write_bytes(b"not a torch archive")
        with self.assertRaisesRegex(ValueError, "checkpoint|torch|artifact"):
            preflight_startup_checkpoint(
                "resume",
                path,
                expected_schema=(
                    (name, tuple(tensor.shape), str(tensor.dtype))
                    for name, tensor in self.model.state_dict().items()
                ),
                expected_contract=self.contract,
            )
        self.assertTrue(torch.equal(self.model.linear.weight, self.model.linear.weight.detach()))

    def test_preflight_payload_mutation_cannot_change_authoritative_load(self) -> None:
        path = self.root / "author.pth"
        expected = self.model.linear.weight.detach().clone()
        torch.save(self.model.state_dict(), path)
        preflight = preflight_startup_checkpoint(
            "init_checkpoint",
            path,
            expected_schema=self.model,
            expected_contract=self.contract,
        )
        payload = preflight.payload
        assert payload is not None
        state = payload["state_dict"]
        state["linear.weight"].add_(100)
        load_initial_weights(preflight, model=self.model)
        self.assertTrue(torch.equal(self.model.linear.weight, expected))

    def test_preflight_rejects_symlinked_intermediate_checkpoint_path(self) -> None:
        real_root = self.root / "real"
        real_root.mkdir()
        checkpoint = real_root / "model.pth"
        torch.save(self.model.state_dict(), checkpoint)
        (self.root / "link").symlink_to(real_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink|regular|checkpoint"):
            preflight_startup_checkpoint(
                "init_checkpoint",
                self.root / "link" / "model.pth",
                expected_schema=self.model,
                expected_contract=self.contract,
            )

    def test_preflight_rejects_model_state_dtype_mismatch(self) -> None:
        path = self.root / "dtype.pth"
        state = {name: tensor.detach().double() for name, tensor in self.model.state_dict().items()}
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "dtype"):
            preflight_startup_checkpoint(
                "init_checkpoint",
                path,
                expected_schema=self.model,
                expected_contract=self.contract,
            )

    def test_preflight_rejects_malformed_optimizer_scheduler_and_rng(self) -> None:
        checkpoint = self.root / "valid.ckpt"
        save_resume_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=TrainingProgress(0, 1, 0),
            best=BestMetrics(float("inf"), float("inf"), 0),
            contract=self.contract,
        )
        valid = torch.load(checkpoint, map_location="cpu", weights_only=False)
        malformed = (
            ("optimizer_state_dict", "bad"),
            ("scheduler_state_dict", []),
            ("rng_state", {"python": "bad", "numpy": (), "torch": torch.tensor([1]), "cuda": []}),
        )
        for field, value in malformed:
            with self.subTest(field=field):
                candidate = copy.deepcopy(valid)
                candidate[field] = value
                torch.save(candidate, checkpoint)
                with self.assertRaisesRegex(ValueError, field.split("_")[0] + "|rng"):
                    preflight_startup_checkpoint(
                        "resume",
                        checkpoint,
                        expected_schema=self.model,
                        expected_contract=self.contract,
                    )

    def test_contract_allows_only_increased_total_epochs(self) -> None:
        current = dict(self.contract)
        current["total_epochs"] = 8
        self.assertEqual(build_run_contract(SimpleNamespace(epochs=8), base_contract=self.contract)["total_epochs"], 8)
        checkpoint = self.root / "last.ckpt"
        save_resume_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            progress=TrainingProgress(0, 1, 0),
            best=BestMetrics(float("inf"), float("inf"), 0),
            contract=self.contract,
        )
        load_resume_checkpoint(
            checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_contract=current,
        )
        current["seed"] = 99
        with self.assertRaisesRegex(ValueError, "contract|fingerprint"):
            load_resume_checkpoint(
                checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                expected_contract=current,
            )

    def test_publication_rolls_back_previous_bundle_on_failure(self) -> None:
        run = self.root / "run"
        run.mkdir()
        (run / "last.ckpt").write_bytes(b"old-last")
        (run / "metrics.csv").write_bytes(b"old-metrics")
        new_checkpoint = _state_bytes({"artifact_kind": "lino_private_training_checkpoint"})
        artifacts = {
            "last.ckpt": new_checkpoint,
            "metrics.csv": b"new-metrics",
            "epoch_002.ckpt": new_checkpoint,
        }
        original_replace = __import__("src.training.checkpointing", fromlist=["os"]).os.replace
        calls = {"count": 0}

        def fail_after_first(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("simulated publication failure")
            return original_replace(*args, **kwargs)

        with mock.patch("src.training.checkpointing.os.replace", side_effect=fail_after_first):
            with self.assertRaisesRegex(ValueError, "publish|artifact"):
                publish_epoch_artifacts(run, artifacts=artifacts)
        self.assertEqual((run / "last.ckpt").read_bytes(), b"old-last")
        self.assertEqual((run / "metrics.csv").read_bytes(), b"old-metrics")
        self.assertFalse((run / "epoch_002.ckpt").exists())

    def test_backup_cleanup_failure_is_postcommit_and_never_partial(self) -> None:
        run = self.root / "run-cleanup"
        run.mkdir()
        (run / "last.ckpt").write_bytes(b"old-last")
        (run / "metrics.csv").write_bytes(b"old-metrics")
        checkpoint = _state_bytes({"artifact_kind": "lino_private_training_checkpoint"})
        original_unlink = __import__("src.training.checkpointing", fromlist=["os"]).os.unlink
        calls = {"count": 0}

        def fail_second_backup_unlink(path, *args, **kwargs):
            if isinstance(path, str) and path.endswith(".bak"):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("backup cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with mock.patch("src.training.checkpointing.os.unlink", side_effect=fail_second_backup_unlink):
            result = publish_epoch_artifacts(
                run,
                artifacts={"last.ckpt": checkpoint, "metrics.csv": b"new-metrics"},
            )
        self.assertEqual(set(result), {"last.ckpt", "metrics.csv"})
        self.assertNotEqual((run / "last.ckpt").read_bytes(), b"old-last")
        self.assertEqual((run / "metrics.csv").read_bytes(), b"new-metrics")

    def test_retention_keeps_milestones_and_latest_aliases(self) -> None:
        run = self.root / "run"
        run.mkdir()
        (run / "last.ckpt").write_bytes(b"last")
        (run / "best_validation.ckpt").write_bytes(b"best")
        for epoch in (1, 2, 3):
            for suffix in (".ckpt", ".pth", ".json"):
                (run / f"lino_epoch_{epoch:03d}{suffix}").write_bytes(b"artifact")
        removed = apply_artifact_retention(run, keep_milestone_epochs=(2,), latest_epoch=3)
        self.assertIn(run / "lino_epoch_001.ckpt", removed)
        self.assertTrue((run / "lino_epoch_002.ckpt").exists())
        self.assertTrue((run / "lino_epoch_003.ckpt").exists())
        self.assertTrue((run / "last.ckpt").exists())


if __name__ == "__main__":
    unittest.main()
