"""Focused contracts for the explicit private LINO trainer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import weakref
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.training.config import PrivateTrainConfig
from src.training.private_manifest import PrivateObjectRecord, PrivateSplitManifest
from src.training.trainer import (
    create_optimizer_scheduler,
    run_private_training,
    validate_epoch,
)


class _Encoder(nn.Module):
    def __init__(self, owner: "_TinyReleasedModel") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", weakref.ref(owner))

    def forward(self, values: torch.Tensor, counts, canonical_resolution: int):
        del counts, canonical_resolution
        return values * self._owner().scale, None


class _Regressor(nn.Module):
    def forward(self, values: torch.Tensor, count: int):
        del count
        return values.mean(dim=1), None


class _TinyReleasedModel(nn.Module):
    """Small released-shape façade that remains differentiable on CPU."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.image_encoder = _Encoder(self)
        self.img_embedding = nn.Identity()
        self.glc_upsample = nn.Identity()
        self.glc_aggregation = nn.Identity()
        self.regressor = _Regressor()
        self.model_step_keys: set[str] = set()
        self.training_during_model_step = True
        self.grad_enabled_during_model_step = True

    def model_step(self, batch):
        self.model_step_keys = set(batch)
        self.training_during_model_step = self.training
        self.grad_enabled_during_model_step = torch.is_grad_enabled()
        # A released model returns BCHW normals.  Keep this tiny and use the
        # first channel as a deterministic normal target in the validation
        # callback below.
        return torch.zeros(
            batch["imgs"].shape[0],
            3,
            batch["imgs"].shape[2],
            batch["imgs"].shape[3],
            dtype=torch.float32,
        )


class _TinyDataset(Dataset):
    def __init__(self, samples, *, split: str) -> None:
        self.samples = list(samples)
        self.split = split
        self.epochs: list[int] = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def _sample(name: str, *, value: float = 0.1) -> dict[str, object]:
    imgs = torch.full((3, 512, 512, 6), value, dtype=torch.float32)
    target_normal = torch.zeros((3, 512, 512), dtype=torch.float32)
    target_normal[2] = 1.0
    target_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
    target_mask[:, 0, 0] = 1.0
    source_normal = torch.zeros((3, 256, 256), dtype=torch.float32)
    source_normal[2] = 1.0
    source_mask = torch.zeros((1, 256, 256), dtype=torch.float32)
    source_mask[:, 0, 0] = 1.0
    return {
        "imgs": imgs,
        "model_mask": torch.ones((1, 512, 512), dtype=torch.float32),
        "target_normal": target_normal,
        "target_mask": target_mask,
        "source_target_normal": source_normal,
        "source_target_mask": source_mask,
        "source_model_mask": torch.ones((1, 256, 256), dtype=torch.float32),
        "roi": torch.tensor([256, 256, 0, 256, 0, 256], dtype=torch.int64),
        "metadata": {
            "object_name": name,
            "source_geometry": {"height": 256, "width": 256},
            "roi": [256, 256, 0, 256, 0, 256],
        },
    }


class PrivateTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.save_dir = self.root / "runs"
        self.train_root = self.root / "train"
        self.test_root = self.root / "test"
        self.train_root.mkdir()
        self.test_root.mkdir()
        self.selection = self.root / "selected_lights.json"
        self.selection.write_text(
            json.dumps({"object.data": [f"image{i:03d}.exr" for i in range(16)]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def config(self, *, epochs: int = 2, startup_mode: str = "cold_start", resume=None):
        return PrivateTrainConfig(
            train_dir=self.train_root,
            test_dir=self.test_root,
            save_dir=self.save_dir,
            startup_mode=startup_mode,
            init_checkpoint=None,
            resume_checkpoint=resume,
            final_selection_manifest=self.selection,
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            normal_filenames=("local_normal.exr",),
            external_mask_filename="binary_mask.exr",
            normal_encoding="unsigned",
            expected_source_geometry=(256, 256),
            mask_policy="external",
            mask_margin=8,
            max_image_num=6,
            light_selection="seeded",
            seed=20260710,
            preprocessing_version="private_external_lino_native_v1",
            max_image_resolution=512,
            canonical_resolution=256,
            pixel_samples=1,
            train_pixel_budget=1,
            precision="fp32",
            device="cpu",
            deterministic=False,
            epochs=epochs,
            train_batch_size=1,
            train_workers=0,
            test_workers=0,
            learning_rate=0.01,
            weight_decay=0.0,
            adamw_betas=(0.9, 0.98),
            scheduler_step_size=1,
            scheduler_gamma=0.5,
            save_every_epochs=1,
            keep_milestone_epochs=(1, 2),
        )

    def manifests(self, _config, split):
        name = "object.data"
        record = PrivateObjectRecord(
            name=name,
            relative_dir=name,
            height=256,
            width=256,
            observation_files=tuple(f"image{i:03d}.exr" for i in range(16)),
            observation_sha256=tuple(f"hash-{i}" for i in range(16)),
            normal_file="local_normal.exr",
            normal_sha256="normal-hash",
            mask_file="binary_mask.exr",
            mask_sha256="mask-hash",
            gt_valid_pixels=1,
            mask_valid_pixels=1,
            mask_only_pixels=0,
        )
        return PrivateSplitManifest(1, split, str(self.train_root if split == "train" else self.test_root), (record,))

    def datasets(self, _config, manifest, *, split):
        return _TinyDataset([_sample(record.name) for record in manifest.objects], split=split)

    def dependencies(self):
        return {
            "manifest_builder": self.manifests,
            "dataset_factory": self.datasets,
            "model_factory": _TinyReleasedModel,
            "schema_provider": lambda: (("scale", (), "torch.float32"),),
        }

    def test_both_split_preflights_finish_before_model_factory(self):
        events = []

        def manifest_builder(config, split):
            events.append(f"preflight:{split}")
            return self.manifests(config, split)

        class ModelCreated(RuntimeError):
            pass

        def model_factory():
            events.append("model")
            raise ModelCreated

        with self.assertRaises(ModelCreated):
            run_private_training(
                self.config(epochs=1),
                model_factory=model_factory,
                manifest_builder=manifest_builder,
                schema_provider=lambda: (("scale", (), "torch.float32"),),
            )
        self.assertEqual(events, ["preflight:train", "preflight:test", "model"])

    def test_two_epoch_cold_start_writes_metrics_last_best_and_raw_exports(self):
        summary = run_private_training(self.config(epochs=2), **self.dependencies())
        with (summary.run_dir / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])
        self.assertTrue((summary.run_dir / "checkpoints/last.ckpt").is_file())
        self.assertTrue((summary.run_dir / "checkpoints/best_validation.ckpt").is_file())
        self.assertTrue((summary.run_dir / "exports/lino_epoch_002.pth").is_file())
        self.assertTrue((summary.run_dir / "exports/lino_best_validation.pth").is_file())
        self.assertTrue((summary.run_dir / "config.resolved.yaml").is_file())
        self.assertTrue((summary.run_dir / "data_contract.json").is_file())

    def test_resume_appends_epoch_without_repeating_metrics(self):
        first = run_private_training(self.config(epochs=1), **self.dependencies())
        resumed = run_private_training(
            self.config(epochs=2, startup_mode="resume", resume=first.last_checkpoint),
            **self.dependencies(),
        )
        with (resumed.run_dir / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])

    def test_smoke_mode_is_isolated_and_non_comparable(self):
        summary = run_private_training(self.config(epochs=1), smoke=True, **self.dependencies())
        self.assertEqual(summary.run_dir, self.config().save_dir / "smoke")
        sidecar = json.loads((summary.run_dir / "exports/lino_epoch_001.json").read_text())
        self.assertEqual(sidecar["run_kind"], "smoke")
        self.assertFalse(sidecar["comparable"])

    def test_validation_uses_released_mask_key_and_eval_mode(self):
        model = _TinyReleasedModel()
        loader = [{**_sample("object.data"), "imgs": _sample("object.data")["imgs"].unsqueeze(0), "model_mask": _sample("object.data")["model_mask"].unsqueeze(0), "source_target_normal": _sample("object.data")["source_target_normal"].unsqueeze(0), "source_target_mask": _sample("object.data")["source_target_mask"].unsqueeze(0), "metadata": [_sample("object.data")["metadata"]]}]
        validate_epoch(model, loader, self.config(), source_predictor=lambda prediction, batch: prediction)
        self.assertEqual(model.model_step_keys, {"imgs", "mask"})
        self.assertFalse(model.training_during_model_step)
        self.assertFalse(model.grad_enabled_during_model_step)

    def test_bad_startup_checkpoint_fails_before_live_model_or_cuda(self):
        events = []
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            run_private_training(
                self.config(startup_mode="init_checkpoint"),
                schema_provider=lambda: (("scale", (), "torch.float32"),),
                model_factory=lambda: events.append("live_model"),
                device_resolver=lambda *_: events.append("cuda"),
                manifest_builder=self.manifests,
            )
        self.assertEqual(events, [])

    def test_optimizer_scheduler_uses_configured_values(self):
        model = _TinyReleasedModel()
        optimizer, scheduler = create_optimizer_scheduler(model, self.config())
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(scheduler.step_size, 1)


if __name__ == "__main__":
    unittest.main()
