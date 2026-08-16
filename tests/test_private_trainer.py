"""Focused contracts for the explicit private LINO trainer."""

from __future__ import annotations

import csv
import json
import contextlib
from dataclasses import replace
from io import StringIO
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.training.config import PrivateTrainConfig
from src.training.private_manifest import PrivateSourceIndexRecord, PrivateSplitIndex
from src.training.trainer import (
    _load_metric_rows,
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


class _NumpyRandomValidationModel(_TinyReleasedModel):
    def model_step(self, batch):
        super().model_step(batch)
        values = np.random.random((batch["imgs"].shape[0], 3, 512, 512))
        return torch.from_numpy(values.astype(np.float32))


class _DtypeCheckedValidationModel(_TinyReleasedModel):
    def __init__(self):
        super().__init__()
        self.autocast_seen = False

    def model_step(self, batch):
        del batch
        self.autocast_seen = bool(getattr(self, "_autocast_marker", False))
        if not self.autocast_seen:
            raise RuntimeError("validation requires BF16 autocast")
        return torch.zeros(1, 3, 512, 512)


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
        record = PrivateSourceIndexRecord(
            name=name,
            relative_dir=name,
            height=256,
            width=256,
            observation_files=tuple(f"image{i:03d}.exr" for i in range(16)),
            normal_file="local_normal.exr",
            mask_file="binary_mask.exr",
        )
        return PrivateSplitIndex(
            2,
            split,
            str(self.train_root if split == "train" else self.test_root),
            (record,),
            "private_exr_index_v1",
            ".data",
            "image",
            ".exr",
            "unsigned",
            (256, 256),
            "external",
            6,
            "seeded",
            20260710,
        )

    def datasets(self, _config, manifest, *, split):
        return _TinyDataset([_sample(record.name) for record in manifest.objects], split=split)

    def dependencies(self):
        return {
            "index_builder": self.manifests,
            "dataset_factory": self.datasets,
            "model_factory": _TinyReleasedModel,
            "schema_provider": lambda: (("scale", (), "torch.float32"),),
        }

    def test_both_split_preflights_finish_before_model_factory(self):
        events = []

        def manifest_builder(config, split):
            events.append(f"index:{split}")
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
                index_builder=manifest_builder,
                schema_provider=lambda: (("scale", (), "torch.float32"),),
            )
        self.assertEqual(events, ["index:train", "index:test", "model"])

    def test_indexing_precedes_device_and_first_lazy_source_read_is_reported(self):
        events: list[str] = []

        def index_builder(config, split):
            events.append(f"{split}_index")
            return self.manifests(config, split)

        class ReadMarkedDataset(_TinyDataset):
            def __getitem__(self, index):
                events.append("first_source_read")
                return super().__getitem__(index)

        def dataset_factory(_config, index, *, split):
            return ReadMarkedDataset(
                [_sample(record.name) for record in index.objects], split=split
            )

        stdout = StringIO()
        with contextlib.redirect_stdout(stdout):
            run_private_training(
                self.config(epochs=1),
                index_builder=index_builder,
                dataset_factory=dataset_factory,
                model_factory=_TinyReleasedModel,
                schema_provider=lambda: (("scale", (), "torch.float32"),),
                device_resolver=lambda _value: (
                    events.append("device") or torch.device("cpu")
                ),
            )

        self.assertLess(events.index("train_index"), events.index("device"))
        self.assertLess(events.index("test_index"), events.index("device"))
        self.assertLess(events.index("device"), events.index("first_source_read"))
        output = stdout.getvalue()
        self.assertIn("content validation: lazy", output)
        self.assertIn("First training batch ready after", output)
        self.assertIn("Total training time:", output)
        contract = json.loads((self.save_dir / "data_contract.json").read_text())
        self.assertEqual(contract["source_revision"], "lino-private-exr-training-v2")
        self.assertEqual(
            contract["gt_validity_policy"], "sdm_corrected_v2_unit_band"
        )

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

    def test_deferred_final_selection_never_reads_an_inference_manifest(self):
        config = replace(self.config(epochs=1), final_selection_manifest=None)

        with mock.patch(
            "src.training.trainer._read_final_selection",
            side_effect=AssertionError("training must not read a deferred manifest"),
        ):
            summary = run_private_training(config, **self.dependencies())

        contract = json.loads((summary.run_dir / "data_contract.json").read_text())
        self.assertIsNone(contract["final_selection_manifest_sha256"])

    def test_resume_appends_epoch_without_repeating_metrics(self):
        first = run_private_training(self.config(epochs=1), **self.dependencies())
        resumed = run_private_training(
            self.config(epochs=2, startup_mode="resume", resume=first.last_checkpoint),
            **self.dependencies(),
        )
        with (resumed.run_dir / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])

    def test_resume_metric_boundary_fails_before_live_model(self):
        first = run_private_training(self.config(epochs=1), **self.dependencies())
        (first.run_dir / "metrics.csv").unlink()
        events = []

        def model_factory():
            events.append("model")
            return _TinyReleasedModel()

        with self.assertRaisesRegex(ValueError, "metrics.csv"):
            run_private_training(
                self.config(epochs=2, startup_mode="resume", resume=first.last_checkpoint),
                model_factory=model_factory,
                index_builder=self.manifests,
                schema_provider=lambda: (("scale", (), "torch.float32"),),
                device_resolver=lambda value: events.append(value),
            )
        self.assertEqual(events, [])

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

    def test_validation_wraps_released_call_in_precision_context(self):
        model = _DtypeCheckedValidationModel()
        sample = _sample("object.data")
        loader = [{
            **{key: value.unsqueeze(0) for key, value in sample.items() if isinstance(value, torch.Tensor)},
            "metadata": [sample["metadata"]],
        }]

        @contextlib.contextmanager
        def marker(_device, _precision):
            model._autocast_marker = True
            try:
                yield
            finally:
                model._autocast_marker = False

        with mock.patch("src.training.trainer._autocast", marker):
            validate_epoch(model, loader, self.config(), source_predictor=lambda prediction, batch: prediction)
        self.assertTrue(model.autocast_seen)

    def test_validation_restores_numpy_state_and_groups_by_object_seed(self):
        model = _NumpyRandomValidationModel()
        sample = _sample("object.data")
        loader = [{
            **{key: value.unsqueeze(0) for key, value in sample.items() if isinstance(value, torch.Tensor)},
            "metadata": [sample["metadata"]],
        }]
        np.random.seed(91)
        before = np.random.get_state()
        first = validate_epoch(model, loader, self.config(), source_predictor=lambda prediction, batch: prediction)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        np.random.seed(77123)
        second = validate_epoch(model, loader, self.config(), source_predictor=lambda prediction, batch: prediction)
        self.assertEqual(first.mae, second.mae)

    def test_metric_csv_rows_require_exact_contiguous_finite_prefix(self):
        path = self.root / "metrics.csv"
        path.write_text(
            "epoch,train_loss,train_mae,validation_loss,validation_mae,learning_rate,global_step,epoch_seconds,peak_cuda_allocated_bytes,peak_cuda_reserved_bytes\n"
            "1,1,2,3,4,0.1,1,2,,\n"
            "3,1,2,3,4,0.1,2,2,,\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "contiguous|epoch"):
            _load_metric_rows(path)

    def test_failed_epoch_publication_preserves_previous_bundle(self):
        first = run_private_training(self.config(epochs=1), **self.dependencies())
        metrics_before = (first.run_dir / "metrics.csv").read_bytes()
        checkpoint_before = first.last_checkpoint.read_bytes()
        with mock.patch(
            "src.training.trainer.publish_tree_artifacts",
            side_effect=ValueError("directory was replaced during publication"),
        ):
            with self.assertRaisesRegex(ValueError, "publication|replaced"):
                run_private_training(
                    self.config(
                        epochs=2,
                        startup_mode="resume",
                        resume=first.last_checkpoint,
                    ),
                    **self.dependencies(),
                )
        self.assertEqual((first.run_dir / "metrics.csv").read_bytes(), metrics_before)
        self.assertEqual(first.last_checkpoint.read_bytes(), checkpoint_before)

    def test_bad_startup_checkpoint_fails_before_live_model_or_cuda(self):
        events = []
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            run_private_training(
                self.config(startup_mode="init_checkpoint"),
                schema_provider=lambda: (("scale", (), "torch.float32"),),
                model_factory=lambda: events.append("live_model"),
                device_resolver=lambda *_: events.append("cuda"),
                index_builder=self.manifests,
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
