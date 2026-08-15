"""CPU-only contract test for the private LINO training workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.inference import run_lino_inference
from src.training.config import PrivateTrainConfig
from src.training.private_manifest import PrivateObjectRecord, PrivateSplitManifest
from src.training.trainer import run_private_training
from tests.comparison_helpers import write_mask_exr, write_rgb_exr
from tests.test_private_trainer import _TinyDataset, _TinyReleasedModel, _sample


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrivateTrainingIntegrationTests(unittest.TestCase):
    """Exercise cold start, resume, export, and strict inference on CPU."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.save_dir = self.root / "runs"
        self.train_root = self.root / "train"
        self.test_root = self.root / "test"
        self.train_root.mkdir()
        self.test_root.mkdir()
        self.selection = self.root / "selected_lights.json"
        self.selection.write_text(
            json.dumps({"object.data": [f"image{i:03d}.exr" for i in range(16)]}) + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def config(
        self,
        *,
        epochs: int = 1,
        startup_mode: str = "cold_start",
        resume_checkpoint: Path | None = None,
    ) -> PrivateTrainConfig:
        return PrivateTrainConfig(
            train_dir=self.train_root,
            test_dir=self.test_root,
            save_dir=self.save_dir,
            startup_mode=startup_mode,
            init_checkpoint=None,
            resume_checkpoint=resume_checkpoint,
            final_selection_manifest=self.selection,
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            normal_filenames=("local_normal.exr",),
            external_mask_filename="binary_mask.exr",
            normal_encoding="unsigned",
            expected_source_geometry=(256, 256),
            mask_policy="external",
            mask_margin=0,
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

    def manifest_builder(self, _config: PrivateTrainConfig, split: str) -> PrivateSplitManifest:
        record = PrivateObjectRecord(
            name="object.data",
            relative_dir="object.data",
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
        root = self.train_root if split == "train" else self.test_root
        return PrivateSplitManifest(1, split, str(root), (record,))

    @staticmethod
    def dataset_factory(_config, manifest, *, split):
        return _TinyDataset([_sample(record.name) for record in manifest.objects], split=split)

    @staticmethod
    def source_predictor(prediction, _batch):
        return prediction

    def dependencies(self):
        return {
            "model_factory": _TinyReleasedModel,
            "schema_provider": lambda: (("scale", (), "torch.float32"),),
            "manifest_builder": self.manifest_builder,
            "dataset_factory": self.dataset_factory,
            "source_predictor": self.source_predictor,
        }

    def _write_inference_fixture(self, checkpoint: Path) -> SdmExrInferenceConfig:
        """Create one tiny 256x256 object for the strict injected inference route."""

        data_root = self.root / "inference"
        object_dir = data_root / "object.data"
        object_dir.mkdir(parents=True)
        for index in range(16):
            values = np.full((256, 256, 3), 0.25 + index, dtype=np.float32)
            write_rgb_exr(object_dir / f"image{index:03d}.exr", values)
        encoded_normal = np.full((256, 256, 3), 0.5, dtype=np.float32)
        encoded_normal[..., 2] = 1.0
        write_rgb_exr(object_dir / "local_normal.exr", encoded_normal)
        write_mask_exr(object_dir / "binary_mask.exr", np.ones((256, 256), dtype=np.float32))

        output_root = self.root / "isolated-inference-output"
        return SdmExrInferenceConfig(
            checkpoint=checkpoint,
            data_root=data_root,
            output_root=output_root,
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            max_image_num=16,
            light_selection="manifest",
            selection_manifest=self.selection,
            seed=20260710,
            mask_policy="external",
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
            expected_source_geometry=(256, 256),
            preprocessing_version="private_external_lino_native_v1",
            require_checkpoint_data_contract=True,
        )

    def test_cold_start_resume_export_and_strict_inference_contract(self):
        first = run_private_training(
            self.config(epochs=1),
            **self.dependencies(),
        )
        resumed = run_private_training(
            self.config(
                epochs=2,
                startup_mode="resume",
                resume_checkpoint=first.last_checkpoint,
            ),
            **self.dependencies(),
        )

        metrics_path = resumed.run_dir / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [1, 2])

        export = resumed.run_dir / "exports/lino_epoch_002.pth"
        sidecar = export.with_suffix(".json")
        self.assertTrue(export.is_file())
        self.assertTrue(sidecar.is_file())
        raw = torch.load(export, map_location="cpu", weights_only=False)
        self.assertEqual(set(raw), {"scale"})
        self.assertFalse(any(key.startswith(("net.", "model.")) for key in raw))
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(payload["preprocessing_version"], "private_external_lino_native_v1")
        self.assertEqual(payload["epoch"], 2)
        self.assertEqual(payload["run_kind"], "experiment")
        self.assertTrue(payload["comparable"])
        self.assertEqual(
            payload["checkpoint_sha256"],
            hashlib.sha256(export.read_bytes()).hexdigest(),
        )

        config = self._write_inference_fixture(export)
        result = run_lino_inference(
            config,
            model_loader=lambda _config, _device: _InferenceStub(),
            dataset_factory=None,
        )
        self.assertEqual(result["run_kind"], "experiment")
        self.assertTrue(result["comparable"])
        self.assertEqual(result["mae_objects"], 1)
        self.assertEqual(len(result["objects"]), 1)
        self.assertTrue(config.provenance_path.is_file())
        self.assertFalse((self.root / "final-eight-object-output").exists())


class _InferenceStub:
    def eval(self):
        return self

    def __call__(self, batch):
        metadata = batch["metadata"]
        height = int(metadata["source_geometry"]["height"])
        width = int(metadata["source_geometry"]["width"])
        output = np.zeros((height, width, 3), dtype=np.float32)
        output[..., 2] = 1.0
        return output


class PrivateTrainingDocumentationTests(unittest.TestCase):
    def test_operator_guide_contains_primary_commands(self):
        guide = (REPO_ROOT / "README_train_infer.md").read_text(encoding="utf-8")
        self.assertIn(
            "python train_private.py --config configs/lino_private_train_fixed.yaml",
            guide,
        )
        self.assertIn(
            "python eval.py --config configs/lino_private_infer_trained_fixed.yaml",
            guide,
        )


if __name__ == "__main__":
    unittest.main()
