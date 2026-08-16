"""Tests for the strict private LINO training configuration contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.training.config import (
    PrivateTrainConfig,
    load_private_train_config,
    resolved_config_dict,
)


class PrivateTrainingConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def valid_mapping(self) -> dict[str, object]:
        return {
            "train_dir": "/mnt/18TData/minhnv/train",
            "test_dir": "/mnt/18TData/minhnv/test",
            "save_dir": "/mnt/18TData/minhnv/runs/lino_private",
            "startup_mode": "cold_start",
            "init_checkpoint": None,
            "resume_checkpoint": None,
            "final_selection_manifest": "/mnt/18TData/minhnv/runs/lino_private/selection.json",
            "object_suffix": ".data",
            "image_prefix": "image",
            "image_extension": ".exr",
            "normal_filenames": ["local_normal.exr"],
            "external_mask_filename": "binary_mask.exr",
            "normal_encoding": "unsigned",
            "expected_source_geometry": [256, 256],
            "mask_policy": "external",
            "mask_margin": 8,
            "max_image_num": 6,
            "light_selection": "seeded",
            "seed": 20260710,
            "preprocessing_version": "private_external_lino_native_v1",
            "max_image_resolution": 512,
            "canonical_resolution": 256,
            "pixel_samples": 2048,
            "train_pixel_budget": 131072,
            "precision": "bf16",
            "device": "cuda",
            "deterministic": True,
            "epochs": 50,
            "train_batch_size": 2,
            "train_workers": 4,
            "test_workers": 2,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "adamw_betas": [0.9, 0.98],
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.5,
            "save_every_epochs": 5,
            "keep_milestone_epochs": [10, 25, 50],
            "source_validation": {
                "mode": "lazy",
                "structural_index_version": "private_exr_index_v1",
                "persistent_content_ledger": False,
                "progress_every_objects": 100,
            },
        }

    def _write_yaml(self, values: dict[object, object]) -> Path:
        path = self.root / "private_train.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return path

    def test_cold_start_contract_is_exact(self):
        config = load_private_train_config(self._write_yaml(self.valid_mapping()))
        expected = PrivateTrainConfig(
            train_dir=Path("/mnt/18TData/minhnv/train"),
            test_dir=Path("/mnt/18TData/minhnv/test"),
            save_dir=Path("/mnt/18TData/minhnv/runs/lino_private"),
            startup_mode="cold_start",
            init_checkpoint=None,
            resume_checkpoint=None,
            final_selection_manifest=Path(
                "/mnt/18TData/minhnv/runs/lino_private/selection.json"
            ),
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
            pixel_samples=2048,
            train_pixel_budget=131072,
            precision="bf16",
            device="cuda",
            deterministic=True,
            epochs=50,
            train_batch_size=2,
            train_workers=4,
            test_workers=2,
            learning_rate=0.0001,
            weight_decay=0.01,
            adamw_betas=(0.9, 0.98),
            scheduler_step_size=10,
            scheduler_gamma=0.5,
            save_every_epochs=5,
            keep_milestone_epochs=(10, 25, 50),
        )
        self.assertEqual(config, expected)
        self.assertEqual(
            resolved_config_dict(config),
            {
                "train_dir": "/mnt/18TData/minhnv/train",
                "test_dir": "/mnt/18TData/minhnv/test",
                "save_dir": "/mnt/18TData/minhnv/runs/lino_private",
                "startup_mode": "cold_start",
                "init_checkpoint": None,
                "resume_checkpoint": None,
                "final_selection_manifest": (
                    "/mnt/18TData/minhnv/runs/lino_private/selection.json"
                ),
                "object_suffix": ".data",
                "image_prefix": "image",
                "image_extension": ".exr",
                "normal_filenames": ["local_normal.exr"],
                "external_mask_filename": "binary_mask.exr",
                "normal_encoding": "unsigned",
                "expected_source_geometry": [256, 256],
                "mask_policy": "external",
                "mask_margin": 8,
                "max_image_num": 6,
                "light_selection": "seeded",
                "seed": 20260710,
                "preprocessing_version": "private_external_lino_native_v1",
                "max_image_resolution": 512,
                "canonical_resolution": 256,
                "pixel_samples": 2048,
                "train_pixel_budget": 131072,
                "precision": "bf16",
                "device": "cuda",
                "deterministic": True,
                "epochs": 50,
                "train_batch_size": 2,
                "train_workers": 4,
                "test_workers": 2,
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "adamw_betas": [0.9, 0.98],
                "scheduler_step_size": 10,
                "scheduler_gamma": 0.5,
                "save_every_epochs": 5,
                "keep_milestone_epochs": [10, 25, 50],
                "source_validation": {
                    "mode": "lazy",
                    "structural_index_version": "private_exr_index_v1",
                    "persistent_content_ledger": False,
                    "progress_every_objects": 100,
                },
            },
        )

    def test_source_validation_contract_is_lazy_only_and_strict(self):
        for field_name, value, message in (
            ("mode", "eager", "lazy source validation only"),
            ("structural_index_version", "other", "structural index version"),
            ("persistent_content_ledger", True, "content ledger"),
            ("progress_every_objects", -1, "non-negative integer"),
            ("progress_every_objects", True, "non-negative integer"),
        ):
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError, message
            ):
                raw = self.valid_mapping()
                raw["source_validation"][field_name] = value
                load_private_train_config(self._write_yaml(raw))

    def test_zero_progress_interval_disables_periodic_index_messages(self):
        raw = self.valid_mapping()
        raw["source_validation"]["progress_every_objects"] = 0

        config = load_private_train_config(self._write_yaml(raw))

        self.assertEqual(config.source_validation.progress_every_objects, 0)

    def test_source_validation_rejects_missing_and_unknown_nested_keys(self):
        raw = self.valid_mapping()
        del raw["source_validation"]["mode"]
        with self.assertRaisesRegex(ValueError, "source_validation.*mode"):
            load_private_train_config(self._write_yaml(raw))

        raw = self.valid_mapping()
        raw["source_validation"]["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "source_validation.*unexpected"):
            load_private_train_config(self._write_yaml(raw))

    def test_loader_converts_paths_and_sequences(self):
        config = load_private_train_config(self._write_yaml(self.valid_mapping()))
        self.assertIsInstance(config.test_dir, Path)
        self.assertIsInstance(config.save_dir, Path)
        self.assertIsInstance(config.final_selection_manifest, Path)
        self.assertEqual(config.normal_filenames, ("local_normal.exr",))
        self.assertEqual(config.keep_milestone_epochs, (10, 25, 50))
        self.assertEqual(
            resolved_config_dict(config)["train_dir"], "/mnt/18TData/minhnv/train"
        )
        self.assertEqual(resolved_config_dict(config)["adamw_betas"], [0.9, 0.98])
        self.assertEqual(list(resolved_config_dict(config)), list(config.__dataclass_fields__))

    def test_final_selection_manifest_may_be_deferred(self):
        raw = self.valid_mapping()
        raw["final_selection_manifest"] = None

        config = load_private_train_config(self._write_yaml(raw))

        self.assertIsNone(config.final_selection_manifest)

    def test_config_is_immutable(self):
        config = load_private_train_config(self._write_yaml(self.valid_mapping()))
        with self.assertRaises(AttributeError):
            config.device = "cpu"

    def test_unknown_keys_are_rejected(self):
        raw = self.valid_mapping()
        raw["unknown_key"] = 1
        with self.assertRaisesRegex(ValueError, "unknown_key"):
            load_private_train_config(self._write_yaml(raw))

    def test_non_string_yaml_keys_are_rejected_with_value_error(self):
        raw = self.valid_mapping()
        raw["unknown_key"] = 1
        raw[1] = "not-a-field"
        with self.assertRaisesRegex(ValueError, "1"):
            load_private_train_config(self._write_yaml(raw))

    def test_missing_keys_are_rejected(self):
        raw = self.valid_mapping()
        del raw["train_pixel_budget"]
        with self.assertRaisesRegex(ValueError, "train_pixel_budget"):
            load_private_train_config(self._write_yaml(raw))

    def test_startup_modes_are_mutually_exclusive(self):
        raw = self.valid_mapping()
        raw["startup_mode"] = "cold_start"
        raw["init_checkpoint"] = "checkpoints/lino.pth"
        with self.assertRaisesRegex(ValueError, "cold_start.*init_checkpoint"):
            load_private_train_config(self._write_yaml(raw))

    def test_init_checkpoint_requires_pth_and_excludes_resume(self):
        raw = self.valid_mapping()
        raw["startup_mode"] = "init_checkpoint"
        raw["init_checkpoint"] = "checkpoints/lino.ckpt"
        with self.assertRaisesRegex(ValueError, "init_checkpoint.*pth"):
            load_private_train_config(self._write_yaml(raw))

        raw["init_checkpoint"] = "checkpoints/lino.pth"
        raw["resume_checkpoint"] = "runs/lino.ckpt"
        with self.assertRaisesRegex(ValueError, "init_checkpoint.*resume_checkpoint"):
            load_private_train_config(self._write_yaml(raw))

    def test_resume_requires_ckpt_suffix(self):
        raw = self.valid_mapping()
        raw["startup_mode"] = "resume"
        raw["resume_checkpoint"] = "runs/lino_epoch_040.pth"
        with self.assertRaisesRegex(ValueError, "resume_checkpoint.*ckpt"):
            load_private_train_config(self._write_yaml(raw))

    def test_resume_excludes_init_checkpoint(self):
        raw = self.valid_mapping()
        raw["startup_mode"] = "resume"
        raw["init_checkpoint"] = "checkpoints/lino.pth"
        raw["resume_checkpoint"] = "runs/lino_epoch_040.ckpt"
        with self.assertRaisesRegex(ValueError, "resume.*init_checkpoint"):
            load_private_train_config(self._write_yaml(raw))

    def test_train_and_test_roots_cannot_alias(self):
        raw = self.valid_mapping()
        raw["test_dir"] = raw["train_dir"]
        with self.assertRaisesRegex(ValueError, "train_dir.*test_dir"):
            load_private_train_config(self._write_yaml(raw))

    def test_private_contract_rejects_wrong_fixed_values(self):
        for key, value in (
            ("object_suffix", ".png"),
            ("image_prefix", "frame"),
            ("image_extension", ".png"),
            ("normal_filenames", ["normal.exr"]),
            ("external_mask_filename", "mask.exr"),
            ("normal_encoding", "signed"),
            ("expected_source_geometry", [512, 512]),
            ("mask_policy", "full"),
            ("max_image_num", 16),
            ("light_selection", "manifest"),
            ("preprocessing_version", "other"),
            ("max_image_resolution", 256),
            ("canonical_resolution", 512),
        ):
            expected_error = (
                "private source geometry|expected_source_geometry"
                if key == "expected_source_geometry"
                else "private LINO geometry|max_image_resolution|canonical_resolution"
                if key in {"max_image_resolution", "canonical_resolution"}
                else key
            )
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, expected_error):
                raw = self.valid_mapping()
                raw[key] = value
                load_private_train_config(self._write_yaml(raw))

    def test_positive_and_nonnegative_numeric_checks(self):
        for key, value in (
            ("mask_margin", -1),
            ("seed", -1),
            ("pixel_samples", 0),
            ("train_pixel_budget", 0),
            ("epochs", 0),
            ("train_batch_size", 0),
            ("train_workers", -1),
            ("test_workers", -1),
            ("learning_rate", 0),
            ("weight_decay", -1),
            ("scheduler_step_size", 0),
            ("scheduler_gamma", 0),
            ("save_every_epochs", 0),
            ("keep_milestone_epochs", [10, 10]),
            ("keep_milestone_epochs", [25, 10]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                raw = self.valid_mapping()
                raw[key] = value
                load_private_train_config(self._write_yaml(raw))

    def test_bool_is_not_accepted_as_an_integer(self):
        raw = self.valid_mapping()
        raw["epochs"] = True
        with self.assertRaisesRegex(ValueError, "epochs"):
            load_private_train_config(self._write_yaml(raw))

    def test_filenames_must_be_safe_basenames(self):
        for key, value in (
            ("normal_filenames", ["../local_normal.exr"]),
            ("external_mask_filename", "subdir/mask.exr"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                raw = self.valid_mapping()
                raw[key] = value
                load_private_train_config(self._write_yaml(raw))

    def test_precision_device_contract(self):
        raw = self.valid_mapping()
        raw["precision"] = "fp32"
        raw["device"] = "cpu"
        self.assertEqual(load_private_train_config(self._write_yaml(raw)).precision, "fp32")

        raw["precision"] = "bf16"
        with self.assertRaisesRegex(ValueError, "bf16.*CUDA"):
            load_private_train_config(self._write_yaml(raw))

    def test_nonfinite_optimizer_values_are_rejected(self):
        for key, value in (
            ("learning_rate", float("nan")),
            ("weight_decay", float("inf")),
            ("scheduler_gamma", float("nan")),
            ("adamw_betas", [0.9, float("inf")]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                raw = self.valid_mapping()
                raw[key] = value
                load_private_train_config(self._write_yaml(raw))

    def test_checked_in_lazy_presets_are_isolated_and_paired(self):
        repo_root = Path(__file__).resolve().parents[1]
        canonical_selection = (
            "output/sdm_lino_comparison/external/selected_lights.json"
        )
        config = load_private_train_config(
            repo_root / "configs/lino_private_train_fixed.yaml"
        )
        self.assertEqual(
            str(config.save_dir), "runs/lino_private_fixed_lazy_sdmvalid_bf16"
        )
        self.assertEqual(config.source_validation.mode, "lazy")
        self.assertIsNone(config.final_selection_manifest)
        inference_yaml = (
            repo_root / "configs/lino_private_infer_trained_fixed.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "runs/lino_private_fixed_lazy_sdmvalid_bf16/exports/"
            "lino_epoch_100.pth",
            inference_yaml,
        )
        self.assertIn(
            'output_root: "./output/lino_private_trained_lazy_sdmvalid"',
            inference_yaml,
        )
        self.assertIn(
            f'selection_manifest: "./{canonical_selection}"', inference_yaml
        )


if __name__ == "__main__":
    unittest.main()
