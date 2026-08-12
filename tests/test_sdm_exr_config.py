"""Tests for the strict SDM-EXR inference configuration and CLI shim."""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml


class SdmExrConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def complete_values(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.root / "weights" / "lino.pth"),
            "data_root": str(self.root / "dataset"),
            "output_root": str(self.root / "outputs"),
            "object_suffix": ".data",
            "image_prefix": "image",
            "image_extension": ".exr",
            "max_image_num": 16,
            "light_selection": "seeded",
            "selection_manifest": "",
            "seed": 20260710,
            "mask_policy": "external",
            "external_mask_filename": "binary_mask.exr",
            "normal_filenames": ["local_normal.exr"],
            "normal_encoding": "signed",
            "expected_source_geometry": None,
            "mask_margin": 8,
            "max_image_resolution": 2048,
            "pixel_samples": 2048,
            "precision": "bf16",
            "device": "auto",
            "num_workers": 0,
            "save_exr": True,
            "save_png": True,
        }

    def load(self, **overrides):
        from src.comparison.config import load_sdm_exr_config

        values = self.complete_values()
        extra = overrides.pop("extra", None)
        values.update(overrides)
        if extra:
            values.update(extra)
        path = self.root / "config.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return load_sdm_exr_config(path)

    def test_complete_config_is_typed_and_has_derived_paths(self):
        config = self.load()

        self.assertIsInstance(config, object)
        self.assertIsInstance(config.checkpoint, Path)
        self.assertIsInstance(config.data_root, Path)
        self.assertIsInstance(config.output_root, Path)
        self.assertEqual(config.normal_filenames, ("local_normal.exr",))
        self.assertIsNone(config.selection_manifest)
        self.assertEqual(config.policy_root, config.output_root / "external")
        self.assertEqual(config.lino_output_dir, config.policy_root / "lino")
        self.assertEqual(config.sdm_view_dir, config.policy_root / "sdm_input")
        self.assertEqual(config.sdm_output_dir, config.policy_root / "sdm")
        self.assertEqual(config.input_manifest_path, config.policy_root / "input_manifest.json")
        self.assertEqual(
            config.effective_selection_manifest_path,
            config.policy_root / "selected_lights.json",
        )
        self.assertEqual(config.provenance_path, config.lino_output_dir / "run.json")

        with self.assertRaises(AttributeError):
            config.mask_policy = "full"

    def test_checked_in_config_has_shared_runtime_defaults_and_is_single_preset(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "configs"
        config_paths = sorted(config_dir.glob("*sdm*exr*infer*.yaml"))
        self.assertEqual([path.name for path in config_paths], ["sdm_exr_infer.yaml"])
        self.assertFalse(any("masked" in path.name or "full" in path.name for path in config_paths))

        from src.comparison.config import load_sdm_exr_config

        config = load_sdm_exr_config(config_paths[0])
        self.assertEqual(config.checkpoint, Path("/path/to/lino.pth"))
        self.assertEqual(config.data_root, Path("/mnt/18TData/minhnv/test"))
        self.assertEqual(config.output_root, Path("./output/sdm_lino_comparison"))
        self.assertEqual(config.object_suffix, ".data")
        self.assertEqual(config.image_prefix, "image")
        self.assertEqual(config.image_extension, ".exr")
        self.assertEqual(config.max_image_num, 16)
        self.assertEqual(config.light_selection, "seeded")
        self.assertIsNone(config.selection_manifest)
        self.assertEqual(config.seed, 20260710)
        self.assertEqual(config.mask_policy, "external")
        self.assertEqual(config.external_mask_filename, "binary_mask.exr")
        self.assertEqual(config.normal_filenames, ("local_normal.exr",))
        self.assertEqual(config.normal_encoding, "signed")
        self.assertIsNone(config.expected_source_geometry)
        self.assertEqual(config.mask_margin, 8)
        self.assertEqual(config.max_image_resolution, 2048)
        self.assertEqual(config.pixel_samples, 2048)
        self.assertEqual(config.precision, "bf16")
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.num_workers, 0)
        self.assertTrue(config.save_exr)
        self.assertTrue(config.save_png)

    def test_external_and_full_are_the_only_mask_policies(self):
        external = self.load(mask_policy="external")
        full = self.load(mask_policy="full")
        self.assertEqual(external.policy_root.name, "external")
        self.assertEqual(full.policy_root.name, "full")
        with self.assertRaisesRegex(ValueError, "mask_policy"):
            self.load(mask_policy="oracle_gt")

    def test_manifest_mode_requires_manifest_path(self):
        with self.assertRaisesRegex(ValueError, "selection_manifest"):
            self.load(light_selection="manifest", selection_manifest="")

    def test_resolution_must_be_a_positive_multiple_of_512(self):
        with self.assertRaisesRegex(ValueError, "max_image_resolution"):
            self.load(max_image_resolution=6000)

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown_key"):
            self.load(extra={"unknown_key": 1})

    def test_non_string_yaml_keys_are_rejected_with_value_error(self):
        # Include a normal unknown key as well so the pre-fix implementation's
        # direct sort attempts to compare str and int and raises TypeError.
        with self.assertRaisesRegex(ValueError, "1"):
            self.load(extra={"unknown_key": 1, 1: "not-a-field"})

    def test_manifest_path_is_typed_and_used_as_effective_path(self):
        selection_manifest = self.root / "selected.json"
        config = self.load(light_selection="manifest", selection_manifest=str(selection_manifest))
        self.assertEqual(config.selection_manifest, selection_manifest)
        self.assertEqual(config.effective_selection_manifest_path, selection_manifest)

    def test_validation_rejects_unsupported_values_and_nonpositive_counts(self):
        cases = (
            ("normal_encoding", "octahedral"),
            ("light_selection", "random"),
            ("precision", "fp8"),
            ("device", "tpu"),
            ("max_image_num", 0),
            ("pixel_samples", 0),
            ("mask_margin", -1),
            ("num_workers", -1),
        )
        for key, value in cases:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(**{key: value})

    def test_signed_and_unsigned_encodings_are_explicitly_supported(self):
        self.assertEqual(self.load(normal_encoding="signed").normal_encoding, "signed")
        self.assertEqual(self.load(normal_encoding="unsigned").normal_encoding, "unsigned")

    def test_expected_source_geometry_is_optional_and_strictly_positive_hw(self):
        self.assertIsNone(self.load(expected_source_geometry=None).expected_source_geometry)
        self.assertEqual(
            self.load(expected_source_geometry=[256, 256]).expected_source_geometry,
            (256, 256),
        )
        for invalid in ([256], [256, 256, 3], [0, 256], [True, 256], "256x256"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "expected_source_geometry"
            ):
                self.load(expected_source_geometry=invalid)

    def test_older_yaml_without_expected_source_geometry_defaults_to_none(self):
        from src.comparison.config import load_sdm_exr_config

        values = self.complete_values()
        values.pop("expected_source_geometry")
        path = self.root / "older-config.yaml"
        path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        self.assertIsNone(load_sdm_exr_config(path).expected_source_geometry)

    def test_nonempty_string_fields_and_normal_filenames_are_required(self):
        for key in ("object_suffix", "image_prefix", "image_extension", "external_mask_filename"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.load(**{key: ""})
        with self.assertRaisesRegex(ValueError, "normal_filenames"):
            self.load(normal_filenames=[])


class EvalDispatchTests(unittest.TestCase):
    def test_main_dispatches_to_legacy_or_configured_runner(self):
        eval_module = importlib.import_module("eval")
        fake_inference = types.ModuleType("src.comparison.inference")
        configured = mock.Mock(name="run_lino_inference")
        fake_inference.run_lino_inference = configured

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            helper = SdmExrConfigTests()
            helper.root = Path(temp_dir)
            values = helper.complete_values()
            path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

            with mock.patch.object(eval_module, "run_legacy") as legacy, mock.patch.dict(
                sys.modules, {"src.comparison.inference": fake_inference}
            ):
                eval_module.main(["--task_name", "DiLiGenT"])
                legacy.assert_called_once()
                configured.assert_not_called()

                eval_module.main(["--config", str(path)])
                configured.assert_called_once()
                self.assertEqual(configured.call_args.args[0].mask_policy, "external")


if __name__ == "__main__":
    unittest.main()
