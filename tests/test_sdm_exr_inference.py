"""CPU-only contract tests for released-checkpoint LINO inference."""

from __future__ import annotations

import json
import hashlib
import io
import os
import sys
import subprocess
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import read_rgb_exr, sha256_file
import src.comparison.inference as inference
from src.comparison.inference import load_local_lino_checkpoint, run_lino_inference
from tests.comparison_helpers import make_object


class _StubModel(torch.nn.Module):
    """Return a known source-resolution normal while recording model batches."""

    def __init__(self, calls: list[dict]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, batch):
        self.calls.append(batch)
        self.assert_batch_has_no_ground_truth(batch)
        metadata = batch["metadata"]
        height = int(metadata["source_geometry"]["height"])
        width = int(metadata["source_geometry"]["width"])
        output = np.zeros((height, width, 3), dtype=np.float32)
        output[..., 0] = 3.0
        output[..., 1] = 4.0
        return output

    @staticmethod
    def assert_batch_has_no_ground_truth(batch):
        if not torch.is_tensor(batch["imgs"]) or batch["imgs"].dtype != torch.float32:
            raise AssertionError("imgs must remain float32")
        if not torch.is_tensor(batch["mask"]) or batch["mask"].dtype != torch.float32:
            raise AssertionError("mask must remain float32")
        if batch["roi"].device.type != "cpu":
            raise AssertionError("roi must remain on CPU")
        if batch["mask_original"].device.type != "cpu":
            raise AssertionError("mask_original must remain on CPU")
        if "nml" in batch or "normal" in batch or "ground_truth" in batch:
            raise AssertionError("ground truth leaked into model batch")


class SdmExrInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.output_root = self.root / "output"
        self.checkpoint = self.root / "weights" / "lino.pth"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(b"stub-checkpoint")

    def config(self, **overrides) -> SdmExrInferenceConfig:
        values = dict(
            checkpoint=self.checkpoint,
            data_root=self.data_root,
            output_root=self.output_root,
            object_suffix=".data",
            image_prefix="image",
            image_extension=".exr",
            max_image_num=2,
            light_selection="seeded",
            selection_manifest=None,
            seed=20260710,
            mask_policy="external",
            external_mask_filename="binary_mask.exr",
            normal_filenames=("local_normal.exr",),
            normal_encoding="signed",
            mask_margin=0,
            max_image_resolution=512,
            pixel_samples=1,
            precision="fp32",
            device="cpu",
            num_workers=0,
            save_exr=True,
            save_png=False,
        )
        values.update(overrides)
        return SdmExrInferenceConfig(**values)

    def make_dataset(self, *names: str) -> None:
        for name in names:
            make_object(self.data_root, name)

    def run_with_stub(self, config: SdmExrInferenceConfig):
        calls: list[dict] = []

        def model_loader(_config, _device):
            return _StubModel(calls)

        result = run_lino_inference(config, model_loader=model_loader)
        return result, calls

    def test_runtime_paths_are_checked_before_model_loader_is_called(self):
        self.make_dataset("alpha.data")
        config = self.config(checkpoint=self.root / "missing.pth")
        called = False

        def model_loader(_config, _device):
            nonlocal called
            called = True
            raise AssertionError("model loader must not run")

        with self.assertRaises(FileNotFoundError):
            run_lino_inference(config, model_loader=model_loader)
        self.assertFalse(called)

    def test_inference_processes_objects_in_manifest_order(self):
        self.make_dataset("zeta.data", "alpha.data")
        result, calls = self.run_with_stub(self.config())
        self.assertEqual(
            [item["object_name"] for item in result["objects"]],
            ["alpha.data", "zeta.data"],
        )
        self.assertEqual(
            [item["metadata"]["object_name"] for item in calls],
            ["alpha.data", "zeta.data"],
        )

    def test_prediction_exr_is_signed_float32_source_geometry(self):
        self.make_dataset("alpha.data")
        config = self.config(save_png=True)
        result, _ = self.run_with_stub(config)
        prediction = config.lino_output_dir / "alpha.data" / "normal_pred.exr"
        preview = config.lino_output_dir / "alpha.data" / "normal_pred.png"
        self.assertTrue(prediction.is_file())
        self.assertTrue(preview.is_file())
        decoded = read_rgb_exr(prediction)
        self.assertEqual(decoded.shape, (2, 3, 3))
        self.assertEqual(decoded.dtype, np.float32)
        expected = np.zeros((2, 3, 3), dtype=np.float32)
        expected[..., 0] = 0.6
        expected[..., 1] = 0.8
        np.testing.assert_allclose(decoded, expected, atol=1e-4)
        self.assertEqual(result["objects"][0]["output_sha256"], sha256_file(prediction))

    def test_policy_output_directories_do_not_overlap(self):
        self.make_dataset("alpha.data")
        external = self.config(mask_policy="external")
        full = self.config(mask_policy="full")
        self.run_with_stub(external)
        self.run_with_stub(full)
        self.assertNotEqual(external.lino_output_dir, full.lino_output_dir)
        self.assertTrue((external.lino_output_dir / "alpha.data" / "normal_pred.exr").is_file())
        self.assertTrue((full.lino_output_dir / "alpha.data" / "normal_pred.exr").is_file())

    def test_inference_refuses_preexisting_symlink_lino_directory(self):
        self.make_dataset("alpha.data")
        config = self.config()
        config.policy_root.mkdir(parents=True, exist_ok=True)
        outside = self.root / "outside"
        outside.mkdir()
        os.symlink(outside, config.lino_output_dir, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "symlink|directory|output"):
            self.run_with_stub(config)
        self.assertFalse((outside / "alpha.data").exists())

    def test_inference_refuses_preexisting_symlink_policy_directory(self):
        self.make_dataset("alpha.data")
        config = self.config()
        config.output_root.mkdir(parents=True, exist_ok=True)
        outside = self.root / "outside-policy"
        outside.mkdir()
        os.symlink(outside, config.policy_root, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "symlink|directory|output"):
            self.run_with_stub(config)
        self.assertFalse((outside / "input_manifest.json").exists())

    def test_inference_refuses_preexisting_symlink_object_directory(self):
        self.make_dataset("alpha.data")
        config = self.config()
        config.lino_output_dir.mkdir(parents=True, exist_ok=True)
        outside = self.root / "outside-object"
        outside.mkdir()
        os.symlink(outside, config.lino_output_dir / "alpha.data", target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "symlink|directory|output"):
            self.run_with_stub(config)
        self.assertFalse((outside / "normal_pred.exr").exists())

    def test_inference_refuses_preexisting_symlink_prediction_file(self):
        self.make_dataset("alpha.data")
        config = self.config()
        object_dir = config.lino_output_dir / "alpha.data"
        object_dir.mkdir(parents=True, exist_ok=True)
        outside = self.root / "outside-prediction.exr"
        outside.write_bytes(b"user-owned")
        os.symlink(outside, object_dir / "normal_pred.exr")

        with self.assertRaisesRegex(ValueError, "symlink|regular|prediction"):
            self.run_with_stub(config)
        self.assertEqual(outside.read_bytes(), b"user-owned")

    def test_inference_rejects_policy_root_replacement_between_objects(self):
        self.make_dataset("alpha.data", "zeta.data")
        config = self.config()
        moved = self.root / "detached-policy"
        swapped = False

        class SwappingModel(_StubModel):
            def forward(inner_self, batch):
                nonlocal swapped
                result = super().forward(batch)
                if batch["metadata"]["object_name"] == "zeta.data" and not swapped:
                    config.policy_root.rename(moved)
                    config.policy_root.mkdir(parents=True)
                    swapped = True
                return result

        with self.assertRaisesRegex(ValueError, "replaced|output directory"):
            run_lino_inference(
                config,
                model_loader=lambda _config, _device: SwappingModel([]),
            )
        self.assertTrue(swapped)
        self.assertFalse(config.provenance_path.exists())

    def test_inference_rejects_a_missing_object_before_run_publication(self):
        self.make_dataset("alpha.data", "zeta.data")
        config = self.config()
        moved_object = self.root / "removed-alpha-output"
        moved = False

        class RemovingModel(_StubModel):
            def forward(inner_self, batch):
                nonlocal moved
                result = super().forward(batch)
                if batch["metadata"]["object_name"] == "zeta.data" and not moved:
                    (config.lino_output_dir / "alpha.data").rename(moved_object)
                    moved = True
                return result

        with self.assertRaisesRegex(ValueError, "missing|object|artifact"):
            run_lino_inference(
                config,
                model_loader=lambda _config, _device: RemovingModel([]),
            )
        self.assertTrue(moved)
        self.assertFalse(config.provenance_path.exists())

    def test_post_publication_validation_failure_removes_run_provenance(self):
        self.make_dataset("alpha.data", "zeta.data")
        config = self.config()
        moved_object = self.root / "removed-after-run-publication"
        removed = False
        real_validate = inference._validate_lino_output_tree

        def remove_before_post_publication_validation(*args, **kwargs):
            nonlocal removed
            if kwargs["require_run"] and not removed:
                (config.lino_output_dir / "alpha.data").rename(moved_object)
                removed = True
            return real_validate(*args, **kwargs)

        with mock.patch.object(
            inference,
            "_validate_lino_output_tree",
            side_effect=remove_before_post_publication_validation,
        ):
            with self.assertRaisesRegex(ValueError, "missing|object|artifact"):
                self.run_with_stub(config)

        self.assertTrue(removed)
        self.assertFalse(config.provenance_path.exists())

    def test_post_publication_run_replacement_is_rejected_without_deleting_competitor(self):
        self.make_dataset("alpha.data")
        config = self.config()
        competing_bytes = b'{"owner":"concurrent"}\n'
        original_run = self.root / "published-run.json"
        replaced = False
        real_validate = inference._validate_lino_output_tree

        def replace_before_post_publication_validation(*args, **kwargs):
            nonlocal replaced
            if kwargs["require_run"] and not replaced:
                config.provenance_path.rename(original_run)
                config.provenance_path.write_bytes(competing_bytes)
                replaced = True
            return real_validate(*args, **kwargs)

        with mock.patch.object(
            inference,
            "_validate_lino_output_tree",
            side_effect=replace_before_post_publication_validation,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "run provenance.*replaced|replaced.*run provenance",
            ):
                self.run_with_stub(config)

        self.assertTrue(replaced)
        self.assertEqual(config.provenance_path.read_bytes(), competing_bytes)
        self.assertTrue(original_run.is_file())

    def test_prepublish_validation_rejects_a_concurrent_run_provenance(self):
        self.make_dataset("alpha.data")
        config = self.config()
        competing_bytes = b'{"owner":"concurrent"}\n'
        injected = False
        real_validate = inference._validate_lino_output_tree

        def inject_before_exact_prepublish_validation(*args, **kwargs):
            nonlocal injected
            if kwargs["require_all_objects"] and not kwargs["require_run"]:
                config.provenance_path.write_bytes(competing_bytes)
                injected = True
            return real_validate(*args, **kwargs)

        with mock.patch.object(
            inference,
            "_validate_lino_output_tree",
            side_effect=inject_before_exact_prepublish_validation,
        ):
            with self.assertRaisesRegex(ValueError, "extra.*run.json"):
                self.run_with_stub(config)

        self.assertTrue(injected)
        self.assertEqual(config.provenance_path.read_bytes(), competing_bytes)

    def test_output_tree_open_closes_policy_descriptor_when_lino_open_fails(self):
        self.make_dataset("alpha.data")
        config = self.config()
        captured_fd: int | None = None
        real_open_policy = inference._open_or_create_directory

        def capture_policy(*args, **kwargs):
            nonlocal captured_fd
            result = real_open_policy(*args, **kwargs)
            captured_fd = result[0]
            return result

        with mock.patch.object(
            inference,
            "_open_or_create_directory",
            side_effect=capture_policy,
        ):
            with mock.patch.object(
                inference,
                "_open_or_create_child_directory",
                side_effect=ValueError("forced LINO child failure"),
            ):
                with self.assertRaisesRegex(ValueError, "forced LINO child failure"):
                    inference._open_lino_output_tree(config)

        self.assertIsNotNone(captured_fd)
        with self.assertRaises(OSError):
            os.fstat(captured_fd)

    def test_child_directory_open_closes_descriptor_when_fstat_fails(self):
        parent = self.root / "parent"
        child = parent / "lino"
        child.mkdir(parents=True)
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        real_open = os.open
        real_fstat = os.fstat
        child_fd: int | None = None

        def capture_child(path, *args, **kwargs):
            nonlocal child_fd
            descriptor = real_open(path, *args, **kwargs)
            if path == child.name:
                child_fd = descriptor
            return descriptor

        def fail_child_fstat(descriptor):
            if descriptor == child_fd:
                raise OSError("forced child fstat failure")
            return real_fstat(descriptor)

        try:
            with mock.patch.object(os, "open", side_effect=capture_child):
                with mock.patch.object(os, "fstat", side_effect=fail_child_fstat):
                    with self.assertRaisesRegex(ValueError, "securely open|fstat"):
                        inference._open_or_create_child_directory(
                            parent_fd,
                            parent,
                            child.name,
                            label="LINO output directory",
                        )
            self.assertIsNotNone(child_fd)
            with self.assertRaises(OSError):
                os.fstat(child_fd)
        finally:
            if child_fd is not None:
                try:
                    os.close(child_fd)
                except OSError:
                    pass
            os.close(parent_fd)

    def test_failed_rerun_removes_stale_run_provenance_before_mutation(self):
        self.make_dataset("alpha.data")
        config = self.config()
        self.run_with_stub(config)
        self.assertTrue(config.provenance_path.is_file())
        rerun = replace(config, seed=config.seed + 1)

        def fail_model_loader(_config, _device):
            raise RuntimeError("forced rerun failure")

        with self.assertRaisesRegex(RuntimeError, "forced rerun failure"):
            run_lino_inference(rerun, model_loader=fail_model_loader)
        self.assertFalse(config.provenance_path.exists())

    def test_output_preflight_rejects_extras_before_manifest_mutation(self):
        self.make_dataset("alpha.data")
        config = self.config()
        self.run_with_stub(config)
        before = config.input_manifest_path.read_bytes()
        (config.lino_output_dir / "unexpected.txt").write_bytes(b"unowned")
        rerun = replace(config, seed=config.seed + 1)

        with self.assertRaisesRegex(ValueError, "extra LINO"):
            self.run_with_stub(rerun)
        self.assertEqual(config.input_manifest_path.read_bytes(), before)

    def test_manifest_source_hash_and_selection_use_one_immutable_snapshot(self):
        self.make_dataset("alpha.data")
        selection = self.root / "selection.json"
        original = b'{"alpha.data":["image_003.exr","image_001.exr"]}\n'
        replacement_bytes = b'{"alpha.data":["image_001.exr","image_003.exr"]}\n'
        selection.write_bytes(original)
        config = self.config(light_selection="manifest", selection_manifest=selection)
        real_persist = inference._persist_manifests
        mutated = False

        def mutate_before_persist(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                selection.write_bytes(replacement_bytes)
                mutated = True
            return real_persist(*args, **kwargs)

        with mock.patch.object(
            inference,
            "_persist_manifests",
            side_effect=mutate_before_persist,
        ):
            result, calls = self.run_with_stub(config)

        self.assertTrue(mutated)
        self.assertEqual(
            calls[0]["metadata"]["selected_images"],
            ["image_003.exr", "image_001.exr"],
        )
        self.assertEqual(result["selection_manifest_sha256"], hashlib.sha256(original).hexdigest())

    def test_manifest_alias_change_after_snapshot_fails_closed(self):
        self.make_dataset("alpha.data")
        selection = self.output_root / "external" / "selected_lights.json"
        selection.parent.mkdir(parents=True)
        original = b'{"alpha.data":["image_003.exr","image_001.exr"]}\n'
        replacement_bytes = b'{"alpha.data":["image_001.exr","image_003.exr"]}\n'
        selection.write_bytes(original)
        config = self.config(light_selection="manifest", selection_manifest=selection)
        real_persist = inference._persist_manifests
        mutated = False

        def mutate_alias_before_persist(*args, **kwargs):
            nonlocal mutated
            if not mutated:
                selection.write_bytes(replacement_bytes)
                mutated = True
            return real_persist(*args, **kwargs)

        with mock.patch.object(
            inference,
            "_persist_manifests",
            side_effect=mutate_alias_before_persist,
        ):
            with self.assertRaisesRegex(ValueError, "changed|selection|snapshot"):
                self.run_with_stub(config)
        self.assertTrue(mutated)
        self.assertFalse(config.provenance_path.exists())

    def test_run_record_contains_checkpoint_and_manifest_sha256(self):
        self.make_dataset("alpha.data")
        config = self.config()
        result, _ = self.run_with_stub(config)
        provenance = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(result, provenance)
        self.assertEqual(provenance["checkpoint_sha256"], sha256_file(self.checkpoint))
        self.assertEqual(
            provenance["input_manifest_sha256"], sha256_file(config.input_manifest_path)
        )
        self.assertEqual(
            provenance["selection_manifest_sha256"],
            sha256_file(config.effective_selection_manifest_path),
        )

    def test_model_batch_does_not_gain_ground_truth_during_device_transfer(self):
        self.make_dataset("alpha.data")
        _, calls = self.run_with_stub(self.config())
        self.assertEqual(len(calls), 1)
        self.assertNotIn("nml", calls[0])
        self.assertNotIn("normal", calls[0])
        self.assertNotIn("ground_truth", calls[0])

    def test_save_exr_is_mandatory_before_model_loader(self):
        self.make_dataset("alpha.data")
        called = False

        def model_loader(_config, _device):
            nonlocal called
            called = True
            raise AssertionError("model loader must not run")

        with self.assertRaisesRegex(ValueError, "save_exr"):
            run_lino_inference(
                self.config(save_exr=False),
                model_loader=model_loader,
            )
        self.assertFalse(called)

    def test_cuda_fp32_is_rejected_before_model_loader(self):
        self.make_dataset("alpha.data")
        called = False

        def model_loader(_config, _device):
            nonlocal called
            called = True
            raise AssertionError("model loader must not run")

        with mock.patch.object(inference.torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "bf16"):
                run_lino_inference(
                    self.config(device="cuda", precision="fp32"),
                    model_loader=model_loader,
                )
        self.assertFalse(called)

    def test_supported_cuda_autocast_uses_bfloat16(self):
        fake_context = mock.MagicMock()
        with mock.patch.object(inference.torch, "autocast", return_value=fake_context) as autocast:
            with inference._autocast_context(torch.device("cuda"), torch.bfloat16):
                pass
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    def test_manifest_alias_is_immutable_and_preserves_exact_order(self):
        self.make_dataset("alpha.data")
        alias = self.output_root / "external" / "selected_lights.json"
        alias.parent.mkdir(parents=True)
        alias.write_bytes(
            b'{"alpha.data":["image_003.exr","image_001.exr"]}\n'
        )
        config = self.config(light_selection="manifest", selection_manifest=alias)
        before = alias.read_bytes()
        result, calls = self.run_with_stub(config)
        self.assertEqual(alias.read_bytes(), before)
        self.assertEqual(
            calls[0]["metadata"]["selected_images"],
            ["image_003.exr", "image_001.exr"],
        )
        provenance = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        self.assertEqual(provenance["selection_manifest_sha256"], sha256_file(alias))
        self.assertEqual(
            provenance["selection_manifest_sha256"],
            provenance["effective_selection_manifest_sha256"],
        )
        self.assertEqual(
            Path(provenance["effective_selection_manifest_path"]).resolve(), alias.resolve()
        )
        self.assertEqual(
            Path(provenance["selection_manifest_path"]).resolve(), alias.resolve()
        )
        self.assertEqual(result["selection_manifest_sha256"], sha256_file(alias))

    def test_cpu_mixed_precision_is_rejected_before_model_loader(self):
        self.make_dataset("alpha.data")
        called = False

        def model_loader(_config, _device):
            nonlocal called
            called = True
            raise AssertionError("model loader must not run")

        with self.assertRaisesRegex(ValueError, "CPU"):
            run_lino_inference(
                self.config(precision="bf16"),
                model_loader=model_loader,
            )
        self.assertFalse(called)

    def test_dataset_length_mismatch_is_rejected_before_model_loader(self):
        self.make_dataset("alpha.data")
        called = False

        def model_loader(_config, _device):
            nonlocal called
            called = True
            raise AssertionError("model loader must not run")

        with self.assertRaisesRegex(ValueError, "dataset length"):
            run_lino_inference(
                self.config(),
                model_loader=model_loader,
                dataset_factory=lambda _config, _manifest: [],
            )
        self.assertFalse(called)

    def _fake_model_modules(self, fake_model_class):
        package = types.ModuleType("src.models")
        package.__path__ = []
        module = types.ModuleType("src.models.Net_module")
        module.LiNo_UniPS = fake_model_class
        return {"src.models": package, "src.models.Net_module": module}

    def _loader_config(self, **overrides):
        return self.config(device="cuda", precision="bf16", **overrides)

    def test_local_loader_constructs_normal_model_and_unwraps_one_state_dict(self):
        captured = {}

        class FakeLiNo:
            def __init__(self, **kwargs):
                captured["constructor"] = kwargs

            def load_state_dict(self, state_dict, strict):
                captured["state_dict"] = state_dict
                captured["strict"] = strict
                return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, device):
                captured["device"] = device
                return self

            def eval(self):
                captured["eval"] = True
                return self

        payload = {"state_dict": {"weight": torch.ones(1)}}
        config = self._loader_config()
        with mock.patch.dict(sys.modules, self._fake_model_modules(FakeLiNo)):
            with mock.patch.object(inference.torch.cuda, "is_available", return_value=True):
                with mock.patch.object(inference.torch, "load", return_value=payload) as load:
                    model = load_local_lino_checkpoint(config, torch.device("cuda"))
        self.assertIsInstance(model, FakeLiNo)
        self.assertEqual(captured["constructor"], {"pixel_samples": 1, "task_name": "SDM_EXR"})
        self.assertIs(captured["state_dict"], payload["state_dict"])
        self.assertFalse(captured["strict"])
        self.assertEqual(captured["device"], torch.device("cuda"))
        self.assertTrue(captured["eval"])
        load.assert_called_once()
        loaded_source = load.call_args.args[0]
        self.assertIsInstance(loaded_source, io.BytesIO)
        self.assertEqual(loaded_source.getvalue(), config.checkpoint.read_bytes())
        self.assertEqual(
            load.call_args.kwargs,
            {"weights_only": False, "map_location": "cpu"},
        )

    def test_local_loader_reports_complete_missing_and_unexpected_keys(self):
        class FakeLiNo:
            def __init__(self, **_kwargs):
                pass

            def load_state_dict(self, _state_dict, strict):
                self.strict = strict
                return types.SimpleNamespace(
                    missing_keys=["missing.a", "missing.b"],
                    unexpected_keys=["unexpected.c"],
                )

            def to(self, _device):
                return self

            def eval(self):
                return self

        config = self._loader_config()
        with mock.patch.dict(sys.modules, self._fake_model_modules(FakeLiNo)):
            with mock.patch.object(inference.torch.cuda, "is_available", return_value=True):
                with mock.patch.object(inference.torch, "load", return_value={"weight": 1}):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"missing_keys=\['missing\.a', 'missing\.b'\].*unexpected_keys=\['unexpected\.c'\]",
                    ):
                        load_local_lino_checkpoint(config, torch.device("cuda"))

    def test_malformed_prediction_shapes_and_nonfinite_values_fail(self):
        self.make_dataset("alpha.data")
        cases = (
            np.zeros((2, 3, 2), dtype=np.float32),
            np.full((2, 3, 3), np.nan, dtype=np.float32),
        )
        for output in cases:
            with self.subTest(shape=output.shape):
                class Stub(torch.nn.Module):
                    def forward(self, _batch):
                        return output

                with self.assertRaisesRegex(ValueError, "LINO forward"):
                    run_lino_inference(
                        self.config(),
                        model_loader=lambda _config, _device: Stub(),
                    )

    def test_atomic_json_write_removes_temporary_file_on_error(self):
        destination = self.root / "atomic" / "run.json"
        with mock.patch.object(inference.json, "dump", side_effect=ValueError("boom")):
            with self.assertRaisesRegex(ValueError, "boom"):
                inference._atomic_json_write(destination, {"ok": True})
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*.tmp")), [])

    def test_models_package_import_is_lazy_without_optional_runtime_stack(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, src.models; "
                    "print('src.models.Net_module' in sys.modules, "
                    "'src.models.Net_pbr_module' in sys.modules)"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")


if __name__ == "__main__":
    unittest.main()
