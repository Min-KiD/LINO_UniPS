"""Analytical tests for source-resolution paired LINO/SDM scoring."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.exr_io import sha256_file, write_normal_exr
from src.comparison.manifest import (
    load_dataset_manifest,
)
from tests.comparison_helpers import write_mask_exr, write_rgb_exr


class ComparisonMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def config(self, **overrides) -> SdmExrInferenceConfig:
        values = dict(
            checkpoint=self.root / "weights" / "lino.pth",
            data_root=self.root / "data",
            output_root=self.root / "outputs",
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
            mask_margin=7,
            max_image_resolution=1024,
            pixel_samples=17,
            precision="fp32",
            device="cpu",
            num_workers=0,
            save_exr=True,
            save_png=True,
        )
        values.update(overrides)
        return SdmExrInferenceConfig(**values)

    def _reset_workspace(self) -> None:
        self.temp_dir.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.addCleanup(self.temp_dir.cleanup)

    def _write_config(self, config: SdmExrInferenceConfig) -> Path:
        import yaml

        path = self.root / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    field: (
                        list(value)
                        if isinstance(value, tuple)
                        else str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for field, value in vars(config).items()
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _prepare_request(self, config: SdmExrInferenceConfig) -> tuple[dict, Path]:
        from compare_sdm import main

        config_path = self._write_config(config)
        repo = self.root / "sdm-repo"
        (repo / "configs").mkdir(parents=True, exist_ok=True)
        (repo / "main.py").write_text("# test SDM entry point\n", encoding="utf-8")
        (repo / "configs" / "baseline_optimized_infer.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        checkpoint = self.root / "sdm.pt"
        checkpoint.write_bytes(b"checkpoint")
        with contextlib.redirect_stdout(io.StringIO()):
            request = main(
                [
                    "prepare-view",
                    "--config",
                    str(config_path),
                    "--sdm-repo",
                    str(repo),
                    "--sdm-checkpoint",
                    str(checkpoint),
                    "--sdm-python",
                    sys.executable,
                ]
            )
        self.request = request
        self.request_path = Path(request["request_path"])
        self.config_path = config_path
        return request, config_path

    def test_identical_orthogonal_and_opposite_normals(self):
        from src.comparison.metrics import angular_metrics

        gt = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        mask = np.ones((1, 1), dtype=bool)
        self.assertAlmostEqual(angular_metrics(gt, gt, mask)["mae"], 0.0, places=8)
        orthogonal = np.asarray([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        self.assertAlmostEqual(
            angular_metrics(gt, orthogonal, mask)["mae"], 90.0, places=8
        )
        opposite = np.asarray([[[-1.0, 0.0, 0.0]]], dtype=np.float32)
        self.assertAlmostEqual(angular_metrics(gt, opposite, mask)["mae"], 180.0, places=8)

    def test_validity_excludes_zero_gt_and_angular_normalizes_in_float64(self):
        from src.comparison.metrics import angular_metrics, normal_validity_mask

        gt = np.asarray(
            [
                [[1.0e-20, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0e-4, 1.0e-4]],
            ],
            dtype=np.float32,
        )
        support = normal_validity_mask(gt)
        np.testing.assert_array_equal(support, [[False, True], [False, False]])
        prediction = gt.copy()
        prediction[1, 1] = [0.0, 1.0, 1.0]
        metrics = angular_metrics(gt, prediction, support)
        self.assertEqual(metrics["valid_pixel_count"], 1)
        self.assertAlmostEqual(metrics["mae"], 0.0, places=5)

    def test_normal_validity_matches_sdm_corrected_v2_unit_band(self):
        from src.comparison.metrics import GT_VALIDITY_POLICY, normal_validity_mask

        self.assertEqual(GT_VALIDITY_POLICY, "sdm_corrected_v2_unit_band")

        normal = np.asarray(
            [[
                [0.0, 0.0, 0.0],
                [1.0e-4, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.499, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]],
            dtype=np.float32,
        )
        expected = np.asarray(
            [[False, False, False, True, True, False, False]], dtype=bool
        )
        np.testing.assert_array_equal(normal_validity_mask(normal), expected)

    def test_empty_support_is_rejected(self):
        from src.comparison.metrics import angular_metrics

        normal = np.ones((2, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "empty"):
            angular_metrics(normal, normal, np.zeros((2, 2), dtype=bool))

    def test_huge_finite_vectors_are_scale_stable_and_complex_is_rejected(self):
        from src.comparison.metrics import angular_metrics, normal_validity_mask

        huge = np.asarray([[[1.0e200, -1.0e200, 1.0e200]]], dtype=np.float64)
        np.testing.assert_array_equal(normal_validity_mask(huge), [[False]])
        self.assertAlmostEqual(
            angular_metrics(huge, huge.copy(), np.ones((1, 1), dtype=bool))["mae"],
            0.0,
            places=8,
        )
        complex_normal = np.asarray([[[1.0 + 0.0j, 0.0j, 0.0j]]])
        with self.assertRaisesRegex(ValueError, "numeric|complex"):
            normal_validity_mask(complex_normal)
        with self.assertRaisesRegex(ValueError, "boolean"):
            angular_metrics(huge, huge, np.ones((1, 1), dtype=np.uint8))

    def _write_object(
        self,
        name: str,
        shape: tuple[int, int],
        direction: tuple[float, float, float],
        *,
        mask_zero: tuple[int, int] | None = None,
    ):
        object_dir = self.root / "data" / name
        object_dir.mkdir(parents=True)
        height, width = shape
        for index in range(2):
            image = np.zeros((height, width, 3), dtype=np.float32)
            image[..., 0] = index + 1
            write_rgb_exr(object_dir / f"image_{index:03d}.exr", image)
        normal = np.zeros((height, width, 3), dtype=np.float32)
        normal[...] = direction
        write_rgb_exr(object_dir / "local_normal.exr", normal)
        mask = np.ones((height, width), dtype=np.float32)
        if mask_zero is not None:
            mask[mask_zero] = 0.0
        write_mask_exr(object_dir / "binary_mask.exr", mask)
        return object_dir

    def _prepare_scoring_fixture(self):
        config = self.config()
        # The first GT-valid pixel has an external mask value of zero.  Score
        # support must remain source-GT-only and therefore still count it.
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0), mask_zero=(0, 0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        config.lino_output_dir.mkdir(parents=True)
        sdm_dir = Path(request["output_path"])
        self.request = request
        self.request_path = Path(request["request_path"])
        self.config_path = config_path
        return config, manifest, sdm_dir

    def _prepare_manifest_scoring_fixture(self):
        selection = self.root / "user-selection.json"
        selection.write_text(
            json.dumps(
                {
                    "wide.data": ["image_000.exr", "image_001.exr"],
                    "tiny.data": ["image_000.exr", "image_001.exr"],
                }
            ),
            encoding="utf-8",
        )
        config = self.config(light_selection="manifest", selection_manifest=selection)
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0), mask_zero=(0, 0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        config.lino_output_dir.mkdir(parents=True)
        sdm_dir = Path(request["output_path"])
        self.request = request
        self.request_path = Path(request["request_path"])
        self.config_path = config_path
        return config, manifest, sdm_dir

    def _write_provenance(
        self,
        config,
        manifest,
        sdm_dir,
        *,
        policy=None,
        stale=False,
        sdm_directions=None,
        finalize=True,
    ):
        from src.comparison.provenance import (
            config_runtime_fingerprint,
            file_identity,
            lino_preprocessing_snapshot,
        )

        config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not config.checkpoint.exists():
            config.checkpoint.write_bytes(b"lino-checkpoint")
        input_digest = sha256_file(config.input_manifest_path)
        selection_path = Path(config.effective_selection_manifest_path)
        effective_path = (
            config.policy_root / "selected_lights.json"
            if config.light_selection == "manifest"
            else selection_path
        )
        selection_digest = sha256_file(selection_path)
        effective_digest = sha256_file(effective_path)
        objects = []
        for record in manifest.objects:
            gt_path = config.data_root / record.relative_dir / record.normal_file
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            lino_path = config.lino_output_dir / record.name / "normal_pred.exr"
            write_normal_exr(lino_path, prediction)
            sdm_path = sdm_dir / f"{Path(record.name).stem}_pred.exr"
            sdm_prediction = prediction
            if sdm_directions and record.name in sdm_directions:
                sdm_prediction = np.zeros_like(prediction)
                sdm_prediction[...] = sdm_directions[record.name]
            write_normal_exr(sdm_path, sdm_prediction)
            objects.append(
                {
                    "object_name": record.name,
                    "output_path": str(lino_path),
                    "output_sha256": sha256_file(lino_path),
                    "source_geometry": {"height": record.height, "width": record.width},
                }
            )
            self.assertEqual(sha256_file(gt_path), record.normal_sha256)
        run = {
            "model": "LINO-UniPS",
            "config_path": str(Path(self.config_path).resolve(strict=True)),
            "config_sha256": sha256_file(self.config_path),
            "config_runtime_fingerprint": config_runtime_fingerprint(config),
            "mask_policy": policy or config.mask_policy,
            "checkpoint_path": str(config.checkpoint.resolve(strict=True)),
            "checkpoint_sha256": sha256_file(config.checkpoint),
            "checkpoint_identity": file_identity(config.checkpoint, label="LINO checkpoint"),
            "preprocessing": lino_preprocessing_snapshot(config),
            "input_manifest_sha256": "stale" if stale else input_digest,
            "selection_manifest_sha256": selection_digest,
            "effective_selection_manifest_sha256": effective_digest,
            "selection_manifest_path": str(selection_path),
            "effective_selection_manifest_path": str(effective_path),
            "objects": objects,
        }
        config.provenance_path.parent.mkdir(parents=True, exist_ok=True)
        config.provenance_path.write_text(json.dumps(run), encoding="utf-8")
        completion_path = Path(self.request["completion_path"])
        if finalize and not completion_path.exists():
            from src.comparison.metrics import finalize_sdm_run

            finalize_sdm_run(
                config,
                self.request_path,
                config_path=self.config_path,
            )

    def _strict_runtime_provenance_fixture(self):
        from src.comparison.inference import _checkpoint_schema_fingerprint
        from src.comparison.provenance import config_runtime_fingerprint, file_identity

        selection = self.root / "strict-selection.json"
        selection.write_text("{}\n", encoding="utf-8")
        config = self.config(
            max_image_num=16,
            light_selection="manifest",
            selection_manifest=selection,
            normal_encoding="unsigned",
            expected_source_geometry=(256, 256),
            mask_margin=8,
            max_image_resolution=512,
            preprocessing_version="private_external_lino_native_v1",
            require_checkpoint_data_contract=True,
        )
        config_path = self._write_config(config)
        config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": torch.ones(1)}, config.checkpoint)
        checkpoint_raw = config.checkpoint.read_bytes()
        checkpoint_digest = sha256_file(config.checkpoint)
        architecture = _checkpoint_schema_fingerprint(checkpoint_raw)
        sidecar = config.checkpoint.with_suffix(".json")
        sidecar_payload = {
            "artifact_kind": "lino_private_inference_weights",
            "checkpoint_sha256": checkpoint_digest,
            "architecture_schema_sha256": architecture,
            "source_revision": "lino-private-exr-training-v2",
            "gt_validity_policy": "sdm_corrected_v2_unit_band",
            "run_kind": "experiment",
            "comparable": True,
            "data_contract": {
                "artifact_kind": "lino_private_training_contract",
                "architecture_schema_sha256": architecture,
                "source_revision": "lino-private-exr-training-v2",
                "gt_validity_policy": "sdm_corrected_v2_unit_band",
                "run_kind": "experiment",
                "comparable": True,
            },
        }
        sidecar.write_text(json.dumps(sidecar_payload) + "\n", encoding="utf-8")
        payload = {
            "config_path": str(config_path.resolve(strict=True)),
            "config_sha256": sha256_file(config_path),
            "config_runtime_fingerprint": config_runtime_fingerprint(config),
            "checkpoint_path": str(config.checkpoint.resolve(strict=True)),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_identity": file_identity(config.checkpoint, label="LINO checkpoint"),
            "checkpoint_sidecar_path": str(sidecar.resolve(strict=True)),
            "checkpoint_sidecar_sha256": sha256_file(sidecar),
            "preprocessing": {
                "mask_margin": config.mask_margin,
                "max_image_resolution": config.max_image_resolution,
                "pixel_samples": config.pixel_samples,
                "precision": config.precision,
                "device": config.device,
                "checkpoint": str(config.checkpoint),
            },
            "run_kind": "experiment",
            "comparable": True,
            "source_revision": "lino-private-exr-training-v2",
            "gt_validity_policy": "sdm_corrected_v2_unit_band",
            "architecture_schema_sha256": architecture,
        }
        return config, config_path, payload, sidecar

    def test_strict_score_provenance_requires_comparable_experiment_metadata(self):
        from src.comparison.metrics import _validate_lino_runtime_provenance

        config, config_path, payload, _sidecar = self._strict_runtime_provenance_fixture()
        _validate_lino_runtime_provenance(
            payload,
            config=config,
            config_path=config_path,
        )

        cases = (
            ("experiment_false", {"comparable": False}, "comparable"),
            ("smoke", {"run_kind": "smoke"}, "smoke"),
            ("unknown", {"run_kind": "unknown"}, "run_kind"),
            ("source", {"source_revision": "other"}, "source_revision"),
            ("architecture", {"architecture_schema_sha256": "0" * 64}, "architecture"),
        )
        for label, updates, pattern in cases:
            with self.subTest(label=label):
                config, config_path, payload, _sidecar = self._strict_runtime_provenance_fixture()
                payload.update(updates)
                with self.assertRaisesRegex(ValueError, pattern):
                    _validate_lino_runtime_provenance(
                        payload,
                        config=config,
                        config_path=config_path,
                    )

        for missing in (
            "run_kind",
            "comparable",
            "source_revision",
            "architecture_schema_sha256",
            "checkpoint_sidecar_path",
            "checkpoint_sidecar_sha256",
        ):
            with self.subTest(missing=missing):
                config, config_path, payload, _sidecar = self._strict_runtime_provenance_fixture()
                payload.pop(missing)
                with self.assertRaisesRegex(ValueError, missing):
                    _validate_lino_runtime_provenance(
                        payload,
                        config=config,
                        config_path=config_path,
                    )

    def test_strict_score_provenance_binds_sidecar_and_checkpoint_fingerprints(self):
        from src.comparison.metrics import _validate_lino_runtime_provenance

        config, config_path, payload, sidecar = self._strict_runtime_provenance_fixture()
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar_payload["data_contract"]["architecture_schema_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(sidecar_payload) + "\n", encoding="utf-8")
        payload["checkpoint_sidecar_sha256"] = sha256_file(sidecar)
        with self.assertRaisesRegex(ValueError, "data contract|architecture"):
            _validate_lino_runtime_provenance(
                payload,
                config=config,
                config_path=config_path,
            )

    def test_strict_score_provenance_requires_final_pth_checkpoint_artifact(self):
        from src.comparison.metrics import _validate_lino_runtime_provenance
        from src.comparison.provenance import (
            config_runtime_fingerprint,
            file_identity,
            lino_preprocessing_snapshot,
        )

        config, _config_path, payload, sidecar = self._strict_runtime_provenance_fixture()
        _validate_lino_runtime_provenance(
            payload,
            config=config,
            config_path=self.root / "config.yaml",
        )

        checkpoint = config.checkpoint.with_suffix(".pt")
        checkpoint.write_bytes(config.checkpoint.read_bytes())
        pt_sidecar = checkpoint.with_suffix(".json")
        pt_sidecar.write_bytes(sidecar.read_bytes())
        pt_config = replace(config, checkpoint=checkpoint)
        pt_config_path = self._write_config(pt_config)
        payload.update(
            {
                "config_path": str(pt_config_path.resolve(strict=True)),
                "config_sha256": sha256_file(pt_config_path),
                "config_runtime_fingerprint": config_runtime_fingerprint(pt_config),
                "checkpoint_path": str(checkpoint.resolve(strict=True)),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_identity": file_identity(checkpoint, label="LINO checkpoint"),
                "checkpoint_sidecar_path": str(pt_sidecar.resolve(strict=True)),
                "checkpoint_sidecar_sha256": sha256_file(pt_sidecar),
                "preprocessing": lino_preprocessing_snapshot(pt_config),
            }
        )
        with self.assertRaisesRegex(ValueError, r"\.pth"):
            _validate_lino_runtime_provenance(
                payload,
                config=pt_config,
                config_path=pt_config_path,
            )

    def test_score_uses_paired_layout_shared_gt_support_and_weighted_aggregates(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(
            config,
            manifest,
            sdm_dir,
            sdm_directions={"wide.data": (0.0, 1.0, 0.0), "tiny.data": (-1.0, 0.0, 0.0)},
        )
        result = score_lino_and_sdm(config, self.request_path, config_path=self.config_path)
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["valid_pixel_count"], 5)
        self.assertAlmostEqual(result["macro_object"]["lino"]["mae"], 0.0, places=8)
        self.assertAlmostEqual(result["pixel_weighted"]["lino"]["mae"], 0.0, places=8)
        self.assertAlmostEqual(result["macro_object"]["sdm"]["mae"], 135.0, places=8)
        self.assertAlmostEqual(result["pixel_weighted"]["sdm"]["mae"], 108.0, places=8)
        self.assertAlmostEqual(result["lino_minus_sdm_mae"], -135.0, places=8)
        self.assertAlmostEqual(result["pixel_weighted_lino_minus_sdm_mae"], -108.0, places=8)
        self.assertTrue((config.policy_root / "comparison" / "per_object.csv").is_file())
        self.assertTrue((config.policy_root / "comparison" / "summary.json").is_file())

    def test_weighted_summary_contains_only_linearly_aggregable_metrics(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        result = score_lino_and_sdm(config, self.request_path, config_path=self.config_path)
        for model_name in ("lino", "sdm"):
            for aggregate_name in ("pixel_weighted", "valid_pixel_weighted"):
                aggregate = result[aggregate_name][model_name]
                self.assertIn("mae", aggregate)
                self.assertIn("accuracy_11_25", aggregate)
                self.assertNotIn("median", aggregate)
                self.assertNotIn("p90", aggregate)
                self.assertNotIn("p95", aggregate)

    def test_score_rejects_missing_extra_geometry_policy_and_stale_manifest(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir, stale=True)
        with self.assertRaisesRegex(ValueError, "manifest|digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._write_provenance(config, manifest, sdm_dir)
        (sdm_dir / "extra_pred.exr").write_bytes(b"not-an-exr")
        with self.assertRaisesRegex(ValueError, "extra|prediction"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        (sdm_dir / "extra_pred.exr").unlink()
        (config.lino_output_dir / manifest.objects[0].name / "normal_pred.exr").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_policy_and_effective_manifest_provenance_mismatch(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        request_path = self.request_path
        request_before = request_path.read_bytes()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["mask_policy"] = "full"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "policy"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        request_path.write_bytes(request_before)
        self._write_provenance(config, manifest, sdm_dir)
        selection = config.effective_selection_manifest_path
        selection.write_text(selection.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_non_comparable_smoke_provenance(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        provenance = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        provenance["run_kind"] = "smoke"
        provenance["comparable"] = False
        config.provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-comparable|smoke"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_lino_policy_mismatch_independently(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        run_path = config.provenance_path
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["mask_policy"] = "full"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "LINO.*policy"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_source_and_effective_selection_digest_mismatch_separately(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_manifest_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        run_path = config.provenance_path
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["selection_manifest_sha256"] = "tampered-source"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection_manifest_sha256"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._write_provenance(config, manifest, sdm_dir)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["effective_selection_manifest_sha256"] = "tampered-effective"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "effective_selection_manifest_sha256"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_requires_all_task4_per_object_provenance_fields(self):
        from src.comparison.metrics import score_lino_and_sdm

        for field in ("output_path", "output_sha256", "source_geometry"):
            with self.subTest(field=field):
                self._reset_workspace()
                config, manifest, sdm_dir = self._prepare_scoring_fixture()
                self._write_provenance(config, manifest, sdm_dir)
                run_path = config.provenance_path
                run = json.loads(run_path.read_text(encoding="utf-8"))
                del run["objects"][0][field]
                run_path.write_text(json.dumps(run), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_requires_canonical_task4_per_object_values(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        run = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        run["objects"][0]["output_path"] = str(self.root / "elsewhere.exr")
        config.provenance_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output_path"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._write_provenance(config, manifest, sdm_dir)
        run = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        run["objects"][0]["output_sha256"] = "wrong"
        config.provenance_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._write_provenance(config, manifest, sdm_dir)
        run = json.loads(config.provenance_path.read_text(encoding="utf-8"))
        run["objects"][0]["source_geometry"] = {"height": 99, "width": 99}
        config.provenance_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "geometry"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_non_builtin_source_geometry_dimensions(self):
        from src.comparison.metrics import score_lino_and_sdm

        for kind in ("bool", "float", "zero", "extra"):
            with self.subTest(kind=kind):
                self._reset_workspace()
                config, manifest, sdm_dir = self._prepare_scoring_fixture()
                self._write_provenance(config, manifest, sdm_dir)
                record = manifest.objects[0]
                geometry = {"height": record.height, "width": record.width}
                if kind == "bool":
                    geometry["height"] = bool(record.height)
                elif kind == "float":
                    geometry["width"] = float(record.width)
                elif kind == "zero":
                    geometry["height"] = 0
                else:
                    geometry["channels"] = 1
                run = json.loads(config.provenance_path.read_text(encoding="utf-8"))
                run["objects"][0]["source_geometry"] = geometry
                config.provenance_path.write_text(json.dumps(run), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "geometry"):
                    score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_requires_task4_and_task5_provenance_path_fields(self):
        from src.comparison.metrics import score_lino_and_sdm

        for path_name, payload_name in (
            ("lino", "selection_manifest_path"),
            ("lino", "effective_selection_manifest_path"),
            ("sdm", "input_manifest_path"),
            ("sdm", "selection_manifest_path"),
            ("sdm", "effective_selection_manifest_path"),
            ("sdm", "output_path"),
        ):
            with self.subTest(path_name=path_name, payload_name=payload_name):
                self._reset_workspace()
                config, manifest, sdm_dir = self._prepare_scoring_fixture()
                self._write_provenance(config, manifest, sdm_dir)
                path = config.provenance_path if path_name == "lino" else self.request_path
                payload = json.loads(path.read_text(encoding="utf-8"))
                del payload[payload_name]
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, payload_name):
                    score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_prediction_geometry_and_extra_lino_object(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        wrong = np.ones((1, 1, 3), dtype=np.float32)
        write_normal_exr(sdm_dir / "wide_pred.exr", wrong)
        with self.assertRaisesRegex(ValueError, "geometry"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        # Restore the expected SDM geometry, then add an unexpected LINO object
        # directory.  Exact object-set validation must fail before scoring.
        restored = np.zeros((1, 4, 3), dtype=np.float32)
        restored[..., 0] = 1.0
        write_normal_exr(sdm_dir / "wide_pred.exr", restored)
        extra = config.lino_output_dir / "extra.data"
        extra.mkdir()
        write_normal_exr(extra / "normal_pred.exr", np.ones((1, 1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "extra"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_sdm_prediction_allowlist_accepts_expected_png_only(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        for record in manifest.objects:
            (sdm_dir / f"{Path(record.name).stem}_pred.png").write_bytes(b"preview")
        # The scorer treats the corresponding PNG as a non-authoritative,
        # optional SDM artifact and still reads only signed EXR predictions.
        score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_sdm_prediction_allowlist_rejects_arbitrary_files_and_dirs(self):
        from src.comparison.metrics import score_lino_and_sdm

        bad_entries = (
            "rogue.png",
            "wide_pred.PNG",
            "wide_pred.EXR",
            "wide_gt.exr",
        )
        for bad_name in bad_entries:
            with self.subTest(bad_name=bad_name):
                self._reset_workspace()
                config, manifest, sdm_dir = self._prepare_scoring_fixture()
                self._write_provenance(config, manifest, sdm_dir)
                (sdm_dir / bad_name).write_bytes(b"unexpected")
                with self.assertRaisesRegex(ValueError, "extra|artifact|prediction"):
                    score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._reset_workspace()
        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        (sdm_dir / "rogue-dir").mkdir()
        with self.assertRaisesRegex(ValueError, "extra|object|prediction"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._reset_workspace()
        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        target = sdm_dir / "wide_pred.exr"
        target.unlink()
        os.symlink(sdm_dir / "tiny_pred.exr", target)
        with self.assertRaisesRegex(ValueError, "missing|symlink|prediction"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_source_gt_digest_and_malformed_predictions(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        source = config.data_root / manifest.objects[0].relative_dir / manifest.objects[0].normal_file
        source.write_bytes(source.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "source|manifest|digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._reset_workspace()
        config, manifest, sdm_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, sdm_dir)
        write_rgb_exr(
            sdm_dir / "wide_pred.exr",
            np.full((1, 4, 3), np.nan, dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "non-finite|failed to read"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_atomic_metric_writes_remove_temporary_files_on_failure(self):
        from src.comparison import metrics

        destination = self.root / "atomic" / "summary.json"
        with mock.patch.object(os, "fsync", side_effect=OSError("flush failed")):
            with self.assertRaises(OSError):
                metrics._atomic_json_write(destination, {"ok": True})
        self.assertFalse(destination.exists())
        self.assertEqual(tuple(destination.parent.glob(".summary.json.*.tmp")), ())

        csv_destination = self.root / "atomic" / "rows.csv"
        with mock.patch.object(os, "fsync", side_effect=OSError("flush failed")):
            with self.assertRaises(OSError):
                metrics._atomic_csv_write(csv_destination, [{"value": 1}], ["value"])
        self.assertFalse(csv_destination.exists())
        self.assertEqual(tuple(csv_destination.parent.glob(".rows.csv.*.tmp")), ())

    def test_score_requires_the_exact_sdm_request_cli(self):
        from compare_sdm import build_parser

        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["score", "--config", "config.yaml"])
        args = parser.parse_args(
            [
                "score",
                "--config",
                "config.yaml",
                "--request",
                "request.json",
            ]
        )
        self.assertEqual(args.request, "request.json")

    def test_finalize_cli_requires_an_exact_request_path(self):
        from compare_sdm import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "finalize-sdm",
                "--config",
                "config.yaml",
                "--request",
                "request.json",
            ]
        )
        self.assertEqual(args.command, "finalize-sdm")
        self.assertEqual(args.request, "request.json")

    def test_finalize_binds_exact_request_and_prediction_hashes(self):
        from compare_sdm import main
        from src.comparison.provenance import sha256_bytes

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(
                output_dir / f"{Path(record.name).stem}_pred.exr", prediction
            )

        completion = main(
            [
                "finalize-sdm",
                "--config",
                str(config_path),
                "--request",
                request["request_path"],
            ]
        )

        completion_path = Path(request["completion_path"])
        self.assertTrue(completion_path.is_file())
        self.assertEqual(completion["request_id"], request["request_id"])
        self.assertEqual(completion["run_fingerprint"], request["run_fingerprint"])
        self.assertEqual(
            completion["request_sha256"],
            sha256_bytes(Path(request["request_path"]).read_bytes()),
        )
        self.assertEqual(
            [item["object_name"] for item in completion["predictions"]],
            [record.name for record in manifest.objects],
        )
        self.assertTrue(all(item["output_sha256"] for item in completion["predictions"]))

    def test_score_requires_completion_and_rejects_prediction_changed_after_finalize(self):
        from compare_sdm import main

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        self._write_provenance(config, manifest, output_dir, finalize=False)

        score_argv = [
            "score",
            "--config",
            str(config_path),
            "--request",
            request["request_path"],
        ]
        with self.assertRaisesRegex(ValueError, "completion|finalize"):
            main(score_argv)

        main(
            [
                "finalize-sdm",
                "--config",
                str(config_path),
                "--request",
                request["request_path"],
            ]
        )
        result = main(score_argv)
        self.assertEqual(result["object_count"], 2)

        changed = np.zeros((1, 4, 3), dtype=np.float32)
        changed[..., 1] = 1.0
        write_normal_exr(output_dir / "wide_pred.exr", changed)
        with self.assertRaisesRegex(ValueError, "digest|completion|changed"):
            main(score_argv)

    def test_finalize_rejects_source_mutation_after_prepare(self):
        from compare_sdm import main

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(
                output_dir / f"{Path(record.name).stem}_pred.exr", prediction
            )
        selected = (
            config.data_root
            / manifest.objects[0].relative_dir
            / manifest.objects[0].selected_images[0]
        )
        selected.write_bytes(selected.read_bytes() + b"changed-after-prepare")

        with self.assertRaisesRegex(ValueError, "source|manifest|rebuild|read"):
            main(
                [
                    "finalize-sdm",
                    "--config",
                    str(config_path),
                    "--request",
                    request["request_path"],
                ]
            )

    def test_finalize_revalidates_the_gt_hidden_view_after_prepare(self):
        from compare_sdm import main

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(
                output_dir / f"{Path(record.name).stem}_pred.exr", prediction
            )
        view_link = (
            config.sdm_view_dir
            / manifest.objects[0].name
            / manifest.objects[0].selected_images[0]
        )
        source_gt = (
            config.data_root
            / manifest.objects[0].relative_dir
            / manifest.objects[0].normal_file
        )
        view_link.unlink()
        os.symlink(source_gt, view_link)
        with self.assertRaisesRegex(ValueError, "view|target|source"):
            main(
                [
                    "finalize-sdm",
                    "--config",
                    str(config_path),
                    "--request",
                    request["request_path"],
                ]
            )

    def test_finalize_rejects_output_directory_replacement_after_validation(self):
        from src.comparison import metrics
        from src.comparison.metrics import finalize_sdm_run

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(output_dir / f"{Path(record.name).stem}_pred.exr", prediction)

        original = metrics._validated_sdm_predictions
        moved = output_dir.with_name(f"{output_dir.name}.moved")

        def replace_after_validation(path, current_manifest):
            result = original(path, current_manifest)
            path.rename(moved)
            path.mkdir()
            return result

        with mock.patch.object(
            metrics, "_validated_sdm_predictions", side_effect=replace_after_validation
        ):
            with self.assertRaisesRegex(ValueError, "output|replaced|identity"):
                finalize_sdm_run(config, request["request_path"], config_path=config_path)

    def test_score_rejects_output_directory_replacement_before_acceptance(self):
        from src.comparison import metrics
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, output_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, output_dir)
        original = metrics._validated_sdm_predictions
        moved = output_dir.with_name(f"{output_dir.name}.moved")

        def replace_after_validation(path, current_manifest):
            result = original(path, current_manifest)
            path.rename(moved)
            path.mkdir()
            return result

        with mock.patch.object(
            metrics, "_validated_sdm_predictions", side_effect=replace_after_validation
        ):
            with self.assertRaisesRegex(ValueError, "output|replaced|identity"):
                score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_score_rejects_stale_lino_config_or_checkpoint_provenance(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, output_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, output_dir)
        run_path = config.provenance_path
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["config_sha256"] = "stale-config"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config|provenance|digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

        self._write_provenance(config, manifest, output_dir)
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["checkpoint_sha256"] = "stale-checkpoint"
        run_path.write_text(json.dumps(run), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checkpoint|provenance|digest"):
            score_lino_and_sdm(config, self.request_path, config_path=self.config_path)

    def test_finalize_refuses_to_overwrite_an_existing_completion(self):
        from compare_sdm import main

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        request, config_path = self._prepare_request(config)
        manifest = load_dataset_manifest(config.input_manifest_path)
        output_dir = Path(request["output_path"])
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(
                output_dir / f"{Path(record.name).stem}_pred.exr", prediction
            )
        argv = [
            "finalize-sdm",
            "--config",
            str(config_path),
            "--request",
            request["request_path"],
        ]
        main(argv)

        with self.assertRaisesRegex(FileExistsError, "exist|overwrite"):
            main(argv)

    def test_score_rejects_tampered_completion_record(self):
        from src.comparison.metrics import score_lino_and_sdm

        config, manifest, output_dir = self._prepare_scoring_fixture()
        self._write_provenance(config, manifest, output_dir)
        completion_path = Path(self.request["completion_path"])
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["request_sha256"] = "tampered"
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "request|completion|exact"):
            score_lino_and_sdm(
                config,
                self.request_path,
                config_path=self.config_path,
            )

    def test_fresh_request_never_reuses_a_previous_run_directory(self):
        from compare_sdm import main

        config = self.config()
        self._write_object("wide.data", (1, 4), (1.0, 0.0, 0.0))
        self._write_object("tiny.data", (1, 1), (1.0, 0.0, 0.0))
        first, config_path = self._prepare_request(config)
        second, _ = self._prepare_request(config)
        first_output = Path(first["output_path"])
        second_output = Path(second["output_path"])
        self.assertNotEqual(first_output, second_output)
        manifest = load_dataset_manifest(config.input_manifest_path)
        for record in manifest.objects:
            prediction = np.zeros((record.height, record.width, 3), dtype=np.float32)
            prediction[..., 0] = 1.0
            write_normal_exr(
                first_output / f"{Path(record.name).stem}_pred.exr", prediction
            )

        with self.assertRaisesRegex(ValueError, "completion|finalize"):
            main(
                [
                    "score",
                    "--config",
                    str(config_path),
                    "--request",
                    second["request_path"],
                ]
            )

    def test_generated_sdm_prediction_basename_collisions_are_rejected(self):
        from src.comparison.manifest import DatasetManifest, ObjectRecord
        from src.comparison.metrics import _expected_sdm_paths

        output = self.root / "sdm"
        output.mkdir()
        records = tuple(
            ObjectRecord(
                name=name,
                relative_dir=name,
                height=1,
                width=1,
                selected_images=("image_000.exr",),
                image_sha256=("a",),
                normal_file="local_normal.exr",
                normal_sha256="b",
                mask_file=None,
                mask_sha256=None,
            )
            for name in ("alpha.data", "alpha")
        )
        manifest = DatasetManifest(
            version=1,
            data_root=str(self.root / "data"),
            seed=1,
            max_image_num=1,
            objects=records,
        )
        with self.assertRaisesRegex(ValueError, "collision|basename"):
            _expected_sdm_paths(output, manifest)

    def test_prediction_decoding_uses_one_immutable_byte_snapshot(self):
        from src.comparison.metrics import _read_prediction_artifact

        path = self.root / "prediction.exr"
        original = np.ones((1, 1, 3), dtype=np.float32)
        write_normal_exr(path, original)
        artifact = _read_prediction_artifact(path, label="prediction")
        path.write_bytes(b"replaced-after-read")
        np.testing.assert_array_equal(artifact["array"], original)
        self.assertTrue(artifact["sha256"])

if __name__ == "__main__":
    unittest.main()
