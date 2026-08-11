"""Tests for the GT-hidden SDM input view and command request."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.comparison.config import SdmExrInferenceConfig
from src.comparison.manifest import build_dataset_manifest
from tests.comparison_helpers import make_object


class SdmViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "data"
        self.output_root = self.root / "outputs"
        self.object_dir = make_object(self.data_root, "alpha.data", image_count=4)

    def config(self, **overrides) -> SdmExrInferenceConfig:
        values = dict(
            checkpoint=self.root / "weights" / "lino.pth",
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
            normal_filenames=("local_normal.exr", "normal.exr"),
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

    def _write_sdm_runtime(self) -> tuple[Path, Path]:
        repo = self.root / "sdm"
        (repo / "configs").mkdir(parents=True)
        (repo / "main.py").write_text("# test stub\n", encoding="utf-8")
        (repo / "configs" / "baseline_optimized_infer.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        checkpoint = self.root / "sdm.pt"
        checkpoint.write_bytes(b"checkpoint")
        return repo, checkpoint

    def test_external_view_links_selected_images_and_mask_only(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        view = prepare_sdm_view(config, manifest)
        destination = view / "alpha.data"

        self.assertEqual(
            sorted(path.name for path in destination.iterdir()),
            sorted((*manifest.objects[0].selected_images, "binary_mask.exr")),
        )
        for filename in manifest.objects[0].selected_images:
            link = destination / filename
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), self.object_dir / filename)
        self.assertTrue((destination / "binary_mask.exr").is_symlink())
        self.assertEqual(
            (destination / "binary_mask.exr").resolve(), self.object_dir / "binary_mask.exr"
        )
        self.assertFalse((destination / "local_normal.exr").exists())
        self.assertFalse((destination / "normal.exr").exists())

    def test_validate_sdm_view_attests_exact_tree_and_rejects_gt_or_wrong_links(self):
        from src.comparison.sdm_view import prepare_sdm_view, validate_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        view = prepare_sdm_view(config, manifest)
        attestation = validate_sdm_view(config, manifest)
        self.assertEqual(attestation["view_path"], str(view.resolve(strict=True)))
        self.assertTrue(attestation["view_tree_sha256"])
        self.assertIn("root_identity", attestation)
        self.assertIn("object_identities", attestation)

        object_view = view / "alpha.data"
        selected = manifest.objects[0].selected_images[0]
        (object_view / selected).unlink()
        os.symlink(self.object_dir / "local_normal.exr", object_view / selected)
        with self.assertRaisesRegex(ValueError, "source|target|view|digest"):
            validate_sdm_view(config, manifest)

    def test_validate_sdm_view_rewalks_after_link_and_entry_race(self):
        from src.comparison import sdm_view
        from src.comparison.sdm_view import prepare_sdm_view, validate_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        view = prepare_sdm_view(config, manifest)
        object_view = view / "alpha.data"
        selected = manifest.objects[0].selected_images[0]
        other = manifest.objects[0].selected_images[1]
        selected_link = object_view / selected
        selected_link.unlink()
        os.symlink(self.object_dir / other, selected_link)
        original_fd_names = sdm_view._fd_names
        raced = False

        def race_names(fd, *, label):
            nonlocal raced
            names = original_fd_names(fd, label=label)
            if not raced and label.endswith("alpha.data"):
                raced = True
                selected_link.unlink()
                os.symlink(self.object_dir / selected, selected_link)
                os.symlink(self.object_dir / "local_normal.exr", object_view / "local_normal.exr")
            return names

        with mock.patch.object(sdm_view, "_fd_names", side_effect=race_names):
            with self.assertRaisesRegex(ValueError, "stale|basename|ground|target|view"):
                validate_sdm_view(config, manifest)

    def test_full_view_has_no_mask_or_recognized_normal(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config(mask_policy="full")
        manifest = build_dataset_manifest(config)
        destination = prepare_sdm_view(config, manifest) / "alpha.data"

        self.assertEqual(
            sorted(path.name for path in destination.iterdir()),
            sorted(manifest.objects[0].selected_images),
        )
        for forbidden in ("binary_mask.exr", "mask.png", "local_normal.exr", "normal.exr"):
            self.assertFalse((destination / forbidden).exists())

    def test_matching_links_are_idempotent(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        first = prepare_sdm_view(config, manifest)
        link = first / "alpha.data" / manifest.objects[0].selected_images[0]
        before = os.readlink(link)
        second = prepare_sdm_view(config, manifest)

        self.assertEqual(second, first)
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), before)

    def test_conflicting_destination_fails_without_mutating_user_data(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        conflict_dir = config.sdm_view_dir / "alpha.data"
        conflict_dir.mkdir(parents=True)
        conflict = conflict_dir / manifest.objects[0].selected_images[0]
        conflict.write_bytes(b"user-owned")

        with self.assertRaisesRegex(ValueError, "conflict"):
            prepare_sdm_view(config, manifest)
        self.assertFalse(conflict.is_symlink())
        self.assertEqual(conflict.read_bytes(), b"user-owned")

    def test_source_digest_mismatch_fails_before_any_link(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        selected = manifest.objects[0].selected_images[0]
        (self.object_dir / selected).write_bytes(b"changed")

        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            prepare_sdm_view(config, manifest)
        self.assertFalse(config.sdm_view_dir.exists())

    def test_stale_view_root_entry_fails_without_mutation(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        config.sdm_view_dir.mkdir(parents=True)
        stale = config.sdm_view_dir / "stale.data"
        stale.write_bytes(b"user-owned")

        with self.assertRaisesRegex(ValueError, "stale|unexpected|conflict"):
            prepare_sdm_view(config, manifest)
        self.assertEqual(stale.read_bytes(), b"user-owned")
        self.assertFalse((config.sdm_view_dir / "alpha.data").exists())

    def test_existing_object_requires_exact_planned_basename_set(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        object_view = config.sdm_view_dir / "alpha.data"
        object_view.mkdir(parents=True)
        stale = object_view / "stale_selected.exr"
        stale.write_bytes(b"user-owned")

        with self.assertRaisesRegex(ValueError, "stale|unexpected|conflict"):
            prepare_sdm_view(config, manifest)
        self.assertEqual(stale.read_bytes(), b"user-owned")
        self.assertEqual(tuple(object_view.iterdir()), (stale,))

    def test_partial_existing_object_is_fail_closed(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        object_view = config.sdm_view_dir / "alpha.data"
        object_view.mkdir(parents=True)

        with self.assertRaisesRegex(ValueError, "planned|stale|unexpected|conflict"):
            prepare_sdm_view(config, manifest)
        self.assertEqual(tuple(object_view.iterdir()), ())

    def test_full_view_rejects_stale_external_mask_without_deleting_it(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config(mask_policy="full")
        manifest = build_dataset_manifest(config)
        object_view = config.sdm_view_dir / "alpha.data"
        object_view.mkdir(parents=True)
        stale_mask = object_view / "binary_mask.exr"
        stale_mask.write_bytes(b"user-owned-mask")

        with self.assertRaisesRegex(ValueError, "stale|forbidden|conflict"):
            prepare_sdm_view(config, manifest)
        self.assertEqual(stale_mask.read_bytes(), b"user-owned-mask")

    def test_replaced_view_root_is_rejected_before_outside_write(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        view_root = config.sdm_view_dir
        outside = self.root / "outside"
        swapped = False
        original_mkdir = Path.mkdir

        def race_mkdir(path, *args, **kwargs):
            nonlocal swapped
            result = original_mkdir(path, *args, **kwargs)
            if Path(path) == view_root and not swapped:
                swapped = True
                Path(path).rmdir()
                outside.mkdir(parents=True)
                os.symlink(outside, path, target_is_directory=True)
            return result

        with mock.patch.object(Path, "mkdir", new=race_mkdir):
            with self.assertRaisesRegex(ValueError, "view|directory|symlink|destination"):
                prepare_sdm_view(config, manifest)
        self.assertTrue(view_root.is_symlink())
        self.assertFalse((outside / "alpha.data").exists())

    def test_replaced_object_directory_is_rejected_before_outside_write(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        view_root = config.sdm_view_dir
        object_view = view_root / "alpha.data"
        outside = self.root / "outside"
        swapped = False
        original_mkdir = os.mkdir
        original_symlink = os.symlink

        def race_mkdir(path, *args, **kwargs):
            nonlocal swapped
            result = original_mkdir(path, *args, **kwargs)
            is_object_creation = Path(path) == object_view or (
                Path(path).name == object_view.name and kwargs.get("dir_fd") is not None
            )
            if is_object_creation and not swapped:
                swapped = True
                if kwargs.get("dir_fd") is None:
                    Path(path).rmdir()
                else:
                    object_view.rmdir()
                outside.mkdir(parents=True)
                original_symlink(outside, object_view, target_is_directory=True)
            return result

        with mock.patch.object(os, "mkdir", new=race_mkdir):
            with self.assertRaisesRegex(ValueError, "object|directory|symlink|destination"):
                prepare_sdm_view(config, manifest)
        self.assertTrue(object_view.is_symlink())
        self.assertFalse((outside / manifest.objects[0].selected_images[0]).exists())

    def test_descriptor_relative_link_cannot_follow_object_path_replacement(self):
        from src.comparison.sdm_view import prepare_sdm_view

        config = self.config()
        manifest = build_dataset_manifest(config)
        object_view = config.sdm_view_dir / "alpha.data"
        outside = self.root / "outside"
        swapped = False
        original_symlink = os.symlink

        def race_symlink(source, destination, *args, **kwargs):
            nonlocal swapped
            if not swapped:
                swapped = True
                object_view.rmdir()
                outside.mkdir(parents=True)
                original_symlink(outside, object_view, target_is_directory=True)
            return original_symlink(source, destination, *args, **kwargs)

        with mock.patch.object(os, "symlink", new=race_symlink):
            with self.assertRaises((ValueError, OSError)):
                prepare_sdm_view(config, manifest)
        self.assertTrue(object_view.is_symlink())
        self.assertFalse((outside / manifest.objects[0].selected_images[0]).exists())

    def test_command_overrides_are_exact_tokens(self):
        from src.comparison.sdm_view import build_sdm_command

        config = self.config(
            object_suffix=".custom",
            image_prefix="observation_",
            max_image_num=9,
            mask_margin=13,
            seed=404,
        )
        repo = self.root / "sdm"
        checkpoint = self.root / "sdm-checkpoint.pt"
        python = self.root / "python"
        argv = build_sdm_command(config, repo, checkpoint, python)

        self.assertEqual(
            argv,
            [
                str(python),
                str(repo / "main.py"),
                "infer",
                "--config",
                str(repo / "configs" / "baseline_optimized_infer.yaml"),
                "--checkpoint",
                str(checkpoint),
                "--test-dir",
                str(config.sdm_view_dir),
                "--output-dir",
                str(config.sdm_output_dir),
                "--light-selection",
                "manifest",
                "--selection-manifest",
                str(config.effective_selection_manifest_path),
                "--mask-policy",
                config.mask_policy,
                "--no-save-ground-truth",
                "--max-image-num",
                str(config.max_image_num),
                "--test-ext",
                config.object_suffix,
                "--test-prefix",
                config.image_prefix,
                "--mask-margin",
                str(config.mask_margin),
                "--seed",
                str(config.seed),
            ],
        )

    def test_command_accepts_a_fresh_per_request_output_directory(self):
        from src.comparison.sdm_view import build_sdm_command

        config = self.config()
        fresh_output = config.sdm_output_dir / "request-abc123"
        argv = build_sdm_command(
            config,
            self.root / "sdm",
            self.root / "sdm-checkpoint.pt",
            self.root / "python",
            output_dir=fresh_output,
        )

        output_index = argv.index("--output-dir")
        self.assertEqual(argv[output_index + 1], str(fresh_output))

    def test_command_canonicalizes_all_filesystem_arguments(self):
        from src.comparison.sdm_view import build_sdm_command

        config = self.config(
            data_root=Path("relative-data"),
            output_root=Path("relative-output"),
            light_selection="manifest",
            selection_manifest=Path("relative-selection.json"),
        )
        argv = build_sdm_command(
            config,
            self.root / "sdm",
            self.root / "sdm-checkpoint.pt",
            self.root / "python",
        )

        for flag in ("--config", "--checkpoint", "--test-dir", "--output-dir", "--selection-manifest"):
            value = Path(argv[argv.index(flag) + 1])
            self.assertTrue(value.is_absolute(), f"{flag} remained relative: {value}")

    def test_prepare_view_cli_keeps_manifest_source_immutable_and_writes_request(self):
        from compare_sdm import main

        selection = self.root / "user-selection.json"
        selection.write_text(
            json.dumps({"alpha.data": ["image_003.exr", "image_001.exr"]}),
            encoding="utf-8",
        )
        config = self.config(light_selection="manifest", selection_manifest=selection)
        config_path = self.root / "config.yaml"
        import yaml

        config_path.write_text(
            yaml.safe_dump(
                {
                    field: (list(value) if isinstance(value, tuple) else str(value) if isinstance(value, Path) else value)
                    for field, value in vars(config).items()
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        repo = self.root / "sdm"
        (repo / "configs").mkdir(parents=True)
        (repo / "main.py").write_text("# test stub\n", encoding="utf-8")
        (repo / "configs" / "baseline_optimized_infer.yaml").write_text("{}\n", encoding="utf-8")
        checkpoint = self.root / "sdm.pt"
        checkpoint.write_bytes(b"checkpoint")
        source_before = selection.read_bytes()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            prepared = main(
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

        self.assertEqual(selection.read_bytes(), source_before)
        canonical = config.policy_root / "selected_lights.json"
        self.assertTrue(canonical.is_file())
        request_path = Path(prepared["request_path"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        self.assertEqual(request["mask_policy"], "external")
        self.assertEqual(request["selection_manifest_path"], str(selection))
        self.assertEqual(request["effective_selection_manifest_path"], str(canonical))
        self.assertIn("--no-save-ground-truth", request["argv"])
        self.assertIn(str(config.sdm_view_dir), output.getvalue())

    def test_prepare_view_cli_seeded_persists_selection_and_does_not_launch_sdm(self):
        from compare_sdm import main

        config = self.config()
        config_path = self.root / "config.yaml"
        import yaml

        config_path.write_text(
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
        repo = self.root / "sdm"
        (repo / "configs").mkdir(parents=True)
        (repo / "main.py").write_text("# test stub\n", encoding="utf-8")
        (repo / "configs" / "baseline_optimized_infer.yaml").write_text(
            "{}\n", encoding="utf-8"
        )
        checkpoint = self.root / "sdm.pt"
        checkpoint.write_bytes(b"checkpoint")

        with mock.patch("subprocess.run") as run:
            prepared = main(
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

        run.assert_not_called()
        selection = config.effective_selection_manifest_path
        self.assertTrue(selection.is_file())
        request = json.loads(Path(prepared["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(request["selection_manifest_path"], str(selection))
        self.assertEqual(
            request["selection_manifest_sha256"],
            request["effective_selection_manifest_sha256"],
        )
        self.assertTrue(Path(request["output_path"]).is_dir())
        self.assertEqual(tuple(Path(request["output_path"]).iterdir()), ())

    def test_prepare_view_creates_unique_request_and_fresh_output_directory(self):
        from compare_sdm import main

        config = self.config()
        config_path = self._write_config(config)
        repo, checkpoint = self._write_sdm_runtime()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            first = main(
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
            second = main(
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

        first_request = Path(first["request_path"])
        second_request = Path(second["request_path"])
        first_output = Path(first["output_path"])
        second_output = Path(second["output_path"])
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first_request, second_request)
        self.assertNotEqual(first_output, second_output)
        self.assertTrue(first_request.is_file())
        self.assertTrue(second_request.is_file())
        self.assertEqual(first_output.parent, config.sdm_output_dir)
        self.assertEqual(second_output.parent, config.sdm_output_dir)
        self.assertEqual(tuple(first_output.iterdir()), ())
        self.assertEqual(tuple(second_output.iterdir()), ())
        self.assertIn("run_fingerprint", first)
        self.assertIn("finalize-sdm", output.getvalue())
        self.assertIn(str(first_request), output.getvalue())

    def test_prepare_view_refuses_a_preexisting_per_request_output_path(self):
        from compare_sdm import main

        config = self.config()
        config_path = self._write_config(config)
        repo, checkpoint = self._write_sdm_runtime()
        collision = config.sdm_output_dir / "fixed-request"
        collision.mkdir(parents=True)

        with mock.patch("secrets.token_hex", return_value="fixed-request"):
            with self.assertRaisesRegex((FileExistsError, ValueError), "exist|fresh|output"):
                main(
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

    def test_prepare_view_refuses_a_symlink_per_request_output_path(self):
        from compare_sdm import main

        config = self.config()
        config_path = self._write_config(config)
        repo, checkpoint = self._write_sdm_runtime()
        config.sdm_output_dir.mkdir(parents=True)
        outside = self.root / "outside-output"
        outside.mkdir()
        os.symlink(outside, config.sdm_output_dir / "fixed-request")

        with mock.patch("secrets.token_hex", return_value="fixed-request"):
            with self.assertRaisesRegex((FileExistsError, ValueError), "exist|fresh|output"):
                main(
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
        self.assertEqual(tuple(outside.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
