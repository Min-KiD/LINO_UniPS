"""Focused descriptor-pinning regressions for provenance publication."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ProvenancePinningTests(unittest.TestCase):
    def test_owned_artifact_returns_the_committed_file_identity(self):
        from src.comparison.inference import _write_bytes_at_fd

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                directory = os.fstat(directory_fd)
                committed = _write_bytes_at_fd(
                    directory_fd,
                    "run.json",
                    b'{"complete":true}\n',
                    expected_directory_identity={
                        "dev": int(directory.st_dev),
                        "ino": int(directory.st_ino),
                    },
                    label="LINO run provenance",
                )
            finally:
                os.close(directory_fd)

            published = (output / "run.json").lstat()
            self.assertEqual(
                committed,
                (int(published.st_dev), int(published.st_ino)),
            )

    def test_directory_walk_closes_child_descriptor_when_fstat_fails(self):
        from src.comparison.provenance import open_or_create_directory

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "output"
            target.mkdir()
            real_open = os.open
            real_fstat = os.fstat
            child_fd: int | None = None

            def capture_child(path, *args, **kwargs):
                nonlocal child_fd
                descriptor = real_open(path, *args, **kwargs)
                if path == target.name:
                    child_fd = descriptor
                return descriptor

            def fail_child_fstat(descriptor):
                if descriptor == child_fd:
                    raise OSError("forced child fstat failure")
                return real_fstat(descriptor)

            try:
                with mock.patch.object(os, "open", side_effect=capture_child):
                    with mock.patch.object(os, "fstat", side_effect=fail_child_fstat):
                        with self.assertRaises(OSError):
                            open_or_create_directory(target, label="output directory")
                self.assertIsNotNone(child_fd)
                with self.assertRaises(OSError):
                    os.fstat(child_fd)
            finally:
                if child_fd is not None:
                    try:
                        os.close(child_fd)
                    except OSError:
                        pass

    def test_owned_artifact_replace_does_not_modify_a_preexisting_hardlink(self):
        from src.comparison.inference import _write_bytes_at_fd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            victim = root / "victim.exr"
            victim.write_bytes(b"user-owned")
            destination = output / "normal_pred.exr"
            os.link(victim, destination)
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                info = os.fstat(directory_fd)
                _write_bytes_at_fd(
                    directory_fd,
                    destination.name,
                    b"new-prediction",
                    expected_directory_identity={
                        "dev": int(info.st_dev),
                        "ino": int(info.st_ino),
                    },
                    label="LINO prediction",
                )
            finally:
                os.close(directory_fd)

            self.assertEqual(victim.read_bytes(), b"user-owned")
            self.assertEqual(destination.read_bytes(), b"new-prediction")

    def test_owned_artifact_write_failure_preserves_previous_bytes(self):
        from src.comparison.inference import _write_bytes_at_fd

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            destination = output / "run.json"
            destination.write_bytes(b"previous-complete-run")
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                info = os.fstat(directory_fd)
                with mock.patch.object(os, "write", side_effect=OSError("disk full")):
                    with self.assertRaisesRegex(ValueError, "publish|run"):
                        _write_bytes_at_fd(
                            directory_fd,
                            destination.name,
                            b"replacement",
                            expected_directory_identity={
                                "dev": int(info.st_dev),
                                "ino": int(info.st_ino),
                            },
                            label="LINO run provenance",
                        )
            finally:
                os.close(directory_fd)

            self.assertEqual(destination.read_bytes(), b"previous-complete-run")

    def test_owned_artifact_swap_before_commit_does_not_truncate_swapped_inode(self):
        from src.comparison.inference import _write_bytes_at_fd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            destination = output / "normal_pred.exr"
            destination.write_bytes(b"previous-prediction")
            victim = root / "victim.exr"
            victim.write_bytes(b"user-owned")
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            real_open = os.open
            real_replace = os.replace
            swapped = False

            def swap_on_old_or_new_commit(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == destination.name and flags & os.O_TRUNC and not swapped:
                    destination.unlink()
                    os.link(victim, destination)
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            def swap_on_atomic_commit(source, target, *args, **kwargs):
                nonlocal swapped
                if target == destination.name and not swapped:
                    destination.unlink()
                    os.link(victim, destination)
                    swapped = True
                return real_replace(source, target, *args, **kwargs)

            try:
                info = os.fstat(directory_fd)
                with mock.patch.object(os, "open", side_effect=swap_on_old_or_new_commit):
                    with mock.patch.object(os, "replace", side_effect=swap_on_atomic_commit):
                        try:
                            _write_bytes_at_fd(
                                directory_fd,
                                destination.name,
                                b"new-prediction",
                                expected_directory_identity={
                                    "dev": int(info.st_dev),
                                    "ino": int(info.st_ino),
                                },
                                label="LINO prediction",
                            )
                        except ValueError:
                            pass
            finally:
                os.close(directory_fd)

            self.assertTrue(swapped)
            self.assertEqual(victim.read_bytes(), b"user-owned")

    def test_owned_artifact_rejects_and_rolls_back_a_post_commit_swap(self):
        from src.comparison.inference import _write_bytes_at_fd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            destination = output / "normal_pred.exr"
            destination.write_bytes(b"previous-prediction")
            victim = root / "victim.exr"
            victim.write_bytes(b"user-owned")
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            real_replace = os.replace
            swapped = False

            def swap_after_commit(source, target, *args, **kwargs):
                nonlocal swapped
                result = real_replace(source, target, *args, **kwargs)
                if target == destination.name and not swapped:
                    destination.unlink()
                    os.link(victim, destination)
                    swapped = True
                return result

            try:
                info = os.fstat(directory_fd)
                with mock.patch.object(os, "replace", side_effect=swap_after_commit):
                    with self.assertRaisesRegex(ValueError, "replaced|publication"):
                        _write_bytes_at_fd(
                            directory_fd,
                            destination.name,
                            b"new-prediction",
                            expected_directory_identity={
                                "dev": int(info.st_dev),
                                "ino": int(info.st_ino),
                            },
                            label="LINO prediction",
                        )
            finally:
                os.close(directory_fd)

            self.assertTrue(swapped)
            self.assertEqual(victim.read_bytes(), b"user-owned")
            self.assertEqual(destination.read_bytes(), b"previous-prediction")

    def test_prepare_and_inference_manifest_publication_reject_path_replacement(self):
        from compare_sdm import _persist_manifests as prepare_persist_manifests
        from src.comparison.inference import _persist_manifests as inference_persist_manifests
        from src.comparison.manifest import DatasetManifest

        for persist in (prepare_persist_manifests, inference_persist_manifests):
            with self.subTest(persist=persist.__module__):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    policy = root / "output" / "external"
                    policy.mkdir(parents=True)
                    moved = root / "detached-policy"
                    outside = root / "outside"
                    outside.mkdir()
                    config = SimpleNamespace(
                        policy_root=policy,
                        light_selection="seeded",
                        selection_manifest=None,
                        effective_selection_manifest_path=policy / "selected_lights.json",
                        input_manifest_path=policy / "input_manifest.json",
                    )
                    manifest = DatasetManifest(
                        version=1,
                        data_root="/data",
                        seed=1,
                        max_image_num=1,
                        objects=(),
                    )
                    real_open = os.open
                    swapped = False

                    def open_after_path_swap(path, flags, *args, **kwargs):
                        nonlocal swapped
                        basename = Path(path).name
                        if (
                            basename.startswith(".input_manifest.json.")
                            and flags & os.O_CREAT
                            and not swapped
                        ):
                            policy.rename(moved)
                            os.symlink(outside, policy, target_is_directory=True)
                            swapped = True
                        return real_open(path, flags, *args, **kwargs)

                    with mock.patch.object(os, "open", side_effect=open_after_path_swap):
                        with self.assertRaisesRegex(ValueError, "replaced|manifest|policy"):
                            persist(config, manifest)
                    self.assertTrue(swapped)
                    self.assertEqual(list(outside.iterdir()), [])

    def test_manifest_publication_rechecks_policy_path_after_resolving_results(self):
        import src.comparison.manifest as manifest_module
        from compare_sdm import _persist_manifests
        from src.comparison.manifest import DatasetManifest

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = root / "output" / "external"
            policy.mkdir(parents=True)
            moved = root / "detached-policy"
            config = SimpleNamespace(
                policy_root=policy,
                light_selection="seeded",
                selection_manifest=None,
                effective_selection_manifest_path=policy / "selected_lights.json",
                input_manifest_path=policy / "input_manifest.json",
            )
            record = DatasetManifest(
                version=1,
                data_root="/data",
                seed=1,
                max_image_num=1,
                objects=(),
            )
            real_assert = manifest_module.assert_directory_path_identity
            checked = False

            def replace_after_check(*args, **kwargs):
                nonlocal checked
                result = real_assert(*args, **kwargs)
                if not checked:
                    checked = True
                    policy.rename(moved)
                    policy.mkdir()
                    for name in ("input_manifest.json", "selected_lights.json"):
                        os.link(moved / name, policy / name)
                return result

            with mock.patch.object(
                manifest_module,
                "assert_directory_path_identity",
                side_effect=replace_after_check,
            ):
                with self.assertRaisesRegex(ValueError, "replaced|policy"):
                    _persist_manifests(config, record)
            self.assertTrue(checked)

    def test_atomic_create_json_fails_closed_on_parent_replacement(self):
        from src.comparison.provenance import atomic_create_json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "records"
            parent.mkdir()
            moved = root / "records-moved"
            destination = parent / "request.json"
            original_link = os.link
            swapped = False

            def replace_parent(*args, **kwargs):
                nonlocal swapped
                if not swapped:
                    swapped = True
                    parent.rename(moved)
                    parent.mkdir()
                return original_link(*args, **kwargs)

            with mock.patch.object(os, "link", side_effect=replace_parent):
                with self.assertRaisesRegex(ValueError, "replaced|provenance"):
                    atomic_create_json(destination, {"ok": True})
            self.assertFalse(destination.exists())
            self.assertFalse((moved / "request.json").exists())

    def test_create_fresh_child_directory_fails_closed_on_parent_replacement(self):
        from src.comparison.provenance import create_fresh_child_directory

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "outputs"
            base.mkdir()
            moved = root / "outputs-moved"
            original_stat = os.stat
            swapped = False

            def replace_after_create(*args, **kwargs):
                nonlocal swapped
                result = original_stat(*args, **kwargs)
                if kwargs.get("dir_fd") is not None and args and args[0] == "run-1" and not swapped:
                    swapped = True
                    base.rename(moved)
                    base.mkdir()
                return result

            with mock.patch.object(os, "stat", side_effect=replace_after_create):
                with self.assertRaisesRegex(ValueError, "replaced|fresh"):
                    create_fresh_child_directory(base, "run-1", label="fresh run")


if __name__ == "__main__":
    unittest.main()
