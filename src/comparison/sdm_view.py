"""Build a safe, ground-truth-hidden input view for released SDM inference.

The released SDM loader discovers files by object-directory basename.  This
module presents only the ordered observations selected by LINO's canonical
manifest (and, for the external policy, a mask under SDM's conventional
``binary_mask.exr`` name).  It never copies or removes source/user files.
"""

from __future__ import annotations

import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable

from .config import SdmExrInferenceConfig
from .exr_io import sha256_file
from .manifest import DatasetManifest, ObjectRecord
from .provenance import canonical_json_bytes, sha256_bytes


# A single process owns preparation for one view at a time.  External writers
# are not coordinated by this lock; identity/containment checks fail closed if
# an external writer replaces a path, and any late partial output is unusable
# until a subsequent exact-set preflight succeeds.  We never roll back paths
# that may belong to another writer.
_VIEW_PREPARE_LOCK = threading.RLock()
_ORIGINAL_DIR_FD_FUNCTIONS = (
    os.open,
    os.stat,
    os.mkdir,
    os.readlink,
    os.symlink,
)


def _safe_basename(value: object) -> bool:
    """Return whether *value* is one portable direct filename component."""

    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    # Reject Windows separators even on POSIX so a manifest is portable and
    # cannot smuggle an object-relative path through a literal backslash.
    if "\\" in value:
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        path.name == value
        and not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
    )


def _resolve_within(path: Path, root: Path, *, label: str) -> Path:
    """Resolve *path* and reject symlink/path traversal outside *root*."""

    try:
        resolved_root = root.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} escapes its allowed root: {path}") from exc
    return resolved_path


def _manifest_root(config: SdmExrInferenceConfig, manifest: DatasetManifest) -> Path:
    raw_config_root = Path(config.data_root)
    raw_manifest_root = Path(manifest.data_root)
    if raw_config_root.is_symlink() or raw_manifest_root.is_symlink():
        raise ValueError("manifest data_root must be a real directory, not a symlink")
    try:
        config_root = raw_config_root.resolve(strict=False)
        manifest_root = raw_manifest_root.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("manifest data_root is invalid") from exc
    if config_root != manifest_root:
        raise ValueError(
            "manifest data_root does not match config data_root: "
            f"{manifest_root} != {config_root}"
        )
    if config_root.is_symlink() or not config_root.is_dir():
        raise ValueError(f"data_root does not exist or is not a directory: {config_root}")
    return config_root


def _object_dir(root: Path, record: ObjectRecord) -> Path:
    if not isinstance(record.name, str) or not _safe_basename(record.name):
        raise ValueError(f"manifest object name is unsafe: {record.name!r}")
    if record.relative_dir != record.name or not _safe_basename(record.relative_dir):
        raise ValueError(f"manifest relative_dir is unsafe for {record.name}")
    candidate = root / record.relative_dir
    if candidate.is_symlink():
        raise ValueError(f"manifest object {record.name} must be a real directory")
    object_dir = _resolve_within(candidate, root, label=f"manifest object {record.name}")
    if object_dir.is_symlink() or not object_dir.is_dir():
        raise ValueError(f"object directory is missing for {record.name}: {object_dir}")
    return object_dir


def _source_path(object_dir: Path, filename: object, *, label: str) -> Path:
    if not _safe_basename(filename):
        raise ValueError(f"manifest {label} is unsafe: {filename!r}")
    candidate = object_dir / str(filename)
    if candidate.is_symlink():
        raise ValueError(f"manifest {label} must be a real file: {candidate}")
    return _resolve_within(candidate, object_dir, label=f"manifest {label}")


def _verify_source(path: Path, expected_digest: object, *, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is missing: {path}")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise ValueError(f"{label} digest is invalid: {path}")
    try:
        actual_digest = sha256_file(path)
    except OSError as exc:
        raise ValueError(f"failed to hash {label}: {path}") from exc
    if actual_digest != expected_digest:
        raise ValueError(
            f"{label} digest mismatch for {path}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    return path


@dataclass(frozen=True)
class _LinkPlan:
    destination: Path
    source: Path
    expected_digest: str


@dataclass(frozen=True)
class _ObjectPlan:
    destination: Path
    links: tuple[_LinkPlan, ...]


def _record_plan(
    config: SdmExrInferenceConfig,
    root: Path,
    record: ObjectRecord,
) -> _ObjectPlan:
    object_dir = _object_dir(root, record)
    selected_images = record.selected_images
    image_digests = record.image_sha256
    if not isinstance(selected_images, (tuple, list)) or not selected_images:
        raise ValueError(f"manifest selected_images must be nonempty for {record.name}")
    if not isinstance(image_digests, (tuple, list)):
        raise ValueError(f"manifest image_sha256 must be a sequence for {record.name}")
    if len(selected_images) != len(image_digests):
        raise ValueError(f"manifest image/hash cardinality mismatch for {record.name}")

    links: list[_LinkPlan] = []
    seen: set[str] = set()
    for filename, digest in zip(selected_images, image_digests):
        if not isinstance(filename, str) or filename in seen:
            raise ValueError(f"manifest selected_images contains an unsafe or duplicate name for {record.name}")
        if filename in config.normal_filenames:
            raise ValueError(
                f"manifest selected image is a recognized normal filename for {record.name}: {filename}"
            )
        seen.add(filename)
        source = _source_path(object_dir, filename, label="selected image")
        _verify_source(source, digest, label=f"selected image for {record.name}")
        links.append(_LinkPlan(Path(record.name) / filename, source, str(digest)))

    # Validate the normal source metadata even though it is deliberately never
    # linked.  This makes tampered loaded manifests fail before any destination
    # mutation and keeps all manifest paths object-relative.
    normal_source = _source_path(object_dir, record.normal_file, label="normal")
    _verify_source(normal_source, record.normal_sha256, label=f"normal for {record.name}")

    if config.mask_policy == "external":
        if not _safe_basename(config.external_mask_filename):
            raise ValueError(
                "external_mask_filename must be an object-relative basename: "
                f"{config.external_mask_filename}"
            )
        # The rich manifest is generated under the same config.  Requiring the
        # filename to agree prevents a stale/tampered manifest from redirecting
        # the configured external source to another basename.
        if record.mask_file != config.external_mask_filename:
            raise ValueError(
                f"manifest external mask filename does not match config for {record.name}"
            )
        mask_source = _source_path(object_dir, config.external_mask_filename, label="external mask")
        _verify_source(mask_source, record.mask_sha256, label=f"external mask for {record.name}")
        links.append(
            _LinkPlan(
                Path(record.name) / "binary_mask.exr",
                mask_source,
                str(record.mask_sha256),
            )
        )

    planned_names = [link.destination.name for link in links]
    if len(set(planned_names)) != len(planned_names):
        raise ValueError(
            f"manifest generated SDM view basenames collide for {record.name}: "
            + ", ".join(sorted(planned_names))
        )
    return _ObjectPlan(Path(record.name), tuple(links))


def _same_source(destination: Path, source: Path) -> bool:
    if not destination.is_symlink():
        return False
    try:
        return destination.resolve(strict=True) == source.resolve(strict=True)
    except (OSError, RuntimeError):
        return False


def _preflight_destination(
    destination: Path,
    source: Path | None,
    *,
    label: str,
    expected_parent: Path | None = None,
) -> None:
    """Check a planned destination without changing it."""

    if expected_parent is not None:
        try:
            actual_parent = destination.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"destination parent is invalid for {label}: {destination}") from exc
        if actual_parent != expected_parent:
            raise ValueError(f"destination parent escaped for {label}: {destination}")
    if not destination.exists() and not destination.is_symlink():
        return
    if source is not None and _same_source(destination, source):
        return
    raise ValueError(f"destination conflict for {label}: {destination}")


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    """Return an lstat identity for a real directory, rejecting symlinks."""

    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a real directory: {path}")
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"failed to inspect {label}: {path}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} is not a real directory: {path}")
    return int(info.st_dev), int(info.st_ino)


def _assert_directory_identity(
    path: Path,
    expected: tuple[int, int],
    *,
    label: str,
) -> Path:
    actual = _directory_identity(path, label=label)
    if actual != expected:
        raise ValueError(f"{label} was replaced during preparation: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved during preparation: {path}") from exc


def _assert_object_containment(
    root: Path,
    root_identity: tuple[int, int],
    object_destination: Path,
    object_identity: tuple[int, int],
) -> tuple[Path, Path]:
    root_resolved = _assert_directory_identity(root, root_identity, label="SDM view root")
    object_resolved = _assert_directory_identity(
        object_destination,
        object_identity,
        label="SDM object destination",
    )
    try:
        object_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"SDM object destination escaped the view root: {object_destination}"
        ) from exc
    return root_resolved, object_resolved


def _revalidate_link_source(link: _LinkPlan, data_root: Path) -> None:
    """Recheck source identity/containment/digest immediately before linking."""

    try:
        resolved_root = data_root.resolve(strict=True)
        resolved_source = link.source.resolve(strict=True)
        resolved_source.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"source escaped or disappeared during preparation: {link.source}") from exc
    if resolved_source != link.source or not link.source.is_file():
        raise ValueError(f"source changed during preparation: {link.source}")
    _verify_source(link.source, link.expected_digest, label="source")


def _descriptor_flags() -> int:
    """Return secure directory-open flags or fail clearly on weak platforms."""

    required_constants = ("O_DIRECTORY", "O_NOFOLLOW")
    missing_constants = [name for name in required_constants if not hasattr(os, name)]
    supported = getattr(os, "supports_dir_fd", set())
    required_functions = _ORIGINAL_DIR_FD_FUNCTIONS
    missing_functions = [
        getattr(function, "__name__", repr(function))
        for function in required_functions
        if function not in supported
    ]
    if missing_constants or missing_functions:
        details = []
        if missing_constants:
            details.append(f"constants={missing_constants}")
        if missing_functions:
            details.append(f"dir_fd={missing_functions}")
        raise RuntimeError(
            "secure SDM view preparation requires O_DIRECTORY/O_NOFOLLOW and "
            f"descriptor-relative operations ({'; '.join(details)})"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _fd_identity(fd: int, *, label: str) -> tuple[int, int]:
    try:
        info = os.fstat(fd)
    except OSError as exc:
        raise ValueError(f"failed to inspect {label} descriptor") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} descriptor is not a directory")
    return int(info.st_dev), int(info.st_ino)


def _open_root_fd(
    view_root: Path,
    expected_identity: tuple[int, int] | None,
    flags: int,
) -> int:
    """Open the final view-root component without following a symlink."""

    parent_fd: int | None = None
    root_fd: int | None = None
    keep_fd = False
    try:
        parent = view_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        parent_fd = os.open(str(parent), flags)
        name = view_root.name
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            existed = True
        except FileNotFoundError:
            existed = False
            if expected_identity is not None:
                raise ValueError(f"SDM view root disappeared during preparation: {view_root}")
            try:
                os.mkdir(name, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise ValueError(f"SDM view root appeared during preparation: {view_root}") from exc
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"SDM view root is not a real directory: {view_root}")
        before_identity = (int(before.st_dev), int(before.st_ino))
        if expected_identity is None and existed:
            raise ValueError(f"SDM view root appeared during preparation: {view_root}")
        if expected_identity is not None and before_identity != expected_identity:
            raise ValueError(f"SDM view root was replaced during preparation: {view_root}")

        root_fd = os.open(name, flags, dir_fd=parent_fd)
        actual_identity = _fd_identity(root_fd, label="SDM view root")
        if actual_identity != before_identity or (
            expected_identity is not None and actual_identity != expected_identity
        ):
            raise ValueError(f"SDM view root was replaced during preparation: {view_root}")
        keep_fd = True
        return root_fd
    except OSError as exc:
        raise ValueError(f"failed to securely open SDM view root: {view_root}") from exc
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if root_fd is not None and not keep_fd:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _open_existing_root_fd(view_root: Path, flags: int) -> tuple[int, tuple[int, int]]:
    """Open an existing view root without creating or following a path."""

    parent_fd: int | None = None
    root_fd: int | None = None
    keep_fd = False
    try:
        root_identity = _directory_identity(view_root, label="SDM view root")
        parent_fd = os.open(str(view_root.parent), flags)
        before = os.stat(view_root.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"SDM view root is not a real directory: {view_root}")
        before_identity = (int(before.st_dev), int(before.st_ino))
        if before_identity != root_identity:
            raise ValueError(f"SDM view root was replaced during validation: {view_root}")
        root_fd = os.open(view_root.name, flags, dir_fd=parent_fd)
        actual = _fd_identity(root_fd, label="SDM view root")
        if actual != root_identity:
            raise ValueError(f"SDM view root was replaced during validation: {view_root}")
        keep_fd = True
        return root_fd, root_identity
    except OSError as exc:
        raise ValueError(f"failed to securely open SDM view root: {view_root}") from exc
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if root_fd is not None and not keep_fd:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _open_object_fd(
    root_fd: int,
    object_path: Path,
    object_name: str,
    expected_identity: tuple[int, int] | None,
    flags: int,
) -> tuple[int, tuple[int, int], bool]:
    """Open/create an object child relative to the trusted root FD."""

    object_fd: int | None = None
    keep_fd = False
    try:
        try:
            before = os.stat(object_name, dir_fd=root_fd, follow_symlinks=False)
            existed = True
        except FileNotFoundError:
            existed = False
            if expected_identity is not None:
                raise ValueError(f"SDM object destination disappeared during preparation: {object_path}")
            try:
                os.mkdir(object_name, dir_fd=root_fd)
            except FileExistsError as exc:
                raise ValueError(
                    f"SDM object destination appeared during preparation: {object_path}"
                ) from exc
            before = os.stat(object_name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"SDM object destination is not a real directory: {object_path}")
        before_identity = (int(before.st_dev), int(before.st_ino))
        if expected_identity is None and existed:
            raise ValueError(f"SDM object destination appeared during preparation: {object_path}")
        if expected_identity is not None and before_identity != expected_identity:
            raise ValueError(f"SDM object destination was replaced during preparation: {object_path}")

        object_fd = os.open(object_name, flags, dir_fd=root_fd)
        actual_identity = _fd_identity(object_fd, label="SDM object destination")
        if actual_identity != before_identity or (
            expected_identity is not None and actual_identity != expected_identity
        ):
            raise ValueError(f"SDM object destination was replaced during preparation: {object_path}")
        keep_fd = True
        return object_fd, actual_identity, existed
    except OSError as exc:
        raise ValueError(f"failed to securely open SDM object destination: {object_path}") from exc
    finally:
        if object_fd is not None and not keep_fd:
            try:
                os.close(object_fd)
            except OSError:
                pass


def _fd_names(fd: int, *, label: str) -> set[str]:
    try:
        return set(os.listdir(fd))
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"descriptor-relative directory inspection is unavailable for {label}") from exc


def _fd_same_source(
    fd: int,
    object_path: Path,
    basename: str,
    source: Path,
) -> bool:
    try:
        info = os.stat(basename, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if not stat.S_ISLNK(info.st_mode):
        return False
    try:
        target = os.readlink(basename, dir_fd=fd)
        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = object_path / target_path
        return target_path.resolve(strict=True) == source.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return False


def _fd_preflight_destination(
    fd: int,
    object_path: Path,
    basename: str,
    source: Path,
) -> bool:
    """Validate one basename via the stable object descriptor.

    Returns whether a non-broken matching symlink already exists.
    """

    try:
        info = os.stat(basename, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ValueError(f"failed to inspect destination {object_path / basename}") from exc
    if not stat.S_ISLNK(info.st_mode) or not _fd_same_source(
        fd, object_path, basename, source
    ):
        raise ValueError(f"destination conflict for {object_path / basename}")
    return True


def _fd_preflight_object(
    fd: int,
    object_path: Path,
    plan: _ObjectPlan,
) -> None:
    planned_names = {link.destination.name for link in plan.links}
    existing_names = _fd_names(fd, label=str(object_path))
    if existing_names != planned_names:
        stale = sorted(existing_names - planned_names)
        missing = sorted(planned_names - existing_names)
        details = []
        if stale:
            details.append(f"stale={stale}")
        if missing:
            details.append(f"missing={missing}")
        raise ValueError(
            f"destination conflict: SDM object destination must contain exactly planned links "
            f"for {object_path}: {', '.join(details)}"
        )
    for link in plan.links:
        _fd_preflight_destination(
            fd,
            object_path,
            link.destination.name,
            link.source,
        )


def _fd_preflight_root(
    root_fd: int,
    root_path: Path,
    plans: tuple[_ObjectPlan, ...],
) -> set[str]:
    expected_objects = {plan.destination.name for plan in plans}
    existing_names = _fd_names(root_fd, label=str(root_path))
    stale = existing_names - expected_objects
    if stale:
        raise ValueError(
            "stale or unexpected SDM view root entries: "
            + ", ".join(sorted(stale))
        )
    for name in existing_names:
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"failed to inspect SDM object destination: {root_path / name}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"SDM view object destination is not a real directory: {root_path / name}")
    return existing_names


def validate_sdm_view(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
) -> dict[str, object]:
    """Read-only attest the exact descriptor-relative GT-hidden SDM view.

    The returned mapping is safe to persist in a request/fingerprint.  Every
    object and basename is checked against the current manifest plan; every
    destination must be a non-broken symlink to the exact verified source
    file, and source bytes are re-hashed during the attestation.  No directory
    or link is created, removed, or replaced by this function.
    """

    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not _safe_basename(config.external_mask_filename):
        raise ValueError(
            "external_mask_filename must be an object-relative basename: "
            f"{config.external_mask_filename}"
        )
    if any(not _safe_basename(filename) for filename in config.normal_filenames):
        raise ValueError("normal_filenames must contain object-relative basenames")

    with _VIEW_PREPARE_LOCK:
        root = _manifest_root(config, manifest)
        plans = tuple(_record_plan(config, root, record) for record in manifest.objects)
        if not plans:
            raise ValueError("manifest contains no objects")
        object_names = [plan.destination.name for plan in plans]
        if len(set(object_names)) != len(object_names):
            raise ValueError("manifest object names collide in SDM view")

        flags = _descriptor_flags()
        view_root = Path(config.sdm_view_dir)
        root_fd, root_identity = _open_existing_root_fd(view_root, flags)
        object_fds: list[tuple[int, tuple[int, int], Path]] = []
        try:
            expected_objects = set(object_names)
            existing_objects = _fd_names(root_fd, label=str(view_root))
            if existing_objects != expected_objects:
                stale = sorted(existing_objects - expected_objects)
                missing = sorted(expected_objects - existing_objects)
                details = []
                if stale:
                    details.append(f"stale={stale}")
                if missing:
                    details.append(f"missing={missing}")
                raise ValueError(
                    "SDM view object set mismatch: " + ", ".join(details)
                )

            object_records: list[dict[str, object]] = []
            object_identities: dict[str, list[int]] = {}
            for plan in plans:
                object_name = plan.destination.name
                object_path = view_root / plan.destination
                try:
                    info = os.stat(object_name, dir_fd=root_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError(f"failed to inspect SDM object directory: {object_path}") from exc
                if not stat.S_ISDIR(info.st_mode):
                    raise ValueError(f"SDM object destination is not a real directory: {object_path}")
                expected_identity = (int(info.st_dev), int(info.st_ino))
                object_fd, object_identity, _ = _open_object_fd(
                    root_fd,
                    object_path,
                    object_name,
                    expected_identity,
                    flags,
                )
                object_fds.append((object_fd, object_identity, object_path))
                planned_names = {link.destination.name for link in plan.links}
                existing_names = _fd_names(object_fd, label=str(object_path))
                if existing_names != planned_names:
                    stale = sorted(existing_names - planned_names)
                    missing = sorted(planned_names - existing_names)
                    details = []
                    if stale:
                        details.append(f"stale={stale}")
                    if missing:
                        details.append(f"missing={missing}")
                    raise ValueError(
                        f"SDM view basename set mismatch for {object_name}: "
                        + ", ".join(details)
                    )
                forbidden = set(config.normal_filenames)
                if config.mask_policy == "full":
                    forbidden.add("binary_mask.exr")
                if existing_names & forbidden:
                    raise ValueError(
                        f"SDM view exposes a forbidden ground-truth basename for {object_name}"
                    )

                entries: list[dict[str, object]] = []
                for link in plan.links:
                    basename = link.destination.name
                    try:
                        entry_info = os.stat(
                            basename,
                            dir_fd=object_fd,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise ValueError(
                            f"failed to inspect SDM view link: {object_path / basename}"
                        ) from exc
                    if not stat.S_ISLNK(entry_info.st_mode):
                        raise ValueError(
                            f"SDM view destination must be a symlink: {object_path / basename}"
                        )
                    try:
                        target = os.readlink(basename, dir_fd=object_fd)
                        target_path = Path(target)
                        if not target_path.is_absolute():
                            target_path = object_path / target_path
                        target_resolved = target_path.resolve(strict=True)
                        source_resolved = link.source.resolve(strict=True)
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise ValueError(
                            f"SDM view link target is invalid: {object_path / basename}"
                        ) from exc
                    if target_resolved != source_resolved:
                        raise ValueError(
                            f"SDM view link target mismatch for {object_path / basename}"
                        )
                    _verify_source(link.source, link.expected_digest, label="SDM view source")
                    entries.append(
                        {
                            "basename": basename,
                            "identity": [int(entry_info.st_dev), int(entry_info.st_ino)],
                            "target": str(target),
                            "source_path": str(source_resolved),
                            "source_sha256": str(link.expected_digest),
                        }
                    )
                identity_json = [int(object_identity[0]), int(object_identity[1])]
                object_identities[object_name] = identity_json
                object_records.append(
                    {
                        "object_name": object_name,
                        "identity": identity_json,
                        "entries": entries,
                    }
                )

            # Rewalk the exact descriptor snapshots after all source hashing.
            # A first pass followed by only a pathname identity check is not a
            # coherent attestation: an attacker can replace a link or add an
            # entry between the hash and digest publication while restoring the
            # directory pathname.  Keep this read-only and fail closed if any
            # root/object set, symlink identity, raw target, resolved source,
            # or source digest changed.  This does not claim to prevent an
            # external writer from racing after the final rewalk; callers must
            # treat the attestation as a point-in-time snapshot.
            if _fd_names(root_fd, label=str(view_root)) != expected_objects:
                raise ValueError("SDM view root entry set changed during validation")
            for plan, (object_fd, _, object_path), object_record in zip(
                plans, object_fds, object_records
            ):
                planned_names = {link.destination.name for link in plan.links}
                if _fd_names(object_fd, label=str(object_path)) != planned_names:
                    raise ValueError(
                        f"SDM view basename set changed during validation for {object_path.name}"
                    )
                entries = object_record["entries"]
                if not isinstance(entries, list) or len(entries) != len(plan.links):
                    raise ValueError(f"SDM view attestation entries changed for {object_path.name}")
                for link, first_entry in zip(plan.links, entries):
                    basename = link.destination.name
                    try:
                        entry_info = os.stat(
                            basename,
                            dir_fd=object_fd,
                            follow_symlinks=False,
                        )
                        if not stat.S_ISLNK(entry_info.st_mode):
                            raise ValueError(
                                f"SDM view destination changed type: {object_path / basename}"
                            )
                        target = os.readlink(basename, dir_fd=object_fd)
                        target_path = Path(target)
                        if not target_path.is_absolute():
                            target_path = object_path / target_path
                        target_resolved = target_path.resolve(strict=True)
                        source_resolved = link.source.resolve(strict=True)
                    except (OSError, RuntimeError, ValueError) as exc:
                        if isinstance(exc, ValueError) and str(exc).startswith(
                            "SDM view destination changed type"
                        ):
                            raise
                        raise ValueError(
                            f"SDM view link changed during validation: {object_path / basename}"
                        ) from exc
                    if target_resolved != source_resolved:
                        raise ValueError(
                            f"SDM view link target changed during validation: {object_path / basename}"
                        )
                    _verify_source(link.source, link.expected_digest, label="SDM view source")
                    second_entry = {
                        "basename": basename,
                        "identity": [int(entry_info.st_dev), int(entry_info.st_ino)],
                        "target": str(target),
                        "source_path": str(source_resolved),
                        "source_sha256": str(link.expected_digest),
                    }
                    if second_entry != first_entry:
                        raise ValueError(
                            f"SDM view link identity or target changed during validation: "
                            f"{object_path / basename}"
                        )

            # A late pathname replacement must not turn a safe detached
            # descriptor walk into a successful attestation for another tree.
            if _directory_identity(view_root, label="SDM view root") != root_identity:
                raise ValueError(f"SDM view root was replaced during validation: {view_root}")
            for _, object_identity, object_path in object_fds:
                if _directory_identity(object_path, label="SDM object destination") != object_identity:
                    raise ValueError(
                        f"SDM object destination was replaced during validation: {object_path}"
                    )

            canonical = {
                "root_identity": [int(root_identity[0]), int(root_identity[1])],
                "objects": object_records,
            }
            digest = sha256_bytes(canonical_json_bytes(canonical))
            root_identity_json = [int(root_identity[0]), int(root_identity[1])]
            return {
                "view_path": str(view_root.resolve(strict=True)),
                "view_tree_sha256": digest,
                "view_tree_digest": digest,
                "root_identity": root_identity_json,
                "object_identities": object_identities,
                "objects": object_records,
            }
        finally:
            for object_fd, _, _ in object_fds:
                try:
                    os.close(object_fd)
                except OSError:
                    pass
            try:
                os.close(root_fd)
            except OSError:
                pass


def _preflight_plans(config: SdmExrInferenceConfig, plans: Iterable[_ObjectPlan]) -> None:
    root = Path(config.sdm_view_dir)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError(f"SDM view destination is not a directory: {root}")

    plans = tuple(plans)
    expected_objects = {plan.destination.name for plan in plans}
    if root.is_dir():
        try:
            existing_entries = tuple(root.iterdir())
        except OSError as exc:
            raise ValueError(f"failed to inspect SDM view destination: {root}") from exc
        stale_entries = tuple(
            entry for entry in existing_entries if entry.name not in expected_objects
        )
        if stale_entries:
            names = ", ".join(sorted(entry.name for entry in stale_entries))
            raise ValueError(f"stale or unexpected SDM view root entries: {names}")
        for entry in existing_entries:
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError(f"SDM view object destination is not a real directory: {entry}")

    for plan in plans:
        object_destination = root / plan.destination
        if object_destination.exists() and (
            object_destination.is_symlink() or not object_destination.is_dir()
        ):
            raise ValueError(f"destination conflict for object directory: {object_destination}")

        planned_names = {link.destination.name for link in plan.links}
        if len(planned_names) != len(plan.links):
            raise ValueError(f"destination link names collide for {object_destination}")
        if object_destination.is_dir():
            try:
                existing_names = {entry.name for entry in object_destination.iterdir()}
            except OSError as exc:
                raise ValueError(
                    f"failed to inspect SDM object destination: {object_destination}"
                ) from exc
            if existing_names != planned_names:
                stale = sorted(existing_names - planned_names)
                missing = sorted(planned_names - existing_names)
                details = []
                if stale:
                    details.append(f"stale={stale}")
                if missing:
                    details.append(f"missing={missing}")
                raise ValueError(
                    f"destination conflict: SDM object destination must contain exactly planned links "
                    f"for {object_destination}: {', '.join(details)}"
                )
        for link in plan.links:
            _preflight_destination(
                root / link.destination,
                link.source,
                label=link.destination.name,
                expected_parent=(
                    object_destination.resolve(strict=True)
                    if object_destination.is_dir()
                    else None
                ),
            )


def prepare_sdm_view(
    config: SdmExrInferenceConfig,
    manifest: DatasetManifest,
) -> Path:
    """Prepare and return the GT-hidden SDM input directory.

    All manifest source paths and SHA-256 digests are checked before any
    destination directory or symlink is created.  Existing matching symlinks
    are retained, while every other destination collision is fatal and
    non-destructive.
    """

    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if not _safe_basename(config.external_mask_filename):
        raise ValueError(
            "external_mask_filename must be an object-relative basename: "
            f"{config.external_mask_filename}"
        )
    if any(not _safe_basename(filename) for filename in config.normal_filenames):
        raise ValueError("normal_filenames must contain object-relative basenames")

    with _VIEW_PREPARE_LOCK:
        root = _manifest_root(config, manifest)
        plans = tuple(_record_plan(config, root, record) for record in manifest.objects)
        if not plans:
            raise ValueError("manifest contains no objects")
        _preflight_plans(config, plans)

        descriptor_flags = _descriptor_flags()
        view_root = Path(config.sdm_view_dir)
        # Keep the legacy pathname mkdir only for the final root component;
        # it is immediately pinned by its inode and reopened descriptor-
        # relative with O_NOFOLLOW.  Object creation below uses root_fd.
        if view_root.exists():
            root_identity = _directory_identity(view_root, label="SDM view root")
        else:
            view_root.mkdir(parents=True, exist_ok=True)
            root_identity = _directory_identity(view_root, label="SDM view root")

        root_fd = _open_root_fd(view_root, root_identity, descriptor_flags)
        object_fds: dict[str, tuple[int, tuple[int, int], Path]] = {}
        try:
            existing_root_names = _fd_preflight_root(root_fd, view_root, plans)
            existing_object_identities: dict[str, tuple[int, int] | None] = {}
            for plan in plans:
                object_destination = view_root / plan.destination
                if plan.destination.name in existing_root_names:
                    existing_object_identities[plan.destination.name] = _directory_identity(
                        object_destination,
                        label="SDM object destination",
                    )
                else:
                    existing_object_identities[plan.destination.name] = None

            for plan in plans:
                object_name = plan.destination.name
                object_destination = view_root / plan.destination
                object_fd, object_identity, _ = _open_object_fd(
                    root_fd,
                    object_destination,
                    object_name,
                    existing_object_identities[object_name],
                    descriptor_flags,
                )
                object_fds[object_name] = (object_fd, object_identity, object_destination)
                if existing_object_identities[object_name] is not None:
                    _fd_preflight_object(object_fd, object_destination, plan)

            for plan in plans:
                object_name = plan.destination.name
                object_fd, object_identity, object_destination = object_fds[object_name]
                for link in plan.links:
                    # Path identity checks make replacement visible to the
                    # caller, while the descriptor makes the write safe even
                    # if replacement occurs immediately afterward.
                    _assert_object_containment(
                        view_root,
                        root_identity,
                        object_destination,
                        object_identity,
                    )
                    _revalidate_link_source(link, root)
                    matching = _fd_preflight_destination(
                        object_fd,
                        object_destination,
                        link.destination.name,
                        link.source,
                    )
                    if matching:
                        continue
                    try:
                        os.symlink(
                            str(link.source),
                            link.destination.name,
                            dir_fd=object_fd,
                        )
                    except FileExistsError as exc:
                        raise ValueError(
                            f"destination conflict for {object_destination / link.destination.name}: "
                            "path appeared during preparation"
                        ) from exc
                    # If an external writer replaced either pathname during
                    # the syscall, fail closed.  The descriptor-relative
                    # syscall cannot redirect to that replacement; a possible
                    # link in an unlinked directory is intentionally not
                    # rolled back.
                    _assert_object_containment(
                        view_root,
                        root_identity,
                        object_destination,
                        object_identity,
                    )
                    _fd_preflight_destination(
                        object_fd,
                        object_destination,
                        link.destination.name,
                        link.source,
                    )

            for plan in plans:
                object_fd, _, object_destination = object_fds[plan.destination.name]
                _fd_preflight_object(object_fd, object_destination, plan)
            _fd_preflight_root(root_fd, view_root, plans)
            return view_root
        finally:
            for object_fd, _, _ in object_fds.values():
                try:
                    os.close(object_fd)
                except OSError:
                    pass
            try:
                os.close(root_fd)
            except OSError:
                pass


def build_sdm_command(
    config: SdmExrInferenceConfig,
    sdm_repo: Path,
    checkpoint: Path,
    python_executable: Path,
    *,
    output_dir: Path | None = None,
) -> list[str]:
    """Return the SDM inference argv without starting a subprocess."""

    if not isinstance(config, SdmExrInferenceConfig):
        raise TypeError("config must be an SdmExrInferenceConfig")
    repo = Path(sdm_repo).resolve(strict=False)
    checkpoint_path = Path(checkpoint).resolve(strict=False)
    python_path = Path(python_executable).resolve(strict=False)
    requested_output = (
        config.sdm_output_dir if output_dir is None else Path(output_dir)
    ).resolve(strict=False)
    view_path = Path(config.sdm_view_dir).resolve(strict=False)
    selection_path = Path(config.effective_selection_manifest_path).resolve(strict=False)
    return [
        str(python_path),
        str(repo / "main.py"),
        "infer",
        "--config",
        str(repo / "configs" / "baseline_optimized_infer.yaml"),
        "--checkpoint",
        str(checkpoint_path),
        "--test-dir",
        str(view_path),
        "--output-dir",
        str(requested_output),
        "--light-selection",
        "manifest",
        "--selection-manifest",
        str(selection_path),
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
    ]


__all__ = ["prepare_sdm_view", "validate_sdm_view", "build_sdm_command"]
