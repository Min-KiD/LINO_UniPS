"""Strict filesystem and hashing helpers for comparison provenance."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_ORIGINAL_DIR_FD_FUNCTIONS = (os.open, os.mkdir, os.stat)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize *payload* deterministically for hashing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def config_runtime_snapshot(config: Any) -> dict[str, Any]:
    """Return all strict inference-config fields in canonical JSON-safe form."""

    values = vars(config)
    if not isinstance(values, Mapping):
        raise TypeError("config must expose dataclass fields")
    return {str(key): _json_safe(values[key]) for key in sorted(values)}


def config_runtime_fingerprint(config: Any) -> str:
    return sha256_json(config_runtime_snapshot(config))


def lino_preprocessing_snapshot(config: Any) -> dict[str, Any]:
    """Return the exact preprocessing/model knobs paired to a LINO run."""

    return {
        "mask_margin": int(config.mask_margin),
        "max_image_resolution": int(config.max_image_resolution),
        "pixel_samples": int(config.pixel_samples),
        "precision": str(config.precision),
        "device": str(config.device),
        "checkpoint": str(Path(config.checkpoint)),
    }


def file_identity(path: str | Path, *, label: str = "file") -> dict[str, int]:
    """Return a regular, non-symlink file's device/inode identity."""

    source = Path(path)
    try:
        info = source.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable: {source}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {source}")
    return {"dev": int(info.st_dev), "ino": int(info.st_ino)}


def _require_real_directory(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory, not a symlink: {path}")
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} cannot be resolved: {path}") from exc


def ensure_real_directory(path: str | Path, *, label: str) -> Path:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    return _require_real_directory(destination, label=label)


def _secure_directory_flags() -> int:
    """Return descriptor flags required for no-follow directory pinning."""

    missing = [
        name
        for name in ("O_DIRECTORY", "O_NOFOLLOW")
        if not hasattr(os, name)
    ]
    supported = getattr(os, "supports_dir_fd", set())
    if any(function not in supported for function in _ORIGINAL_DIR_FD_FUNCTIONS):
        missing.append("descriptor-relative os.open/os.mkdir/os.stat")
    if missing:
        raise RuntimeError(
            "secure provenance publication requires O_DIRECTORY/O_NOFOLLOW and "
            f"descriptor-relative operations: {', '.join(missing)}"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _directory_identity(path: str | Path, *, label: str) -> tuple[int, int]:
    source = Path(path)
    try:
        info = source.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is missing or unreadable: {source}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a real directory, not a symlink: {source}")
    return int(info.st_dev), int(info.st_ino)


def directory_identity(path: str | Path, *, label: str = "directory") -> dict[str, int]:
    """Return a JSON-safe device/inode identity for a real directory."""

    device, inode = _directory_identity(path, label=label)
    return {"dev": device, "ino": inode}


def _identity_tuple(value: Any, *, label: str) -> tuple[int, int]:
    if isinstance(value, Mapping):
        device, inode = value.get("dev"), value.get("ino")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        device, inode = value
    else:
        raise ValueError(f"{label} identity is invalid")
    if type(device) is not int or type(inode) is not int or device < 0 or inode < 0:
        raise ValueError(f"{label} identity is invalid")
    return int(device), int(inode)


def same_directory_identity(
    path: str | Path,
    expected: Any,
    *,
    label: str = "directory",
) -> bool:
    """Compare a real directory's current device/inode identity."""

    try:
        return _directory_identity(path, label=label) == _identity_tuple(expected, label=label)
    except (OSError, ValueError):
        return False


def open_or_create_directory(
    path: str | Path,
    *,
    label: str,
) -> tuple[int, dict[str, int], Path]:
    """Walk/create an absolute directory tree without following any symlink."""

    absolute = Path(os.path.abspath(str(path)))
    parts = absolute.parts
    if not parts or parts[0] != os.sep:
        raise ValueError(f"{label} must resolve to an absolute path: {path}")
    flags = _secure_directory_flags()
    descriptor = os.open(os.sep, flags)
    child_fd: int | None = None
    try:
        for component in parts[1:]:
            try:
                info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError as exc:
                    raise ValueError(
                        f"{label} appeared during secure creation: {absolute}"
                    ) from exc
                info = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"{label} must contain only real directories: {absolute}")
            expected = (int(info.st_dev), int(info.st_ino))
            child_fd = os.open(component, flags, dir_fd=descriptor)
            child_info = os.fstat(child_fd)
            actual = (int(child_info.st_dev), int(child_info.st_ino))
            if actual != expected:
                raise ValueError(f"{label} was replaced during secure creation: {absolute}")
            os.close(descriptor)
            descriptor = child_fd
            child_fd = None
        info = os.fstat(descriptor)
        identity = {"dev": int(info.st_dev), "ino": int(info.st_ino)}
        return descriptor, identity, absolute
    except Exception:
        if child_fd is not None:
            try:
                os.close(child_fd)
            except OSError:
                pass
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def assert_directory_path_identity(
    path: str | Path,
    expected: Any,
    *,
    label: str,
) -> None:
    """Require *path* still to name the directory represented by *expected*."""

    source = Path(path)
    expected_tuple = _identity_tuple(expected, label=label)
    try:
        actual = _directory_identity(source, label=label)
    except ValueError as exc:
        raise ValueError(f"{label} was replaced during publication: {source}") from exc
    if actual != expected_tuple:
        raise ValueError(f"{label} was replaced during publication: {source}")


def _safe_basename(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
        or "\\" in value
    ):
        raise ValueError(f"{label} name must be one safe path component")
    return value


def _new_file_at_fd(directory_fd: int, basename: str, *, label: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _ in range(16):
        candidate = f".{basename}.{secrets.token_hex(12)}.tmp"
        try:
            return os.open(candidate, flags, 0o600, dir_fd=directory_fd), candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"failed to allocate temporary {label} file: {basename}")


def _backup_file_at_fd(
    directory_fd: int,
    basename: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> str:
    for _ in range(16):
        candidate = f".{basename}.{secrets.token_hex(12)}.bak"
        try:
            os.link(
                basename,
                candidate,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            continue
        backup = os.stat(candidate, dir_fd=directory_fd, follow_symlinks=False)
        if (int(backup.st_dev), int(backup.st_ino)) != expected_identity:
            os.unlink(candidate, dir_fd=directory_fd)
            raise ValueError(f"{label} was replaced before publication: {basename}")
        return candidate
    raise FileExistsError(f"failed to preserve previous {label}: {basename}")


def atomic_replace_bytes_at_fd(
    directory_fd: int,
    basename: str,
    payload: bytes,
    *,
    expected_directory_identity: Any,
    label: str,
    directory_path: str | Path | None = None,
) -> tuple[int, int]:
    """Atomically replace one owned file without opening/truncating its inode."""

    name = _safe_basename(basename, label=label)
    if not isinstance(payload, bytes):
        raise TypeError("published payload must be bytes")
    expected = _identity_tuple(expected_directory_identity, label=f"{label} directory")
    temporary_fd: int | None = None
    temporary_name: str | None = None
    backup_name: str | None = None
    temporary_identity: tuple[int, int] | None = None
    committed = False
    previous_existed = False
    try:
        directory_info = os.fstat(directory_fd)
        if (int(directory_info.st_dev), int(directory_info.st_ino)) != expected:
            raise ValueError(f"{label} directory was replaced before publication")
        if directory_path is not None:
            assert_directory_path_identity(
                directory_path,
                expected_directory_identity,
                label=f"{label} directory",
            )

        try:
            previous = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            previous = None
        if previous is not None and not stat.S_ISREG(previous.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {name}")
        previous_identity = (
            (int(previous.st_dev), int(previous.st_ino)) if previous is not None else None
        )
        previous_existed = previous_identity is not None

        temporary_fd, temporary_name = _new_file_at_fd(directory_fd, name, label=label)
        offset = 0
        while offset < len(payload):
            written = os.write(temporary_fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while publishing artifact")
            offset += written
        os.fsync(temporary_fd)
        temporary_info = os.fstat(temporary_fd)
        temporary_identity = (
            int(temporary_info.st_dev),
            int(temporary_info.st_ino),
        )
        os.close(temporary_fd)
        temporary_fd = None

        directory_before = os.fstat(directory_fd)
        if (int(directory_before.st_dev), int(directory_before.st_ino)) != expected:
            raise ValueError(f"{label} directory was replaced before publication")
        if directory_path is not None:
            assert_directory_path_identity(
                directory_path,
                expected_directory_identity,
                label=f"{label} directory",
            )
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {name}")
        current_identity = (
            (int(current.st_dev), int(current.st_ino)) if current is not None else None
        )
        if current_identity != previous_identity:
            raise ValueError(f"{label} was replaced before publication: {name}")
        if previous_identity is not None:
            backup_name = _backup_file_at_fd(
                directory_fd,
                name,
                previous_identity,
                label=label,
            )

        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        committed = True
        temporary_name = None

        published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        published_identity = (int(published.st_dev), int(published.st_ino))
        if not stat.S_ISREG(published.st_mode) or published_identity != temporary_identity:
            raise ValueError(f"{label} was replaced during publication: {name}")

        directory_after = os.fstat(directory_fd)
        if (int(directory_after.st_dev), int(directory_after.st_ino)) != expected:
            raise ValueError(f"{label} directory was replaced during publication")
        if directory_path is not None:
            assert_directory_path_identity(
                directory_path,
                expected_directory_identity,
                label=f"{label} directory",
            )
        os.fsync(directory_fd)
        if backup_name is not None:
            os.unlink(backup_name, dir_fd=directory_fd)
            backup_name = None
        return published_identity
    except Exception as exc:
        if committed:
            try:
                if backup_name is not None:
                    os.replace(
                        backup_name,
                        name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                    )
                    backup_name = None
                elif not previous_existed:
                    os.unlink(name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except OSError:
                pass
        if isinstance(exc, (TypeError, ValueError)):
            raise
        raise ValueError(f"failed to securely publish {label}: {name}") from exc
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        cleanup_names = [temporary_name]
        if not committed:
            cleanup_names.append(backup_name)
        for candidate in cleanup_names:
            if candidate is not None:
                try:
                    os.unlink(candidate, dir_fd=directory_fd)
                except OSError:
                    pass


def read_regular_bytes_at_fd(
    directory_fd: int,
    basename: str,
    *,
    expected_directory_identity: Any,
    label: str,
    directory_path: str | Path | None = None,
) -> bytes:
    """Read one immutable regular-file snapshot through a pinned directory."""

    name = _safe_basename(basename, label=label)
    expected = _identity_tuple(expected_directory_identity, label=f"{label} directory")
    descriptor: int | None = None
    try:
        directory_before = os.fstat(directory_fd)
        if (int(directory_before.st_dev), int(directory_before.st_ino)) != expected:
            raise ValueError(f"{label} directory was replaced before reading")
        if directory_path is not None:
            assert_directory_path_identity(
                directory_path,
                expected_directory_identity,
                label=f"{label} directory",
            )
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {name}")
        expected_file = (int(before.st_dev), int(before.st_ino))
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            int(opened.st_dev),
            int(opened.st_ino),
        ) != expected_file:
            raise ValueError(f"{label} was replaced while opening: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (int(after.st_dev), int(after.st_ino)) != expected_file:
            raise ValueError(f"{label} was replaced while reading: {name}")
        directory_after = os.fstat(directory_fd)
        if (int(directory_after.st_dev), int(directory_after.st_ino)) != expected:
            raise ValueError(f"{label} directory was replaced while reading")
        if directory_path is not None:
            assert_directory_path_identity(
                directory_path,
                expected_directory_identity,
                label=f"{label} directory",
            )
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"failed to securely read {label}: {name}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def create_fresh_child_directory(
    base: str | Path,
    child_name: str,
    *,
    label: str,
) -> Path:
    """Atomically create one new real child directory and never reuse it."""

    if (
        not isinstance(child_name, str)
        or not child_name
        or Path(child_name).name != child_name
        or child_name in {".", ".."}
        or "\\" in child_name
    ):
        raise ValueError(f"{label} name must be one safe path component")
    root = ensure_real_directory(base, label=f"{label} root")
    candidate = root / child_name
    flags = _secure_directory_flags()
    root_fd: int | None = None
    child_fd: int | None = None
    try:
        root_identity = _directory_identity(root, label=f"{label} root")
        root_fd = os.open(str(root), flags)
        root_fd_info = os.fstat(root_fd)
        if (int(root_fd_info.st_dev), int(root_fd_info.st_ino)) != root_identity:
            raise ValueError(f"{label} root was replaced during creation: {root}")
        try:
            os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(
                f"{label} already exists; refusing stale reuse: {candidate}"
            )
        try:
            os.mkdir(child_name, mode=0o700, dir_fd=root_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{label} already exists; refusing stale reuse: {candidate}"
            ) from exc
        before = os.stat(child_name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{label} is not a real directory: {candidate}")
        child_identity = (int(before.st_dev), int(before.st_ino))
        child_fd = os.open(child_name, flags, dir_fd=root_fd)
        child_info = os.fstat(child_fd)
        if (int(child_info.st_dev), int(child_info.st_ino)) != child_identity:
            raise ValueError(f"{label} was replaced during creation: {candidate}")
        root_after = _directory_identity(root, label=f"{label} root")
        if root_after != root_identity:
            raise ValueError(f"{label} root was replaced during creation: {root}")
        candidate_after = _directory_identity(candidate, label=label)
        if candidate_after != child_identity:
            raise ValueError(f"{label} was replaced during creation: {candidate}")
        return candidate
    except FileExistsError:
        raise
    except OSError as exc:
        raise ValueError(f"failed to securely create {label}: {candidate}") from exc
    finally:
        if child_fd is not None:
            try:
                os.close(child_fd)
            except OSError:
                pass
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def atomic_create_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Publish JSON once without replacing any existing file or symlink."""

    destination = Path(path)
    parent = ensure_real_directory(destination.parent, label="provenance directory")
    flags = _secure_directory_flags()
    parent_fd: int | None = None
    temporary_name: str | None = None
    temporary_fd: int | None = None
    parent_identity: tuple[int, int] | None = None
    published = False
    try:
        parent_identity = _directory_identity(parent, label="provenance directory")
        parent_fd = os.open(str(parent), flags)
        parent_info = os.fstat(parent_fd)
        if (int(parent_info.st_dev), int(parent_info.st_ino)) != parent_identity:
            raise ValueError(f"provenance directory was replaced before publication: {parent}")
        serialized = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False).encode("utf-8") + b"\n"
        for _ in range(16):
            candidate_name = f".{destination.name}.{secrets.token_hex(12)}.tmp"
            try:
                create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):
                    create_flags |= os.O_CLOEXEC
                temporary_fd = os.open(
                    candidate_name,
                    create_flags,
                    0o600,
                    dir_fd=parent_fd,
                )
                temporary_name = candidate_name
                break
            except FileExistsError:
                continue
        if temporary_fd is None or temporary_name is None:
            raise FileExistsError(f"failed to allocate temporary provenance file in {parent}")
        with os.fdopen(temporary_fd, "wb", closefd=True) as stream:
            temporary_fd = None
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            published = True
        except FileExistsError as exc:
            raise FileExistsError(
                f"provenance destination already exists; refusing overwrite: {destination}"
            ) from exc
        parent_after_fd = os.fstat(parent_fd)
        if (int(parent_after_fd.st_dev), int(parent_after_fd.st_ino)) != parent_identity:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
            raise ValueError(f"provenance directory was replaced during publication: {parent}")
        os.fsync(parent_fd)
        if _directory_identity(parent, label="provenance directory") != parent_identity:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
            raise ValueError(f"provenance directory was replaced during publication: {parent}")
    except Exception:
        if published and parent_fd is not None:
            try:
                os.unlink(destination.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if temporary_fd is not None:
            try:
                os.close(temporary_fd)
            except OSError:
                pass
        if temporary_name is not None and parent_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass
    return destination


def read_json_mapping(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """Read a regular, non-symlink JSON object and return its exact bytes."""

    source = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(f"secure {label} reads require O_NOFOLLOW")
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(str(source), flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file: {source}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read {label}: {source}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return dict(payload), raw


def same_path(left: Any, right: str | Path) -> bool:
    if not isinstance(left, (str, os.PathLike)) or not str(left).strip():
        return False
    try:
        return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def python_runtime_version(executable: str | Path) -> str:
    """Return the exact interpreter version without starting model code."""

    requested = Path(executable).resolve(strict=True)
    if requested == Path(sys.executable).resolve(strict=True):
        return sys.version
    try:
        completed = subprocess.run(
            [str(requested), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"failed to query SDM Python version: {requested}") from exc
    version = (completed.stdout or completed.stderr).strip()
    if not version:
        raise ValueError(f"SDM Python returned an empty version: {requested}")
    return version


__all__ = [
    "assert_directory_path_identity",
    "atomic_create_json",
    "atomic_replace_bytes_at_fd",
    "canonical_json_bytes",
    "create_fresh_child_directory",
    "directory_identity",
    "ensure_real_directory",
    "open_or_create_directory",
    "read_regular_bytes_at_fd",
    "read_json_mapping",
    "same_path",
    "same_directory_identity",
    "python_runtime_version",
    "sha256_bytes",
    "sha256_json",
]
