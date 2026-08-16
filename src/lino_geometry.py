"""Dependency-free runtime geometry contracts for LINO variants."""

from __future__ import annotations


PRIVATE_EXTERNAL_VERSION = "private_external_lino_native_v1"
PRIVATE_256_VERSION = "private_external_lino_256_v2"
PRIVATE_LINO_GEOMETRIES = {
    PRIVATE_EXTERNAL_VERSION: (512, 256),
    PRIVATE_256_VERSION: (256, 128),
}


def private_lino_geometry(version: str) -> tuple[int, int]:
    """Return the exact internal/canonical pair for a private LINO route."""

    try:
        return PRIVATE_LINO_GEOMETRIES[version]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported private preprocessing_version: {version}") from exc


__all__ = [
    "PRIVATE_256_VERSION",
    "PRIVATE_EXTERNAL_VERSION",
    "PRIVATE_LINO_GEOMETRIES",
    "private_lino_geometry",
]
