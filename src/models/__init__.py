"""Lazy public exports for the normal and PBR LINO architectures.

Importing the package itself must remain dependency-light: the released
normal module pulls optional Lightning/TorchMetrics packages, while the PBR
module is not needed by the SDM-EXR comparison path.  Public attribute access
retains the original ``LiNo_UniPS`` and ``LiNo_UniPS_PBR`` names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-checker-only imports
    from .Net_module import LiNo_UniPS
    from .Net_pbr_module import LiNo_UniPS as LiNo_UniPS_PBR


__all__ = ["LiNo_UniPS", "LiNo_UniPS_PBR"]


def __getattr__(name: str) -> Any:
    if name == "LiNo_UniPS":
        from .Net_module import LiNo_UniPS as normal_model

        return normal_model
    if name == "LiNo_UniPS_PBR":
        from .Net_pbr_module import LiNo_UniPS as pbr_model

        return pbr_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
