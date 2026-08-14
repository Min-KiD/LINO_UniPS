"""Strict configuration schema for SDM-style EXR comparison runs.

The comparison runner deliberately keeps all of its knobs in one YAML file.  This
module parses that file without consulting the filesystem for dataset or
checkpoint contents; those runtime checks belong to the inference runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_CONFIG_FIELDS = frozenset(
    {
        "checkpoint",
        "data_root",
        "output_root",
        "object_suffix",
        "image_prefix",
        "image_extension",
        "max_image_num",
        "light_selection",
        "selection_manifest",
        "seed",
        "mask_policy",
        "external_mask_filename",
        "normal_filenames",
        "normal_encoding",
        "mask_margin",
        "max_image_resolution",
        "pixel_samples",
        "precision",
        "device",
        "num_workers",
        "save_exr",
        "save_png",
    }
)
_OPTIONAL_CONFIG_FIELDS = frozenset(
    {"expected_source_geometry", "preprocessing_version"}
)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str) -> int:
    # bool is an int subclass, but accepting true/false for a numeric setting is
    # almost certainly a typo in an inference preset.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True)
class SdmExrInferenceConfig:
    """Validated, immutable settings shared by the comparison runners."""

    checkpoint: Path
    data_root: Path
    output_root: Path
    object_suffix: str
    image_prefix: str
    image_extension: str
    max_image_num: int
    light_selection: str
    selection_manifest: Path | None
    seed: int
    mask_policy: str
    external_mask_filename: str
    normal_filenames: tuple[str, ...]
    normal_encoding: str
    mask_margin: int
    max_image_resolution: int
    pixel_samples: int
    precision: str
    device: str
    num_workers: int
    save_exr: bool
    save_png: bool
    expected_source_geometry: tuple[int, int] | None = None
    preprocessing_version: str = "released_transfer_v1"

    def __post_init__(self) -> None:
        for name in ("checkpoint", "data_root", "output_root"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise ValueError(f"{name} must be a pathlib.Path")
            if not str(value).strip():
                raise ValueError(f"{name} must be a non-empty path")

        for name in (
            "object_suffix",
            "image_prefix",
            "image_extension",
            "external_mask_filename",
        ):
            _require_text(getattr(self, name), name)

        if self.light_selection not in {"seeded", "manifest"}:
            raise ValueError("light_selection must be one of: seeded, manifest")
        if self.mask_policy not in {"external", "full"}:
            raise ValueError("mask_policy must be one of: external, full")
        if self.normal_encoding not in {"signed", "unsigned"}:
            raise ValueError("normal_encoding must be one of: signed, unsigned")
        if self.preprocessing_version not in {
            "released_transfer_v1",
            "private_external_lino_native_v1",
        }:
            raise ValueError(
                "preprocessing_version must be one of: "
                "released_transfer_v1, private_external_lino_native_v1"
            )
        if self.expected_source_geometry is not None:
            if (
                not isinstance(self.expected_source_geometry, tuple)
                or len(self.expected_source_geometry) != 2
                or any(
                    type(value) is not int or value <= 0
                    for value in self.expected_source_geometry
                )
            ):
                raise ValueError(
                    "expected_source_geometry must be null or a positive [height, width] pair"
                )
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of: bf16, fp16, fp32")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")

        for name in ("max_image_num", "pixel_samples"):
            value = _require_int(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in ("mask_margin", "num_workers"):
            value = _require_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        resolution = _require_int(self.max_image_resolution, "max_image_resolution")
        if resolution < 512 or resolution % 512:
            raise ValueError("max_image_resolution must be at least 512 and divisible by 512")
        _require_int(self.seed, "seed")

        if not isinstance(self.normal_filenames, tuple) or not self.normal_filenames:
            raise ValueError("normal_filenames must be a non-empty sequence")
        for filename in self.normal_filenames:
            _require_text(filename, "normal_filenames")

        if self.selection_manifest is not None:
            if not isinstance(self.selection_manifest, Path):
                raise ValueError("selection_manifest must be a pathlib.Path or null")
            if not str(self.selection_manifest).strip():
                raise ValueError("selection_manifest must be a non-empty path")
        if self.light_selection == "manifest" and self.selection_manifest is None:
            raise ValueError("selection_manifest is required when light_selection is manifest")

        _require_bool(self.save_exr, "save_exr")
        _require_bool(self.save_png, "save_png")

    @property
    def policy_root(self) -> Path:
        """Root for all artifacts of the selected mask policy."""

        return self.output_root / self.mask_policy

    @property
    def lino_output_dir(self) -> Path:
        return self.policy_root / "lino"

    @property
    def sdm_view_dir(self) -> Path:
        return self.policy_root / "sdm_input"

    @property
    def sdm_output_dir(self) -> Path:
        return self.policy_root / "sdm"

    @property
    def input_manifest_path(self) -> Path:
        return self.policy_root / "input_manifest.json"

    @property
    def effective_selection_manifest_path(self) -> Path:
        if self.selection_manifest is not None:
            return self.selection_manifest
        return self.policy_root / "selected_lights.json"

    @property
    def provenance_path(self) -> Path:
        return self.lino_output_dir / "run.json"


def _as_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string")
    return Path(value)


def _as_normal_filenames(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("normal_filenames must be a non-empty sequence")
    filenames = tuple(value)
    if not filenames:
        raise ValueError("normal_filenames must be a non-empty sequence")
    for filename in filenames:
        _require_text(filename, "normal_filenames")
    return filenames


def _as_optional_geometry(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "expected_source_geometry must be null or a positive [height, width] pair"
        )
    height = _require_int(value[0], "expected_source_geometry height")
    width = _require_int(value[1], "expected_source_geometry width")
    if height <= 0 or width <= 0:
        raise ValueError("expected_source_geometry values must be positive")
    return (height, width)


def load_sdm_exr_config(path: str | Path) -> SdmExrInferenceConfig:
    """Load and validate a complete SDM-EXR YAML preset.

    Only structural validation is performed here.  In particular, checkpoint
    and dataset paths are intentionally allowed not to exist so the preset can
    be inspected on machines without the large runtime assets.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("configuration must be a YAML mapping")

    raw_keys = list(raw)
    non_string_keys = [key for key in raw_keys if not isinstance(key, str)]
    if non_string_keys:
        invalid = ", ".join(repr(key) for key in non_string_keys)
        raise ValueError(f"configuration keys must be strings; invalid key(s): {invalid}")

    unknown = sorted(set(raw_keys) - (_CONFIG_FIELDS | _OPTIONAL_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"unknown configuration key(s): {', '.join(map(str, unknown))}")
    missing = sorted(_CONFIG_FIELDS - set(raw))
    if missing:
        raise ValueError(f"missing required configuration key(s): {', '.join(missing)}")

    selection_manifest_raw = raw["selection_manifest"]
    if selection_manifest_raw is None:
        selection_manifest = None
    elif isinstance(selection_manifest_raw, str) and not selection_manifest_raw.strip():
        selection_manifest = None
    else:
        selection_manifest = _as_path(selection_manifest_raw, "selection_manifest")

    return SdmExrInferenceConfig(
        checkpoint=_as_path(raw["checkpoint"], "checkpoint"),
        data_root=_as_path(raw["data_root"], "data_root"),
        output_root=_as_path(raw["output_root"], "output_root"),
        object_suffix=_require_text(raw["object_suffix"], "object_suffix"),
        image_prefix=_require_text(raw["image_prefix"], "image_prefix"),
        image_extension=_require_text(raw["image_extension"], "image_extension"),
        max_image_num=_require_int(raw["max_image_num"], "max_image_num"),
        light_selection=_require_text(raw["light_selection"], "light_selection"),
        selection_manifest=selection_manifest,
        seed=_require_int(raw["seed"], "seed"),
        mask_policy=_require_text(raw["mask_policy"], "mask_policy"),
        external_mask_filename=_require_text(
            raw["external_mask_filename"], "external_mask_filename"
        ),
        normal_filenames=_as_normal_filenames(raw["normal_filenames"]),
        normal_encoding=_require_text(raw["normal_encoding"], "normal_encoding"),
        expected_source_geometry=_as_optional_geometry(raw.get("expected_source_geometry")),
        preprocessing_version=_require_text(
            raw.get("preprocessing_version", "released_transfer_v1"),
            "preprocessing_version",
        ),
        mask_margin=_require_int(raw["mask_margin"], "mask_margin"),
        max_image_resolution=_require_int(
            raw["max_image_resolution"], "max_image_resolution"
        ),
        pixel_samples=_require_int(raw["pixel_samples"], "pixel_samples"),
        precision=_require_text(raw["precision"], "precision"),
        device=_require_text(raw["device"], "device"),
        num_workers=_require_int(raw["num_workers"], "num_workers"),
        save_exr=_require_bool(raw["save_exr"], "save_exr"),
        save_png=_require_bool(raw["save_png"], "save_png"),
    )


__all__ = ["SdmExrInferenceConfig", "load_sdm_exr_config"]
