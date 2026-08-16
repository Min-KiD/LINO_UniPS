"""Strict, immutable configuration for private LINO EXR training."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, cast

import yaml

from src.lino_geometry import private_lino_geometry


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _safe_basename(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe basename")
    if "\x00" in value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must be a safe basename")
    if Path(value).name != value or Path(value).is_absolute():
        raise ValueError(f"{field_name} must be a safe basename")
    return value


def _path_value(value: object, field_name: str, *, nullable: bool = False) -> Path | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or null" if nullable else ""
        raise ValueError(f"{field_name} must be a non-empty path{suffix}")
    if "\x00" in value:
        raise ValueError(f"{field_name} must be a valid path")
    return Path(value)


def _sequence_value(value: object, field_name: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list")
    return list(value)


def _int_value(value: object, field_name: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _float_value(value: object, field_name: str) -> float:
    if not _is_real(value):
        raise ValueError(f"{field_name} must be a number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _bool_value(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


@dataclass(frozen=True)
class SourceValidationConfig:
    mode: str = "lazy"
    structural_index_version: str = "private_exr_index_v1"
    persistent_content_ledger: bool = False
    progress_every_objects: int = 100

    def __post_init__(self) -> None:
        if self.mode != "lazy":
            raise ValueError("private training supports lazy source validation only")
        if self.structural_index_version != "private_exr_index_v1":
            raise ValueError("unsupported structural index version")
        if self.persistent_content_ledger:
            raise ValueError("persistent content ledger is not supported")
        if not _is_int(self.progress_every_objects) or self.progress_every_objects < 0:
            raise ValueError("progress_every_objects must be a non-negative integer")


@dataclass(frozen=True)
class PrivateTrainConfig:
    train_dir: Path
    test_dir: Path
    save_dir: Path
    startup_mode: str
    init_checkpoint: Path | None
    resume_checkpoint: Path | None
    final_selection_manifest: Path | None
    object_suffix: str
    image_prefix: str
    image_extension: str
    normal_filenames: tuple[str, ...]
    external_mask_filename: str
    normal_encoding: str
    expected_source_geometry: tuple[int, int]
    mask_policy: str
    mask_margin: int
    max_image_num: int
    light_selection: str
    seed: int
    preprocessing_version: str
    max_image_resolution: int
    canonical_resolution: int
    pixel_samples: int
    train_pixel_budget: int
    precision: str
    device: str
    deterministic: bool
    epochs: int
    train_batch_size: int
    train_workers: int
    test_workers: int
    learning_rate: float
    weight_decay: float
    adamw_betas: tuple[float, float]
    scheduler_step_size: int
    scheduler_gamma: float
    save_every_epochs: int
    keep_milestone_epochs: tuple[int, ...]
    activation_checkpointing: bool
    train_log_every_batches: int = 10
    source_validation: SourceValidationConfig = field(
        default_factory=SourceValidationConfig
    )

    def __post_init__(self) -> None:
        path_fields = (
            "train_dir",
            "test_dir",
            "save_dir",
        )
        for field_name in path_fields:
            if not isinstance(getattr(self, field_name), Path):
                raise ValueError(f"{field_name} must be a path")
        for field_name in (
            "init_checkpoint",
            "resume_checkpoint",
            "final_selection_manifest",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Path):
                raise ValueError(f"{field_name} must be a path or null")

        if self.startup_mode not in {"cold_start", "init_checkpoint", "resume"}:
            raise ValueError("startup_mode must be one of: cold_start, init_checkpoint, resume")
        if self.startup_mode == "cold_start":
            if self.init_checkpoint is not None or self.resume_checkpoint is not None:
                raise ValueError("cold_start cannot configure init_checkpoint or resume_checkpoint")
        elif self.startup_mode == "init_checkpoint":
            if self.init_checkpoint is None:
                raise ValueError("init_checkpoint startup requires init_checkpoint")
            if self.resume_checkpoint is not None:
                raise ValueError("init_checkpoint cannot configure resume_checkpoint")
            if self.init_checkpoint.suffix != ".pth":
                raise ValueError("init_checkpoint must have .pth suffix")
        elif self.resume_checkpoint is None:
            raise ValueError("resume startup requires resume_checkpoint")
        elif self.resume_checkpoint.suffix != ".ckpt":
            raise ValueError("resume_checkpoint must have .ckpt suffix")
        if self.startup_mode == "resume" and self.init_checkpoint is not None:
            raise ValueError("resume cannot configure init_checkpoint")

        if self.train_dir.resolve(False) == self.test_dir.resolve(False):
            raise ValueError("train_dir and test_dir must resolve to different roots")
        if self.object_suffix != ".data":
            raise ValueError("object_suffix must be .data")
        if self.image_prefix != "image":
            raise ValueError("image_prefix must be image")
        if self.image_extension != ".exr":
            raise ValueError("image_extension must be .exr")
        if self.normal_filenames != ("local_normal.exr",):
            raise ValueError("normal_filenames must be (local_normal.exr,)")
        if self.external_mask_filename != "binary_mask.exr":
            raise ValueError("external_mask_filename must be binary_mask.exr")
        for field_name, filename in (
            ("object_suffix", self.object_suffix),
            ("image_prefix", self.image_prefix),
            ("image_extension", self.image_extension),
            ("external_mask_filename", self.external_mask_filename),
        ):
            _safe_basename(filename, field_name)
        if not isinstance(self.normal_filenames, tuple):
            raise ValueError("normal_filenames must be a tuple")
        for filename in self.normal_filenames:
            _safe_basename(filename, "normal_filenames")

        if self.normal_encoding != "unsigned":
            raise ValueError("private LINO training requires normal_encoding unsigned")
        if self.expected_source_geometry != (256, 256):
            raise ValueError("private source geometry must be [256, 256]")
        if self.mask_policy != "external":
            raise ValueError("private LINO training requires mask_policy external")
        if self.max_image_num != 6:
            raise ValueError("max_image_num must be 6")
        if self.light_selection != "seeded":
            raise ValueError("light_selection must be seeded")
        try:
            expected_geometry = private_lino_geometry(self.preprocessing_version)
        except ValueError as exc:
            raise ValueError(
                "preprocessing_version must be private_external_lino_native_v1 "
                "or private_external_lino_256_v2"
            ) from exc
        if (self.max_image_resolution, self.canonical_resolution) != expected_geometry:
            raise ValueError(
                "private LINO geometry must match preprocessing_version: "
                f"internal {expected_geometry[0]} and canonical {expected_geometry[1]}"
            )

        nonnegative_ints = ("mask_margin", "seed", "train_workers", "test_workers")
        for field_name in nonnegative_ints:
            value = getattr(self, field_name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        positive_ints = (
            "max_image_num",
            "max_image_resolution",
            "canonical_resolution",
            "pixel_samples",
            "train_pixel_budget",
            "epochs",
            "train_batch_size",
            "scheduler_step_size",
            "save_every_epochs",
        )
        for field_name in positive_ints:
            value = getattr(self, field_name)
            if not _is_int(value) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if self.precision == "bf16" and self.device == "cpu":
            raise ValueError("precision bf16 requires CUDA; device cpu is unsupported")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("device must be cuda, cpu, or auto")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if not isinstance(self.activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be a boolean")
        if (
            not _is_int(self.train_log_every_batches)
            or self.train_log_every_batches < 0
        ):
            raise ValueError("train_log_every_batches must be a nonnegative integer")

        for field_name, minimum, strict in (
            ("learning_rate", 0.0, True),
            ("weight_decay", 0.0, False),
            ("scheduler_gamma", 0.0, True),
        ):
            value = getattr(self, field_name)
            if not _is_real(value) or isinstance(value, bool) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if (strict and value <= minimum) or (not strict and value < minimum):
                comparator = "positive" if strict else "nonnegative"
                raise ValueError(f"{field_name} must be {comparator}")
        if not isinstance(self.adamw_betas, tuple) or len(self.adamw_betas) != 2:
            raise ValueError("adamw_betas must contain exactly two values")
        for beta in self.adamw_betas:
            if not _is_real(beta) or isinstance(beta, bool) or not isfinite(float(beta)):
                raise ValueError("adamw_betas must be finite")
            if not 0 <= beta < 1:
                raise ValueError("adamw_betas must be in [0, 1)")
        if not isinstance(self.keep_milestone_epochs, tuple):
            raise ValueError("keep_milestone_epochs must be a tuple")
        if any(not _is_int(epoch) or epoch <= 0 for epoch in self.keep_milestone_epochs):
            raise ValueError("keep_milestone_epochs must contain positive integers")
        if tuple(sorted(set(self.keep_milestone_epochs))) != self.keep_milestone_epochs:
            raise ValueError("keep_milestone_epochs must be unique and sorted")
        if not isinstance(self.source_validation, SourceValidationConfig):
            raise ValueError("source_validation must be a SourceValidationConfig")


_CONFIG_FIELDS = tuple(field.name for field in fields(PrivateTrainConfig))
_OPTIONAL_CONFIG_FIELDS = frozenset({"train_log_every_batches"})


def _raw_mapping(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("private training config must be a YAML mapping")
    invalid_keys = [key for key in raw if not isinstance(key, str)]
    if invalid_keys:
        invalid = ", ".join(repr(key) for key in invalid_keys)
        raise ValueError(
            "private training config keys must be strings; "
            f"invalid key(s): {invalid}"
        )
    return dict(raw)


def _string_value(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _normal_filenames(value: object) -> tuple[str, ...]:
    filenames = _sequence_value(value, "normal_filenames")
    return tuple(_safe_basename(filename, "normal_filenames") for filename in filenames)


def _geometry(value: object) -> tuple[int, int]:
    geometry = _sequence_value(value, "expected_source_geometry")
    if len(geometry) != 2:
        raise ValueError("expected_source_geometry must contain two integers")
    result = tuple(_int_value(item, "expected_source_geometry") for item in geometry)
    return cast(tuple[int, int], result)


def _milestones(value: object) -> tuple[int, ...]:
    milestones = _sequence_value(value, "keep_milestone_epochs")
    return tuple(_int_value(item, "keep_milestone_epochs") for item in milestones)


def _source_validation(value: object) -> SourceValidationConfig:
    mapping = _raw_mapping(value)
    expected = {field.name for field in fields(SourceValidationConfig)}
    unknown = sorted(set(mapping) - expected)
    if unknown:
        raise ValueError(
            "unknown source_validation keys: " + ", ".join(unknown)
        )
    missing = sorted(expected - set(mapping))
    if missing:
        raise ValueError(
            "missing source_validation keys: " + ", ".join(missing)
        )
    return SourceValidationConfig(
        mode=_string_value(mapping["mode"], "source_validation.mode"),
        structural_index_version=_string_value(
            mapping["structural_index_version"],
            "source_validation.structural_index_version",
        ),
        persistent_content_ledger=_bool_value(
            mapping["persistent_content_ledger"],
            "source_validation.persistent_content_ledger",
        ),
        progress_every_objects=mapping["progress_every_objects"],  # type: ignore[arg-type]
    )


def load_private_train_config(path: str | Path) -> PrivateTrainConfig:
    """Load and validate a complete private LINO training YAML mapping."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read private training config: {config_path}") from exc
    values = _raw_mapping(raw)
    unknown = sorted(set(values) - set(_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"unknown private training config keys: {', '.join(unknown)}")
    missing = [
        field_name
        for field_name in _CONFIG_FIELDS
        if field_name not in values and field_name not in _OPTIONAL_CONFIG_FIELDS
    ]
    if missing:
        raise ValueError(f"missing private training config keys: {', '.join(missing)}")

    def path_field(field_name: str, *, nullable: bool = False) -> Path | None:
        return _path_value(values[field_name], field_name, nullable=nullable)

    def int_field(field_name: str) -> int:
        return _int_value(values[field_name], field_name)

    return PrivateTrainConfig(
        train_dir=path_field("train_dir"),  # type: ignore[arg-type]
        test_dir=path_field("test_dir"),  # type: ignore[arg-type]
        save_dir=path_field("save_dir"),  # type: ignore[arg-type]
        startup_mode=_string_value(values["startup_mode"], "startup_mode"),
        init_checkpoint=path_field("init_checkpoint", nullable=True),
        resume_checkpoint=path_field("resume_checkpoint", nullable=True),
        final_selection_manifest=path_field("final_selection_manifest", nullable=True),
        object_suffix=_string_value(values["object_suffix"], "object_suffix"),
        image_prefix=_string_value(values["image_prefix"], "image_prefix"),
        image_extension=_string_value(values["image_extension"], "image_extension"),
        normal_filenames=_normal_filenames(values["normal_filenames"]),
        external_mask_filename=_string_value(
            values["external_mask_filename"], "external_mask_filename"
        ),
        normal_encoding=_string_value(values["normal_encoding"], "normal_encoding"),
        expected_source_geometry=_geometry(values["expected_source_geometry"]),
        mask_policy=_string_value(values["mask_policy"], "mask_policy"),
        mask_margin=int_field("mask_margin"),
        max_image_num=int_field("max_image_num"),
        light_selection=_string_value(values["light_selection"], "light_selection"),
        seed=int_field("seed"),
        preprocessing_version=_string_value(
            values["preprocessing_version"], "preprocessing_version"
        ),
        max_image_resolution=int_field("max_image_resolution"),
        canonical_resolution=int_field("canonical_resolution"),
        pixel_samples=int_field("pixel_samples"),
        train_pixel_budget=int_field("train_pixel_budget"),
        precision=_string_value(values["precision"], "precision"),
        device=_string_value(values["device"], "device"),
        deterministic=_bool_value(values["deterministic"], "deterministic"),
        epochs=int_field("epochs"),
        train_batch_size=int_field("train_batch_size"),
        train_workers=int_field("train_workers"),
        test_workers=int_field("test_workers"),
        learning_rate=_float_value(values["learning_rate"], "learning_rate"),
        weight_decay=_float_value(values["weight_decay"], "weight_decay"),
        adamw_betas=tuple(
            _float_value(item, "adamw_betas")
            for item in _sequence_value(values["adamw_betas"], "adamw_betas")
        ),  # type: ignore[arg-type]
        scheduler_step_size=int_field("scheduler_step_size"),
        scheduler_gamma=_float_value(values["scheduler_gamma"], "scheduler_gamma"),
        save_every_epochs=int_field("save_every_epochs"),
        keep_milestone_epochs=_milestones(values["keep_milestone_epochs"]),
        activation_checkpointing=_bool_value(
            values["activation_checkpointing"], "activation_checkpointing"
        ),
        train_log_every_batches=_int_value(
            values.get("train_log_every_batches", 10),
            "train_log_every_batches",
        ),
        source_validation=_source_validation(values["source_validation"]),
    )


def _resolved_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_resolved_value(item) for item in value]
    if isinstance(value, list):
        return [_resolved_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _resolved_value(getattr(value, item.name))
            for item in fields(value)
        }
    return value


def resolved_config_dict(config: PrivateTrainConfig) -> dict[str, object]:
    """Serialize a validated config in dataclass field order."""

    if not isinstance(config, PrivateTrainConfig):
        raise TypeError("config must be a PrivateTrainConfig")
    return {
        field.name: _resolved_value(getattr(config, field.name))
        for field in fields(PrivateTrainConfig)
    }
