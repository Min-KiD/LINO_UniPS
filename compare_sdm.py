"""Prepare, seal, and score one GT-hidden SDM comparison run.

This module never imports SDM or launches model inference. It prints the exact
SDM command for the user, then validates signed EXRs and provenance artifacts.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import sys
from pathlib import Path
from typing import Any


def _persist_manifests(
    config: Any,
    manifest: Any,
    *,
    selection_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Persist rich/effective manifests through one pinned policy directory."""

    from src.comparison.manifest import persist_comparison_manifests_at_fd
    from src.comparison.provenance import open_or_create_directory

    policy_fd, policy_identity, policy_path = open_or_create_directory(
        config.policy_root,
        label="comparison policy output directory",
    )
    try:
        return persist_comparison_manifests_at_fd(
            config,
            manifest,
            policy_fd=policy_fd,
            policy_identity=policy_identity,
            policy_path=policy_path,
            selection_source_bytes=selection_source_bytes,
        )
    finally:
        os.close(policy_fd)


def _validate_runtime_inputs(
    sdm_repo: Path,
    checkpoint: Path,
    python_executable: Path,
) -> tuple[Path, Path]:
    repo = Path(sdm_repo)
    if not repo.is_dir():
        raise ValueError(f"SDM repository does not exist or is not a directory: {repo}")
    main_path = repo / "main.py"
    config_path = repo / "configs" / "baseline_optimized_infer.yaml"
    if not main_path.is_file():
        raise ValueError(f"SDM main.py does not exist: {main_path}")
    if not config_path.is_file():
        raise ValueError(f"SDM optimized inference config does not exist: {config_path}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"SDM checkpoint does not exist: {checkpoint_path}")

    python_path = Path(python_executable)
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError(f"SDM Python executable is not executable: {python_path}")
    return main_path, config_path


def _prepare_view(args: argparse.Namespace) -> dict[str, Any]:
    # Keep all comparison/runtime imports lazy so ``compare_sdm --help`` does
    # not require OpenCV, Torch, Lightning, or the released SDM repository.
    from src.comparison.config import load_sdm_exr_config
    from src.comparison.exr_io import sha256_file
    from src.comparison.manifest import build_dataset_manifest
    from src.comparison.provenance import (
        atomic_create_json,
        create_fresh_child_directory,
        directory_identity,
        ensure_real_directory,
        python_runtime_version,
        sha256_json,
    )
    from src.comparison.sdm_view import (
        build_sdm_command,
        prepare_sdm_view,
        validate_sdm_view,
    )

    config = load_sdm_exr_config(args.config)
    config_path = Path(args.config).resolve(strict=True)
    repo = Path(args.sdm_repo).resolve(strict=True)
    checkpoint = Path(args.sdm_checkpoint).resolve(strict=True)
    python_executable = Path(args.sdm_python).resolve(strict=True)
    main_path, optimized_config_path = _validate_runtime_inputs(
        repo, checkpoint, python_executable
    )

    selection_source_bytes: bytes | None = None
    if config.light_selection == "manifest":
        from src.comparison.exr_io import read_file_bytes

        selection_source_bytes = read_file_bytes(
            config.effective_selection_manifest_path,
            label="selection manifest",
        )
    manifest = build_dataset_manifest(
        config,
        selection_manifest_bytes=selection_source_bytes,
    )
    generated_prediction_names: dict[str, str] = {}
    for record in manifest.objects:
        prediction_name = f"{Path(record.name).stem}_pred.exr"
        previous = generated_prediction_names.get(prediction_name)
        if previous is not None and previous != record.name:
            raise ValueError(
                "generated SDM prediction basename collision: "
                f"{previous!r} and {record.name!r} both map to {prediction_name!r}"
            )
        generated_prediction_names[prediction_name] = record.name
    persisted = _persist_manifests(
        config,
        manifest,
        selection_source_bytes=selection_source_bytes,
    )
    view_path = prepare_sdm_view(config, manifest).resolve(strict=True)
    view_attestation = validate_sdm_view(config, manifest)
    request_id = secrets.token_hex(16)
    output_path = create_fresh_child_directory(
        config.sdm_output_dir,
        request_id,
        label="fresh SDM output directory",
    )
    request_dir = Path(config.policy_root).resolve(strict=False) / "sdm_requests"
    completion_dir = Path(config.policy_root).resolve(strict=False) / "sdm_completions"
    request_dir = ensure_real_directory(request_dir, label="SDM request directory")
    completion_dir = ensure_real_directory(completion_dir, label="SDM completion directory")
    request_path = request_dir / f"{request_id}.json"
    completion_path = completion_dir / f"{request_id}.json"
    argv = build_sdm_command(
        config,
        repo,
        checkpoint,
        python_executable,
        output_dir=output_path,
    )

    python_version = python_runtime_version(python_executable)
    fingerprint_inputs: dict[str, Any] = {
        "request_id": request_id,
        "config_sha256": sha256_file(config_path),
        "mask_policy": config.mask_policy,
        "input_manifest_sha256": persisted["input_manifest_sha256"],
        "selection_manifest_sha256": persisted["selection_manifest_sha256"],
        "effective_selection_manifest_sha256": persisted[
            "effective_selection_manifest_sha256"
        ],
        "sdm_main_sha256": sha256_file(main_path),
        "sdm_config_sha256": sha256_file(optimized_config_path),
        "sdm_checkpoint_sha256": sha256_file(checkpoint),
        "sdm_python_sha256": sha256_file(python_executable),
        "sdm_python_version": python_version,
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
        "sdm_output_base_identity": directory_identity(
            config.sdm_output_dir,
            label="SDM output base",
        ),
        "sdm_output_identity": directory_identity(
            output_path,
            label="SDM output directory",
        ),
        "sdm_request_dir_identity": directory_identity(
            request_dir,
            label="SDM request directory",
        ),
        "sdm_completion_dir_identity": directory_identity(
            completion_dir,
            label="SDM completion directory",
        ),
        "argv": argv,
    }

    request: dict[str, Any] = {
        "schema_version": 2,
        "model": "SDM-UniPS",
        "request_id": request_id,
        "request_path": str(request_path),
        "completion_path": str(completion_path),
        "run_fingerprint": sha256_json(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "config_path": str(config_path),
        "config_sha256": fingerprint_inputs["config_sha256"],
        "mask_policy": config.mask_policy,
        "input_manifest_sha256": persisted["input_manifest_sha256"],
        "selection_manifest_sha256": persisted["selection_manifest_sha256"],
        "effective_selection_manifest_sha256": persisted[
            "effective_selection_manifest_sha256"
        ],
        "selection_manifest_path": str(persisted["selection_manifest_path"]),
        "effective_selection_manifest_path": str(
            persisted["effective_selection_manifest_path"]
        ),
        "input_manifest_path": str(persisted["input_manifest_path"]),
        "sdm_repo": str(repo),
        "sdm_main": str(main_path),
        "sdm_main_sha256": fingerprint_inputs["sdm_main_sha256"],
        "sdm_config": str(optimized_config_path),
        "sdm_config_sha256": fingerprint_inputs["sdm_config_sha256"],
        "sdm_checkpoint": str(checkpoint),
        "sdm_checkpoint_sha256": fingerprint_inputs["sdm_checkpoint_sha256"],
        "sdm_python": str(python_executable),
        "sdm_python_sha256": fingerprint_inputs["sdm_python_sha256"],
        "sdm_python_version": python_version,
        "view_attestation": view_attestation,
        "view_tree_sha256": view_attestation["view_tree_sha256"],
        "view_root_identity": view_attestation["root_identity"],
        "view_object_identities": view_attestation["object_identities"],
        "sdm_output_base_identity": fingerprint_inputs["sdm_output_base_identity"],
        "sdm_output_identity": fingerprint_inputs["sdm_output_identity"],
        "sdm_request_dir_identity": fingerprint_inputs["sdm_request_dir_identity"],
        "sdm_completion_dir_identity": fingerprint_inputs["sdm_completion_dir_identity"],
        "view_path": str(view_path),
        "configured_output_path": str(Path(config.sdm_output_dir).resolve(strict=False)),
        "output_path": str(output_path),
        "argv": argv,
    }
    atomic_create_json(request_path, request)

    print(view_path)
    print(shlex.join(argv))
    print(
        shlex.join(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "finalize-sdm",
                "--config",
                str(config_path),
                "--request",
                str(request_path),
            ]
        )
    )
    return request


def _score(args: argparse.Namespace) -> dict[str, Any]:
    # Keep scoring imports lazy so parser/help and prepare-view remain usable
    # without model, Torch, or the released SDM environment.
    from src.comparison.config import load_sdm_exr_config
    from src.comparison.metrics import score_lino_and_sdm

    config_path = Path(args.config).resolve(strict=True)
    config = load_sdm_exr_config(config_path)
    result = score_lino_and_sdm(
        config,
        Path(args.request),
        config_path=config_path,
    )
    macro = result["macro_object"]
    difference = float(result["lino_minus_sdm_mae"])
    print(f"LINO macro MAE: {float(macro['lino']['mae']):.6f}")
    print(f"SDM macro MAE: {float(macro['sdm']['mae']):.6f}")
    print(f"LINO - SDM macro MAE: {difference:.6f}")
    return result


def _finalize_sdm(args: argparse.Namespace) -> dict[str, Any]:
    from src.comparison.config import load_sdm_exr_config
    from src.comparison.metrics import finalize_sdm_run

    config_path = Path(args.config).resolve(strict=True)
    config = load_sdm_exr_config(config_path)
    completion = finalize_sdm_run(
        config,
        Path(args.request),
        config_path=config_path,
    )
    print(completion["completion_path"])
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, finalize, and score a GT-hidden SDM-UniPS comparison."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-view",
        help="Create the GT-hidden view and persist a non-launched SDM run request.",
    )
    prepare.add_argument("--config", required=True, help="LINO SDM-EXR YAML config.")
    prepare.add_argument("--sdm-repo", required=True, help="Local SDM-UniPS repository.")
    prepare.add_argument(
        "--sdm-checkpoint", required=True, help="Local released SDM checkpoint."
    )
    prepare.add_argument(
        "--sdm-python", required=True, help="Python executable used for SDM inference."
    )
    finalize = subparsers.add_parser(
        "finalize-sdm",
        help="Validate one completed SDM run and bind its predictions to its request.",
    )
    finalize.add_argument("--config", required=True, help="LINO SDM-EXR YAML config.")
    finalize.add_argument(
        "--request",
        required=True,
        help="Exact immutable request JSON printed by prepare-view.",
    )
    score = subparsers.add_parser(
        "score",
        help="Score paired LINO and SDM signed normal EXRs at source resolution.",
    )
    score.add_argument("--config", required=True, help="LINO SDM-EXR YAML config.")
    score.add_argument(
        "--request",
        required=True,
        help="Exact finalized SDM request JSON printed by prepare-view.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-view":
        return _prepare_view(args)
    if args.command == "finalize-sdm":
        return _finalize_sdm(args)
    if args.command == "score":
        return _score(args)
    raise ValueError(f"unsupported compare_sdm command: {args.command}")


if __name__ == "__main__":
    main()
