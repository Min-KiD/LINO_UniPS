"""Evaluation entry point for the released LINO model and SDM-EXR runner."""

from __future__ import annotations

import argparse
import os
from typing import Any


def predict_normal(testdata: Any, lino: Any) -> None:
    """Run the unchanged Lightning-based legacy prediction flow.

    Imports stay inside the function so importing ``eval`` for configuration or
    CLI tests does not require the optional Lightning runtime.
    """

    from torch.utils.data import DataLoader
    import pytorch_lightning as pl

    test_loader = DataLoader(testdata, batch_size=1)
    trainer = pl.Trainer(accelerator="auto", devices=1, precision="bf16-mixed")
    trainer.test(model=lino, dataloaders=test_loader)


def run_legacy(args: argparse.Namespace) -> None:
    """Preserve the released ``python eval.py --task_name ...`` behavior."""

    import torch
    from pytorch_lightning import seed_everything

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    seed_everything(seed=args.seed, workers=True)
    lino = torch.hub.load(
        repo_dir,
        "lino_unips",
        source="local",
        pretrained=True,
        task_name=args.task_name,
        ckpt_path=args.ckpt_path,
    )
    testdata = torch.hub.load(
        repo_dir,
        "load_test_data",
        source="local",
        data_root=[args.data_root],
        numofimages=args.num_images,
    )
    predict_normal(testdata, lino)


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible parser plus the YAML config switch."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
        default="DiLiGenT",
        help="Name of the task",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/DiLiGenT/",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="Number of images to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Optional path to a local lino.pth checkpoint. If omitted, LINO_MODEL_URL/default URL is used.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the strict SDM-EXR YAML inference config.",
    )
    return parser


def main(argv: list[str] | None = None) -> Any:
    """Dispatch either the strict config runner or the legacy evaluation path."""

    args = build_parser().parse_args(argv)
    if args.config:
        from src.comparison.config import load_sdm_exr_config
        from src.comparison.inference import run_lino_inference

        return run_lino_inference(
            load_sdm_exr_config(args.config),
            config_path=args.config,
        )
    return run_legacy(args)


if __name__ == "__main__":
    main()
