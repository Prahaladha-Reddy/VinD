from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from vinD.pipeline import run_analysis


LOGGER = logging.getLogger("vinD.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vind-analyze",
        description="Run vinD analysis on TRIBE prediction arrays stored in .npz format.",
    )
    parser.add_argument(
        "predictions",
        help="Path to an .npz file containing a 'preds' array.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where report.md and plots/ will be written.",
    )
    parser.add_argument(
        "--hrf-lag",
        type=int,
        default=5,
        help="Hemodynamic lag in seconds used for video alignment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        parser.error(f"Predictions file not found: {predictions_path}")

    with np.load(predictions_path) as data:
        if "preds" not in data:
            parser.error(f"{predictions_path} does not contain a 'preds' array")
        preds = data["preds"]

    LOGGER.info("Loaded predictions with shape %s from %s", preds.shape, predictions_path)
    run_analysis(preds, output_dir=args.output_dir, hrf_lag=args.hrf_lag)
    LOGGER.info("Analysis finished in %s", Path(args.output_dir).resolve())
    return 0
