from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import modal
import numpy as np


LOGGER = logging.getLogger("vinD.run_tribe")
DEFAULT_MODAL_APP_NAME = "tribe-v2"
DEFAULT_MODAL_CLASS_NAME = "TribePredictor"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vind-tribe",
        description="Call a deployed Modal TRIBE predictor, then run local vinD analysis.",
    )
    parser.add_argument("video_url", help="Video URL supported by the deployed TRIBE worker.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for the downloaded predictions and vinD outputs.",
    )
    parser.add_argument(
        "--modal-app-name",
        default=os.getenv("VIND_MODAL_APP_NAME", DEFAULT_MODAL_APP_NAME),
        help="Deployed Modal app name. Defaults to env VIND_MODAL_APP_NAME or tribe-v2.",
    )
    parser.add_argument(
        "--modal-class-name",
        default=os.getenv("VIND_MODAL_CLASS_NAME", DEFAULT_MODAL_CLASS_NAME),
        help="Deployed Modal class name. Defaults to env VIND_MODAL_CLASS_NAME or TribePredictor.",
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

    LOGGER.info(
        "Requesting predictions from Modal app=%s class=%s",
        args.modal_app_name,
        args.modal_class_name,
    )

    predictor_cls = modal.Cls.from_name(args.modal_app_name, args.modal_class_name)
    result = predictor_cls().predict.remote(args.video_url)

    if "preds_bytes" not in result or "shape" not in result:
        raise RuntimeError("Modal response is missing required prediction payload fields")

    preds = np.frombuffer(result["preds_bytes"], dtype=np.float32).reshape(result["shape"])
    LOGGER.info("Received prediction array with shape %s", preds.shape)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "tribe_preds.npz"
    np.savez_compressed(npz_path, preds=preds)
    LOGGER.info("Saved predictions to %s", npz_path)

    from vinD import run_analysis

    run_analysis(preds, segments=result.get("segments"), output_dir=output_dir)
    LOGGER.info("Analysis complete in %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
